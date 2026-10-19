"""Deterministic race-strategy simulation tools (pure Python/NumPy, no agents here)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import DEFAULT_CONFIG, ModelConfig
from .models import DriverState, PitStop, RaceState, SimulationResult, Strategy


# Plain-language names for constraint codes (used in reasons shown to people).
RULE_NAMES = {"STINT_TOO_LONG": "tyres too old", "TWO_COMPOUND_RULE": "two-compound rule",
              "COMPOUND_UNAVAILABLE": "no tyre sets left", "PIT_LANE_CLOSED": "pit lane closed",
              "PIT_OUTSIDE_RACE": "stop outside the race", "SLICKS_ON_WET": "slick tyres on a wet track",
              "TOO_MANY_STOPS": "too many stops", "DUPLICATE_STOP": "two stops on one lap",
              "UNKNOWN_COMPOUND": "unknown tyre type", "NUMERICAL_INTEGRITY": "failed number checks"}


def effective_max_stint(compound: str, deg_multiplier: float, cfg: ModelConfig = DEFAULT_CONFIG) -> int:
    """Safe tyre life shrinks in proportion to the degradation multiplier."""
    return int(math.floor(cfg.compounds[compound].max_stint_laps / max(deg_multiplier, 1e-9)))


def tyre_schedule(state: RaceState, driver: DriverState, stops: Sequence[PitStop]) -> List[Dict]:
    """Compound and start-of-lap tyre age for every remaining lap."""
    stop_map = {s.lap: s.compound for s in stops}
    compound, age = driver.compound, driver.tyre_age_laps
    rows = []
    for lap in state.laps():
        rows.append({"lap": lap, "compound": compound, "tyre_age": age,
                     "pit": lap in stop_map})
        if lap in stop_map:
            compound, age = stop_map[lap], 0
        else:
            age += 1
    return rows


def stints(state: RaceState, driver: DriverState, stops: Sequence[PitStop]) -> List[Dict]:
    """Stints over the remaining race; end_age = laps completed on that set at stint end."""
    out = []
    start_lap, compound, start_age = state.first_lap, driver.compound, driver.tyre_age_laps
    for s in sorted(stops, key=lambda x: x.lap):
        out.append({"compound": compound, "from_lap": start_lap, "to_lap": s.lap,
                    "start_age": start_age, "end_age": start_age + (s.lap - start_lap + 1)})
        start_lap, compound, start_age = s.lap + 1, s.compound, 0
    out.append({"compound": compound, "from_lap": start_lap, "to_lap": state.total_laps,
                "start_age": start_age,
                "end_age": start_age + (state.total_laps - start_lap + 1)})
    return out


def check_strategy_constraints(state: RaceState, driver: DriverState, strategy: Strategy,
                               cfg: ModelConfig = DEFAULT_CONFIG) -> List[Dict[str, str]]:
    """Feasibility rules. Returns violations as {code, message}; empty = feasible."""
    v: List[Dict[str, str]] = []
    laps = [s.lap for s in strategy.stops]
    if len(set(laps)) != len(laps):
        v.append({"code": "DUPLICATE_STOP", "message": "Two stops are planned on the same lap."})
    if len(strategy.stops) > cfg.max_additional_stops:
        v.append({"code": "TOO_MANY_STOPS",
                  "message": f"{len(strategy.stops)} more stops is over the limit of {cfg.max_additional_stops}."})
    for s in strategy.stops:
        if s.compound not in cfg.compounds:
            v.append({"code": "UNKNOWN_COMPOUND", "message": f"Unknown tyre type '{s.compound}'."})
        if not state.first_lap <= s.lap <= state.total_laps - 1:
            v.append({"code": "PIT_OUTSIDE_RACE",
                      "message": f"A stop on lap {s.lap} is outside the race window "
                                 f"(laps {state.first_lap}–{state.total_laps - 1})."})
        if s.lap in state.pit_closed_laps:
            v.append({"code": "PIT_LANE_CLOSED", "message": f"The pit lane is closed on lap {s.lap}."})
    if any(x["code"] in ("UNKNOWN_COMPOUND", "PIT_OUTSIDE_RACE", "DUPLICATE_STOP") for x in v):
        return v

    need: Dict[str, int] = {}
    for s in strategy.stops:
        need[s.compound] = need.get(s.compound, 0) + 1
    for comp, n in need.items():
        have = driver.available_sets.get(comp, 0)
        if n > have:
            v.append({"code": "COMPOUND_UNAVAILABLE",
                      "message": f"Needs {n} new set{'s' if n > 1 else ''} of {comp.title()} tyres, "
                                 f"but only {have} {'is' if have == 1 else 'are'} left."})

    for i, st in enumerate(stints(state, driver, strategy.stops)):
        limit = effective_max_stint(st["compound"], state.deg_multiplier, cfg)
        if i == 0:
            # Tyres already past their limit can't be un-worn: boxing at the first chance is allowed.
            limit = max(limit, st["start_age"] + 1)
        if st["end_age"] > limit:
            v.append({"code": "STINT_TOO_LONG",
                      "message": f"{st['compound'].title()} tyres would be {st['end_age']} laps old by "
                                 f"lap {st['to_lap']}; their safe limit is {limit} laps at "
                                 f"{state.deg_multiplier:g}× wear."})

    if cfg.require_two_dry_compounds:
        used = set(driver.compounds_used) | {driver.compound} | {s.compound for s in strategy.stops}
        wet_used = any(not cfg.compounds[c].is_dry for c in used)
        dry_used = {c for c in used if cfg.compounds[c].is_dry}
        if not wet_used and len(dry_used) < 2:
            v.append({"code": "TWO_COMPOUND_RULE",
                      "message": "Breaks the two-compound rule: a dry race needs two different dry "
                                 f"tyre types, but only {', '.join(c.title() for c in sorted(dry_used))} "
                                 "would be used."})

    slick_wet = sum(1 for r in tyre_schedule(state, driver, strategy.stops)
                    if state.condition_on_lap(r["lap"]) == "wet" and cfg.compounds[r["compound"]].is_dry)
    if slick_wet > cfg.max_slick_laps_on_wet:
        v.append({"code": "SLICKS_ON_WET",
                  "message": f"{slick_wet} laps on slick tyres while the track is wet "
                             f"(unsafe above {cfg.max_slick_laps_on_wet})."})
    return v


def curve_tyre_adj(curve: Sequence[float], age: float, multiplier: float = 1.0) -> float:
    """Tyre effect from an ML curve: the age-0 offset plus wear (scaled by the wear multiplier);
    past the end of the curve it continues with the last lap's slope."""
    last = len(curve) - 1
    a = max(0.0, float(age))
    if a <= last:
        lo = int(math.floor(a))
        hi = min(lo + 1, last)
        val = curve[lo] + (curve[hi] - curve[lo]) * (a - lo)
    else:
        val = curve[last] + (curve[last] - curve[last - 1]) * (a - last)
    return curve[0] + (val - curve[0]) * multiplier


def tyre_adj(state: RaceState, compound: str, age: float, cfg: ModelConfig = DEFAULT_CONFIG,
             multiplier: Optional[float] = None) -> float:
    """Seconds slower than a new SOFT: ML curve when one is loaded for this compound, else the formula."""
    m = state.deg_multiplier if multiplier is None else multiplier
    curve = state.tyre_curves.get(compound) if state.tyre_curves else None
    if curve:
        return curve_tyre_adj(curve, age, m)
    cp = cfg.compounds[compound]
    return (cp.pace_offset_s + cp.deg_s_per_lap * m * age
            + cp.cliff_s_per_lap * m * max(0, age - cp.cliff_age_laps))


def lap_components(state: RaceState, driver: DriverState, lap: int, compound: str, age: int,
                   cfg: ModelConfig = DEFAULT_CONFIG, traffic_s: float = 0.0) -> Dict[str, float]:
    tyre_adj_s = tyre_adj(state, compound, age, cfg)
    cond = state.condition_on_lap(lap)
    weather_adj = cfg.track_condition_penalty_s[cond] + cfg.tyre_weather_penalty_s[cond][compound]
    kind = state.neutral_kind(lap)          # 'SC', 'VSC' or None
    sc = kind is not None
    traffic = 0.0 if sc else traffic_s
    fuel_adj = state.progress_effect(lap)
    green = driver.baseline_pace_s + tyre_adj_s + traffic + weather_adj + fuel_adj
    factor = cfg.sc_lap_time_factor if kind == "SC" else cfg.vsc_lap_time_factor
    sc_adj = max(0.0, driver.baseline_pace_s * factor - green) if sc else 0.0
    return {"baseline_s": driver.baseline_pace_s, "tyre_adj_s": tyre_adj_s, "traffic_adj_s": traffic,
            "fuel_adj_s": fuel_adj,
            "weather_adj_s": weather_adj, "sc_adj_s": sc_adj, "lap_time_s": green + sc_adj,
            "condition": cond, "safety_car": sc, "neutral": kind}


def pit_loss_on_lap(state: RaceState, lap: int, cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    """Pit-stop loss for a stop at the end of `lap` (cheaper under a Safety Car or VSC)."""
    kind = state.neutral_kind(lap)
    if kind == "SC":
        return state.pit_loss_s * cfg.sc_pit_loss_factor
    if kind == "VSC":
        return state.pit_loss_s * cfg.vsc_pit_loss_factor
    return state.pit_loss_s


def simulate_strategy(state: RaceState, driver: DriverState, strategy: Strategy,
                      cfg: ModelConfig = DEFAULT_CONFIG,
                      rival_traces: Optional[Dict[str, Dict[int, float]]] = None) -> SimulationResult:
    """Simulate the remaining race lap by lap for one strategy.

    rival_traces: {code: {lap: cumulative time at end of lap on the common reference}}
    used only for the optional traffic adjustment after a stop.
    """
    stop_map = {s.lap: s.compound for s in strategy.stops}
    compound, age = driver.compound, driver.tyre_age_laps
    cumulative = driver.gap_to_leader_s
    traffic_left, traffic_count = 0, 0
    rows: List[Dict] = []
    for lap in state.laps():
        traffic_s = cfg.traffic_penalty_s if (cfg.traffic_enabled and traffic_left > 0) else 0.0
        comp = lap_components(state, driver, lap, compound, age, cfg, traffic_s)
        if traffic_left > 0:
            traffic_left -= 1
            if comp["traffic_adj_s"] > 0:
                traffic_count += 1
        pit_loss = 0.0
        if lap in stop_map:
            pit_loss = pit_loss_on_lap(state, lap, cfg)
        cumulative += comp["lap_time_s"] + pit_loss
        rows.append({"lap": lap, "compound": compound, "tyre_age": age, **comp,
                     "pit_loss_s": pit_loss, "cumulative_ref_s": cumulative})
        if lap in stop_map:
            compound, age = stop_map[lap], 0
            if cfg.traffic_enabled and rival_traces:
                gaps = [cumulative - tr[lap] for tr in rival_traces.values() if lap in tr]
                if any(0.0 < g <= cfg.traffic_gap_threshold_s for g in gaps):
                    traffic_left = cfg.traffic_laps
        else:
            age += 1
    lap_total = float(np.sum([r["lap_time_s"] for r in rows]))
    pit_total = float(np.sum([r["pit_loss_s"] for r in rows]))
    return SimulationResult(
        strategy_id=strategy.id, driver_code=driver.code, laps=rows,
        lap_time_total_s=lap_total, pit_loss_total_s=pit_total,
        pit_loss_count=sum(1 for r in rows if r["pit_loss_s"] > 0),
        remaining_time_s=lap_total + pit_total,
        finish_time_ref_s=driver.gap_to_leader_s + lap_total + pit_total,
        stints=stints(state, driver, strategy.stops), traffic_laps=traffic_count)


def rank_key(remaining_time_s: float, strategy: Strategy):
    """Deterministic objective + tie-break: time (ms), fewer stops, later first stop, id."""
    first = strategy.first_pit_lap if strategy.first_pit_lap is not None else 10_000
    return (round(remaining_time_s, 3), len(strategy.stops), -first, strategy.id)


def project_positions(target_finish_ref_s: float, target_code: str,
                      rival_finish: Dict[str, float], current_gaps: Dict[str, float]) -> Dict:
    """Order the simulated group by projected finish time on the common reference."""
    times = dict(rival_finish)
    times[target_code] = target_finish_ref_s
    order = sorted(times, key=lambda c: (round(times[c], 3), current_gaps.get(c, 0.0), c))
    pos = order.index(target_code) + 1
    ahead = order[pos - 2] if pos > 1 else None
    behind = order[pos] if pos < len(order) else None
    return {
        "position_in_group": pos, "group_size": len(order), "order": order,
        "gap_to_car_ahead_s": round(target_finish_ref_s - times[ahead], 3) if ahead else None,
        "car_ahead": ahead, "car_behind": behind,
        "gap_to_car_behind_s": round(times[behind] - target_finish_ref_s, 3) if behind else None,
    }

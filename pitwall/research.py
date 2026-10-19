"""Race-research tools: deterministic analyses of the race state (no forecasting beyond inputs)."""
from __future__ import annotations

from typing import Dict

from .config import DEFAULT_CONFIG, ModelConfig
from .models import RaceState
from .simulation import effective_max_stint


def analyze_track_status(state: RaceState, cfg: ModelConfig = DEFAULT_CONFIG) -> Dict:
    sc_active = state.sc_on_lap(state.first_lap)
    periods = state.neutral_ahead()
    sc_upcoming = any(p["start"] > state.first_lap for p in periods)
    lines = [f"Lap {state.completed_laps} of {state.total_laps} is done, so {state.remaining_laps} laps "
             f"remain (laps {state.first_lap}–{state.total_laps})."]
    for p in periods:
        name, f = ("Safety Car", cfg.sc_pit_loss_factor) if p["kind"] == "SC" else ("Virtual Safety Car",
                                                                                  cfg.vsc_pit_loss_factor)
        lines.append(f"{name} on laps {p['start']}–{p['end']}: a stop then costs "
                     f"{state.pit_loss_s * f:.1f} s instead of {state.pit_loss_s:.1f} s.")
    if not periods:
        lines.append(f"No Safety Car or VSC expected; a normal stop costs {state.pit_loss_s:.1f} s.")
    if state.pit_closed_laps:
        lines.append(f"Pit lane closed on laps {', '.join(map(str, state.pit_closed_laps))}.")
    return {"remaining_laps": state.remaining_laps, "first_lap": state.first_lap,
            "sc_active_next_lap": sc_active, "sc_upcoming": sc_upcoming,
            "sc_window": [state.sc_start_lap, state.sc_end_lap] if state.sc_start_lap else None,
            "neutral_periods": periods,
            "pit_loss_green_s": state.pit_loss_s,
            "pit_loss_sc_s": round(state.pit_loss_s * cfg.sc_pit_loss_factor, 3),
            "pit_closed_laps": state.pit_closed_laps, "summary": lines}


def analyze_tyres(state: RaceState, cfg: ModelConfig = DEFAULT_CONFIG) -> Dict:
    t = state.target
    cp = cfg.compounds[t.compound]
    life = effective_max_stint(t.compound, state.deg_multiplier, cfg)
    age_at_flag = t.tyre_age_laps + state.remaining_laps
    deg_now = cp.deg_s_per_lap * state.deg_multiplier + (
        cp.cliff_s_per_lap * state.deg_multiplier if t.tyre_age_laps >= cp.cliff_age_laps else 0.0)
    dry_used = sorted({c for c in t.compounds_used if cfg.compounds[c].is_dry})
    lines = [f"{t.code} is on {t.compound.title()} tyres that are {t.tyre_age_laps} laps old. At "
             f"{state.deg_multiplier:g}× wear they are safe for {life} laps and currently lose "
             f"{deg_now:.3f} s per lap of age.",
             f"Staying out would leave them {age_at_flag} laps old at the flag, "
             f"{'within' if age_at_flag <= life else 'beyond'} their safe life."]
    return {"compound": t.compound, "tyre_age_laps": t.tyre_age_laps, "safe_life_laps": life,
            "laps_until_life_limit": life - t.tyre_age_laps, "age_at_flag_if_stay_out": age_at_flag,
            "current_wear_s_per_lap": round(deg_now, 4), "deg_multiplier": state.deg_multiplier,
            "available_sets": {k: v for k, v in t.available_sets.items() if v > 0},
            "dry_compounds_used": dry_used,
            "two_compound_rule_satisfied": len(dry_used) >= 2 or any(
                not cfg.compounds[c].is_dry for c in t.compounds_used),
            "summary": lines}


def analyze_gaps(state: RaceState, cfg: ModelConfig = DEFAULT_CONFIG) -> Dict:
    t = state.target
    field = sorted(state.all_drivers(), key=lambda d: (d.gap_to_leader_s, d.code))
    pos = [d.code for d in field].index(t.code) + 1
    rows = []
    for d in field:
        rows.append({"code": d.code, "gap_to_leader_s": d.gap_to_leader_s,
                     "gap_to_target_s": round(d.gap_to_leader_s - t.gap_to_leader_s, 3),
                     "pace_delta_vs_target_s": round(d.baseline_pace_s - t.baseline_pace_s, 3),
                     "compound": d.compound, "tyre_age_laps": d.tyre_age_laps})
    # Cars that would pass the target during a green-flag stop (behind by less than pit loss).
    exposed = [r["code"] for r in rows
               if 0 < r["gap_to_target_s"] < state.pit_loss_s and r["code"] != t.code]
    ahead = field[pos - 2] if pos > 1 else None
    behind = field[pos] if pos < len(field) else None
    lines = [f"{t.code} is running P{pos} of the {len(field)} simulated cars."]
    if ahead:
        lines.append(f"{ahead.code} is {t.gap_to_leader_s - ahead.gap_to_leader_s:.1f} s ahead.")
    if behind:
        lines.append(f"{behind.code} is {behind.gap_to_leader_s - t.gap_to_leader_s:.1f} s behind.")
    lines.append(f"A normal stop ({state.pit_loss_s:.1f} s) would drop {t.code} behind "
                 f"{', '.join(exposed) if exposed else 'nobody in the group'}.")
    return {"position_in_group": pos, "group_size": len(field), "table": rows,
            "cars_lost_on_green_stop": exposed, "summary": lines}


def analyze_weather(state: RaceState, cfg: ModelConfig = DEFAULT_CONFIG) -> Dict:
    wet_laps = [lap for lap in state.laps() if state.condition_on_lap(lap) == "wet"]
    lines = [f"The track is {state.track_condition} now."]
    for r in state.wet_runs():
        if r["start"] > state.first_lap:
            lines.append(f"Forecast: wet from lap {r['start']} to lap {r['end']}, starting "
                         f"{r['start'] - state.completed_laps} laps from now.")
        elif r["end"] < state.total_laps:
            lines.append(f"Wet now, drying after lap {r['end']}.")
    lines.append(f"{len(wet_laps)} of the {state.remaining_laps} remaining laps are modelled as wet.")
    return {"track_condition_now": state.track_condition, "rain_start_lap": state.rain_start_lap,
            "rain_end_lap": state.rain_end_lap, "wet_laps": len(wet_laps),
            "first_wet_lap": wet_laps[0] if wet_laps else None,
            "wet_runs": state.wet_runs(),
            "forecast_label": "forecast from scenario input" if state.wet_runs() else None,
            "summary": lines}


RESEARCH_TOOLS = {
    "track_status": analyze_track_status,
    "tyre_analysis": analyze_tyres,
    "gap_analysis": analyze_gaps,
    "weather_analysis": analyze_weather,
}

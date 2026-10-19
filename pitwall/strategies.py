"""Candidate-strategy generation tool used by the Planner (and for rival projections)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import DEFAULT_CONFIG, ModelConfig
from .models import DriverState, PitStop, RaceState, Strategy


@dataclass
class CandidateSpec:
    """Planner-controlled description of which candidate families to generate."""
    include_stay_out: bool = True
    include_pit_now: bool = True
    delays: List[int] = field(default_factory=list)            # extra laps before boxing
    compounds: List[str] = field(default_factory=list)         # compounds for single stops
    event_laps: List[Tuple[int, str]] = field(default_factory=list)  # (stop lap, reason)
    wet_transition: bool = False      # stop for INTERMEDIATE right before rain
    return_to_slicks: bool = False    # second stop back to slicks when rain ends
    two_stop: bool = False            # added by revision when one stop cannot reach the flag
    shift_closed_to_open: bool = False  # move stops on closed pit laps to the next open lap
    # Whole plans proposed by the AI Planner: [{"stops": [(lap, compound), ...], "why": str}]
    custom_plans: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def suitable_compounds(state: RaceState, driver: DriverState, lap_after_stop: int,
                       cfg: ModelConfig = DEFAULT_CONFIG) -> List[str]:
    """Available compounds suited to the track condition on the first lap after a stop."""
    cond = state.condition_on_lap(min(lap_after_stop, state.total_laps))
    want_dry = cond == "dry"
    return [c for c, p in cfg.compounds.items()
            if driver.available_sets.get(c, 0) > 0 and p.is_dry == want_dry]


def _next_open_lap(state: RaceState, lap: int) -> Optional[int]:
    while lap <= state.total_laps - 1:
        if lap not in state.pit_closed_laps:
            return lap
        lap += 1
    return None


def _sid(stops: List[PitStop]) -> str:
    if not stops:
        return "STAY_OUT"
    return "PIT_" + "_".join(f"L{s.lap}_{s.compound[:3]}" for s in stops)


def _label(stops: List[PitStop]) -> str:
    if not stops:
        return "Stay out to the flag"
    first, rest = stops[0], stops[1:]
    return ", ".join([f"Box lap {first.lap} for {first.compound.title()}"]
                     + [f"lap {s.lap} for {s.compound.title()}" for s in rest])


def build_candidates(state: RaceState, driver: DriverState, spec: CandidateSpec,
                     cfg: ModelConfig = DEFAULT_CONFIG, origin: str = "plan_v1") -> List[Strategy]:
    """Materialise candidates from a spec. Out-of-race timings are kept so the Critic can
    reject them explicitly; duplicates (same stop sequence) are removed deterministically."""
    out: Dict[Tuple, Strategy] = {}
    now = state.first_lap

    def add(stops: List[PitStop], family: str, why: str):
        key = tuple((s.lap, s.compound) for s in stops)
        if key not in out:
            out[key] = Strategy(_sid(stops), _label(stops), family, stops, why, origin)

    def comps_for(lap: int) -> List[str]:
        if spec.compounds:
            return [c for c in spec.compounds if c in cfg.compounds]
        return suitable_compounds(state, driver, lap + 1, cfg)

    if spec.include_stay_out:
        add([], "stay_out", "finish on the current set")

    timings: List[Tuple[int, str, str]] = []
    if spec.include_pit_now:
        timings.append((now, "pit_now", "box this lap"))
    for d in spec.delays:
        timings.append((now + d, "delay", f"box after a {d}-lap delay"))
    for lap, why in spec.event_laps:
        timings.append((lap, "event_aligned", why))
    if spec.shift_closed_to_open:
        shifted = []
        for lap, fam, why in timings:
            if lap in state.pit_closed_laps:
                nl = _next_open_lap(state, lap)
                if nl is not None:
                    shifted.append((nl, "earliest_open", f"earliest open pit lap (L{lap} closed)"))
        timings += shifted

    for lap, fam, why in timings:
        for c in comps_for(lap):
            add([PitStop(lap, c)], fam, why)

    arrival = state.rain_arrival()
    rs = arrival["start"] if arrival else None
    if spec.return_to_slicks and state.dries_after() is not None:
        dry_lap = state.dries_after()
        if now <= dry_lap <= state.total_laps - 1:
            for dc in suitable_compounds(state, driver, dry_lap + 1, cfg):
                add([PitStop(dry_lap, dc)], "event_aligned", "back to slicks as the track dries")
    if spec.wet_transition and rs is not None:
        wet = [c for c in ("INTERMEDIATE", "WET") if driver.available_sets.get(c, 0) > 0]
        re_ = arrival["end"] if arrival["end"] < state.total_laps else None
        for stop_lap, why in ((rs - 1, "fit wets for the first wet lap"),
                              (rs, "box on the first wet lap")):
            if stop_lap < now:
                continue
            for wc in wet:
                first = [PitStop(stop_lap, wc)]
                add(first, "event_aligned", why)
                if spec.return_to_slicks and re_ is not None and re_ < state.total_laps - 1:
                    for dc in suitable_compounds(state, driver, re_ + 1, cfg):
                        add(first + [PitStop(re_, dc)], "event_aligned",
                            why + "; back to slicks when the track dries")

    if spec.two_stop:
        # compounds suited to the track condition right after each stop (dry -> slicks, wet -> wets)
        for p1 in (now, now + cfg.short_delay_laps):
            p2 = p1 + math.ceil((state.total_laps - p1) / 2)
            for c1 in suitable_compounds(state, driver, p1 + 1, cfg):
                for c2 in suitable_compounds(state, driver, p2 + 1, cfg):
                    add([PitStop(p1, c1), PitStop(p2, c2)], "two_stop",
                        "two stops to stay within safe tyre life")
        # Fresh slicks now, then wet tyres just before the forecast rain arrives.
        if arrival and rs is not None:
            wet = [c for c in ("INTERMEDIATE", "WET") if driver.available_sets.get(c, 0) > 0]
            for p1 in (now, now + cfg.short_delay_laps):
                if p1 >= rs - 1:
                    continue
                for c1 in suitable_compounds(state, driver, p1 + 1, cfg):
                    for wc in wet:
                        add([PitStop(p1, c1), PitStop(rs - 1, wc)], "two_stop",
                            "fresh slicks now, wet tyres when the rain arrives")

    for plan in spec.custom_plans:
        stops = [PitStop(int(lap), comp) for lap, comp in plan["stops"] if comp in cfg.compounds]
        if stops:
            add(stops, "ai_plan", plan.get("why") or "proposed by the AI planner")

    return sorted(out.values(), key=lambda s: (len(s.stops), s.first_pit_lap or 0, s.id))

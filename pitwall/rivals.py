"""Rival projection tool: each simulated competitor is assumed to follow its own
lowest-time feasible strategy under the same model (an explicit modelling assumption)."""
from __future__ import annotations

from typing import Dict

from .config import DEFAULT_CONFIG, ModelConfig
from .models import RaceState
from .simulation import check_strategy_constraints, rank_key, simulate_strategy
from .strategies import CandidateSpec, build_candidates


def rival_spec(state: RaceState, cfg: ModelConfig = DEFAULT_CONFIG) -> CandidateSpec:
    events = [(max(p["start"], state.first_lap), "stop under Safety Car/VSC") for p in state.neutral_ahead()
              if max(p["start"], state.first_lap) <= state.total_laps - 1]
    arrival = state.rain_arrival()
    return CandidateSpec(include_stay_out=True, include_pit_now=True,
                         delays=[cfg.short_delay_laps, 2 * cfg.short_delay_laps],
                         event_laps=events, wet_transition=arrival is not None,
                         return_to_slicks=state.dries_after() is not None or bool(
                             arrival and arrival["end"] < state.total_laps),
                         two_stop=True, shift_closed_to_open=True)


def project_rivals(state: RaceState, cfg: ModelConfig = DEFAULT_CONFIG) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for rival in state.competitors:
        cands = build_candidates(state, rival, rival_spec(state, cfg), cfg, origin="rival_model")
        feasible = [c for c in cands if not check_strategy_constraints(state, rival, c, cfg)]
        assumption = "own lowest-time feasible strategy (model assumption)"
        if not feasible:
            feasible = [c for c in cands if c.id == "STAY_OUT"]
            assumption = "no feasible option found; stay out assumed"
        sims = {c.id: simulate_strategy(state, rival, c, cfg) for c in feasible}
        best = min(feasible, key=lambda c: rank_key(sims[c.id].remaining_time_s, c))
        sim = sims[best.id]
        out[rival.code] = {
            "code": rival.code, "strategy_id": best.id, "strategy_label": best.label,
            "assumption": assumption, "remaining_time_s": sim.remaining_time_s,
            "finish_time_ref_s": sim.finish_time_ref_s,
            "trace": {r["lap"]: r["cumulative_ref_s"] for r in sim.laps},
        }
    return out

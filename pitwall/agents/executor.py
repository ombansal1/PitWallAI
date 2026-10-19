"""Strategy Executor Agent: runs the Python simulation tools named in the plan."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from ..config import DEFAULT_CONFIG, ModelConfig
from ..models import RaceState, SimulationResult, Strategy
from ..rivals import project_rivals
from ..simulation import project_positions, simulate_strategy
from ..workflow_state import WorkflowState

AGENT = "Strategy Executor"

# Test/demo hook only: (strategy_id, result, attempt) -> result. Used to inject a numerical
# fault so the Critic's rejection-and-revision path can be exercised. Never set by the app.
FaultInjector = Callable[[str, SimulationResult, int], SimulationResult]


class ExecutorAgent:
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG, fault_injector: Optional[FaultInjector] = None):
        self.cfg = cfg
        self.fault_injector = fault_injector

    def prepare(self, state: RaceState, tools: List[str], ws: WorkflowState) -> None:
        if state.tyre_curves:
            ws.record(AGENT, "tool", "tyre_model",
                      f"Using the {state.tyre_model_label} for tyre wear (learned from real laps); base pace "
                      f"re-measured with it: {state.target.baseline_pace_s:.3f} s.")
        if "project_rivals" in tools and state.competitors and not ws.rivals:
            ws.rivals = project_rivals(state, self.cfg)
            ws.record(AGENT, "tool", "project_rivals",
                      "; ".join(f"{c} assumed to {r['strategy_label'][0].lower()}{r['strategy_label'][1:]}"
                                for c, r in ws.rivals.items()))

    def execute(self, state: RaceState, candidates: Iterable[Strategy], tools: List[str],
                ws: WorkflowState, reason: str = "plan") -> List[str]:
        traces = {c: r["trace"] for c, r in ws.rivals.items()} if ws.rivals else None
        done = []
        for cand in candidates:
            attempt = ws.attempts.get(cand.id, 0) + 1
            ws.attempts[cand.id] = attempt
            res = simulate_strategy(state, state.target, cand, self.cfg, rival_traces=traces)
            if self.fault_injector is not None:
                res = self.fault_injector(cand.id, res, attempt)
            ws.simulations[cand.id] = res
            if "project_positions" in tools and ws.rivals:
                ws.positions[cand.id] = project_positions(
                    res.finish_time_ref_s, state.target.code,
                    {c: r["finish_time_ref_s"] for c, r in ws.rivals.items()},
                    {d.code: d.gap_to_leader_s for d in state.all_drivers()})
            done.append(cand.id)
            ws.record(AGENT, "tool", "simulate_strategy",
                      f"{cand.label}: {res.remaining_time_s:.2f} s to the flag ({res.lap_time_total_s:.2f} s "
                      f"lapping + {res.pit_loss_total_s:.2f} s pit lane). Run {attempt}, from {reason}.")
        return done

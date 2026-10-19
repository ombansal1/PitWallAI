"""Strategy Critic Agent: validates inputs, feasibility, numerical integrity and plan coverage,
and issues routed revision requests (to the Planner or the Executor)."""
from __future__ import annotations

import math
from typing import Any, Dict, List

from ..config import DEFAULT_CONFIG, QUESTIONS, ModelConfig
from ..models import RaceState, SimulationResult, Strategy
from ..simulation import RULE_NAMES, check_strategy_constraints, pit_loss_on_lap
from ..validation import validate_race_input
from ..workflow_state import WorkflowState

AGENT = "Strategy Critic"


class CriticAgent:
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    # ------------------------------------------------------------ input validation
    def validate_inputs(self, raw: Any, question_key: str, ws: WorkflowState) -> List[Dict[str, str]]:
        issues = validate_race_input(raw, self.cfg)
        if question_key not in QUESTIONS:
            issues.append({"field": "question", "message": f"unknown question {question_key!r}"})
        ws.record(AGENT, "validation", "validate_inputs",
                  "inputs valid" if not issues else f"{len(issues)} input issue(s) — workflow stopped",
                  issues=issues)
        return issues

    # ------------------------------------------------------------ numerical integrity
    def integrity_errors(self, state: RaceState, cand: Strategy, sim: SimulationResult) -> List[str]:
        cfg, errs = self.cfg, []
        base = state.target.baseline_pace_s
        expected_laps = list(state.laps())
        if [r["lap"] for r in sim.laps] != expected_laps:
            errs.append(f"lap sequence does not cover laps {expected_laps[0]}-{expected_laps[-1]}")
            return errs
        numeric = [sim.lap_time_total_s, sim.pit_loss_total_s, sim.remaining_time_s, sim.finish_time_ref_s]
        for r in sim.laps:
            numeric += [r["lap_time_s"], r["pit_loss_s"], r["cumulative_ref_s"], r["tyre_adj_s"],
                        r["weather_adj_s"], r["sc_adj_s"], r["traffic_adj_s"], r.get("fuel_adj_s", 0.0)]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in numeric):
            errs.append("non-finite value in simulation output")
            return errs
        lo, hi = base * cfg.min_lap_factor, base * cfg.max_lap_factor
        bad = [r["lap"] for r in sim.laps if not lo <= r["lap_time_s"] <= hi]
        if bad:
            errs.append(f"implausible lap time on laps {bad[:5]} (bounds {lo:.1f}-{hi:.1f} s)")
        lap_sum = sum(r["lap_time_s"] for r in sim.laps)
        pit_sum = sum(r["pit_loss_s"] for r in sim.laps)
        if abs(lap_sum - sim.lap_time_total_s) > 1e-6 or abs(pit_sum - sim.pit_loss_total_s) > 1e-6:
            errs.append("totals do not match per-lap breakdown")
        if abs(sim.lap_time_total_s + sim.pit_loss_total_s - sim.remaining_time_s) > 1e-6:
            errs.append("remaining time ≠ lap times + pit losses")
        if abs(state.target.gap_to_leader_s + sim.remaining_time_s - sim.finish_time_ref_s) > 1e-6:
            errs.append("finish time not on the common gap reference")
        stop_laps = {s.lap: s for s in cand.stops}
        pit_rows = [r["lap"] for r in sim.laps if r["pit_loss_s"] > 0]
        if sorted(pit_rows) != sorted(stop_laps) or sim.pit_loss_count != len(cand.stops):
            errs.append(f"pit loss applied on laps {pit_rows}, expected exactly once on {sorted(stop_laps)}")
        for r in sim.laps:
            if r["lap"] in stop_laps:
                exp = pit_loss_on_lap(state, r["lap"], cfg)
                if abs(r["pit_loss_s"] - exp) > 1e-6:
                    errs.append(f"pit loss on lap {r['lap']} is {r['pit_loss_s']:.3f}, expected {exp:.3f}")
        for prev, cur in zip(sim.laps, sim.laps[1:]):
            if prev["lap"] in stop_laps:
                if cur["tyre_age"] != 0 or cur["compound"] != stop_laps[prev["lap"]].compound:
                    errs.append(f"tyre not reset after stop on lap {prev['lap']}")
            elif cur["tyre_age"] != prev["tyre_age"] + 1 or cur["compound"] != prev["compound"]:
                errs.append(f"tyre age/compound inconsistent at lap {cur['lap']}")
                break
        return errs

    # ------------------------------------------------------------ review
    def review(self, state: RaceState, plan, ws: WorkflowState) -> Dict[str, Any]:
        cfg = self.cfg
        verdicts: Dict[str, Dict[str, Any]] = {}
        rerun: List[str] = []
        all_codes: List[str] = []
        for cid, cand in ws.candidates.items():
            violations = check_strategy_constraints(state, state.target, cand, cfg)
            entry: Dict[str, Any] = {"status": "accepted", "violations": violations, "integrity": []}
            if violations:
                entry["status"] = "rejected"
                all_codes += [v["code"] for v in violations]
            sim = ws.simulations.get(cid)
            if sim is None:
                entry["status"] = "rejected" if violations else "not_simulated"
            else:
                errs = self.integrity_errors(state, cand, sim)
                if errs:
                    entry["integrity"] = errs
                    if not violations:
                        if ws.attempts.get(cid, 0) < 2:
                            entry["status"] = "integrity_failed_rerun"
                            rerun.append(cid)
                        else:
                            entry["status"] = "rejected"
                            all_codes.append("NUMERICAL_INTEGRITY")
            verdicts[cid] = entry

        accepted = [c for c, v in verdicts.items() if v["status"] == "accepted"]
        requests: List[Dict[str, Any]] = []
        if rerun:
            requests.append({"target": "executor", "code": "NUMERICAL_INTEGRITY", "candidate_ids": rerun,
                             "reason": f"{len(rerun)} simulation result(s) failed the number checks, so they "
                                       "must be re-run from scratch."})
        acc = [ws.candidates[c] for c in accepted]
        if not accepted and not rerun:
            requests.append({"target": "planner", "code": "NO_FEASIBLE_CANDIDATE",
                             "violation_codes": sorted(set(all_codes)),
                             "reason": "No option passed the rules (" + ", ".join(
                                 RULE_NAMES.get(c, c) for c in sorted(set(all_codes)))
                                       + "), so different options are needed."})
        # Coverage checks: obvious options missing from the plan.
        wet_laps = sum(1 for l in state.laps() if state.condition_on_lap(l) == "wet")
        if (state.rain_arrival() is not None and wet_laps > cfg.max_slick_laps_on_wet
                and any(state.target.available_sets.get(c, 0) for c in ("INTERMEDIATE", "WET"))
                and not any(any(not cfg.compounds[s.compound].is_dry for s in c.stops) for c in acc)):
            requests.append({"target": "planner", "code": "MISSING_WET_TRANSITION",
                             "reason": f"The forecast has {wet_laps} wet laps, but no valid option fits wet tyres."})
        if plan.question_key != "compound_choice":
            for p in state.neutral_ahead():
                window = range(max(p["start"], state.first_lap), min(p["end"], state.total_laps - 1) + 1)
                open_in_window = [l for l in window if l not in state.pit_closed_laps]
                if open_in_window and not any(c.first_pit_lap in window for c in ws.candidates.values()):
                    name = "Safety Car" if p["kind"] == "SC" else "Virtual Safety Car"
                    requests.append({"target": "planner", "code": "MISSING_SC_OPTION",
                                     "reason": f"The {name} on laps {p['start']}–{p['end']} makes a stop cheaper, "
                                               "but no option stops then."})
                    break
        pit_now = [c for c in ws.candidates.values() if c.family == "pit_now"]
        if pit_now and all(any(v["code"] == "PIT_LANE_CLOSED" for v in verdicts[c.id]["violations"]) for c in pit_now) \
                and not any(c.family == "earliest_open" for c in ws.candidates.values()):
            requests.append({"target": "planner", "code": "PIT_NOW_BLOCKED",
                             "reason": f"The pit lane is closed on lap {state.first_lap}, so stopping this lap "
                                       "is impossible."})

        report = {"iteration": len(ws.critic_reports) + 1, "verdicts": verdicts, "accepted": accepted,
                  "rejected": [c for c, v in verdicts.items() if v["status"] == "rejected"],
                  "revision_requests": requests}
        prev = ws.critic_reports[-1]["verdicts"] if ws.critic_reports else {}
        ws.critic_reports.append(report)
        for cid, v in verdicts.items():
            # log each verdict once: new candidates or a changed status
            if v["status"] != "accepted" and prev.get(cid, {}).get("status") != v["status"]:
                why = "; ".join([x["message"] for x in v["violations"]] + v["integrity"])
                ws.record(AGENT, "validation", f"{v['status']}: {cid}", why)
            elif v["status"] == "accepted" and prev.get(cid, {}).get("status") == "integrity_failed_rerun":
                ws.record(AGENT, "validation", f"accepted after re-run: {cid}", "integrity checks now pass")
        ws.record(AGENT, "validation", "review",
                  f"iteration {report['iteration']}: {len(accepted)} accepted, "
                  f"{len(report['rejected'])} rejected, {len(rerun)} to re-run; "
                  f"{len(requests)} revision request(s)",
                  requests=[{k: r[k] for k in ('target', 'code', 'reason')} for r in requests])
        return report

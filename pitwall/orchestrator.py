"""Workflow controller: shared state, tool execution, conditional routing and a bounded
Critic-driven revision loop (plain Python; no framework required)."""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .agents import CriticAgent, ExecutorAgent, OperationsAgent, PlannerAgent, ResearchAgent
from .agents.executor import FaultInjector
from .config import DEFAULT_CONFIG, QUESTIONS, ModelConfig
from .llm import MockReasoner
from .simulation import rank_key
from .validation import parse_race_state
from .workflow_state import WorkflowState

ORCH = "Orchestrator"


def input_fingerprint(raw: Dict[str, Any], question_key: str) -> str:
    blob = json.dumps({"raw": raw, "q": question_key}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_analysis(raw: Dict[str, Any], question_key: str = "full_strategy",
                 cfg: ModelConfig = DEFAULT_CONFIG, trigger_event: Optional[Dict[str, Any]] = None,
                 reasoner=None, fault_injector: Optional[FaultInjector] = None,
                 progress=None) -> Dict[str, Any]:
    """Run the five-agent workflow. `progress(entry)` (optional) is called after every logged step,
    so a UI can show the agents working in real time."""
    reasoner = reasoner or MockReasoner()
    if hasattr(reasoner, "begin_run"):
        reasoner.begin_run()
    snapshot = copy.deepcopy(raw)  # inputs are never modified
    ws = WorkflowState(analysis_id="PW-" + uuid.uuid4().hex[:10], question=question_key,
                       input_snapshot=snapshot, trigger_event=trigger_event, model_mode=reasoner.mode,
                       listener=progress)
    planner, research = PlannerAgent(cfg), ResearchAgent(cfg)
    executor, critic, ops = ExecutorAgent(cfg, fault_injector), CriticAgent(cfg), OperationsAgent(cfg)
    ws.record(ORCH, "agent", "start", f"Question: {QUESTIONS.get(question_key, question_key)} {reasoner.label}."
              + (f" Triggered by: {trigger_event.get('description')}." if trigger_event else ""))

    # 1. Validate inputs (not recoverable: reported to the user, never repaired).
    issues = critic.validate_inputs(snapshot, question_key, ws)
    if issues:
        ws.input_issues, ws.status = issues, "invalid_input"
        return _result(ws, cfg, reasoner, snapshot, question_key)
    state = parse_race_state(copy.deepcopy(snapshot), cfg)
    ws.race_state = state

    # 2. Plan → 3. research → 4. generate + simulate.
    plan = planner.create_plan(state, question_key, ws, reasoner)
    ws.plan_history.append(plan.to_dict())
    ws.research = research.run(state, plan.analyses, ws)
    executor.prepare(state, plan.simulation_tools, ws)
    for c in planner.materialise(state, plan, ws):
        ws.candidates[c.id] = c
    executor.execute(state, list(ws.candidates.values()), plan.simulation_tools, ws)

    # 5. Critic loop with routing and an explicit retry limit.
    while True:
        report = critic.review(state, plan, ws)
        reqs = report["revision_requests"]
        if not reqs:
            break
        if ws.retry_count >= cfg.max_revisions:
            ws.record(ORCH, "revision", "retry_limit_reached",
                      f"{len(reqs)} unresolved request(s) after {ws.retry_count} revision(s); stopping")
            break
        ws.retry_count += 1
        rev = {"revision": ws.retry_count, "requests": [{k: r[k] for k in ("target", "code", "reason")}
                                                        for r in reqs], "actions": []}
        progressed = False
        exec_reqs = [r for r in reqs if r["target"] == "executor"]
        plan_reqs = [r for r in reqs if r["target"] == "planner"]
        for r in exec_reqs:
            ids = [i for i in r["candidate_ids"] if i in ws.candidates]
            ws.record(ORCH, "revision", "route_to_executor", f"re-run {ids}: {r['reason']}")
            executor.execute(state, [ws.candidates[i] for i in ids], plan.simulation_tools, ws,
                             reason="critic re-run")
            rev["actions"].append(f"Executor re-ran {len(ids)} option(s): "
                                  + "; ".join(ws.candidates[i].label for i in ids))
            progressed = progressed or bool(ids)
        if plan_reqs:
            ws.record(ORCH, "revision", "route_to_planner", "; ".join(r["code"] for r in plan_reqs))
            new_plan, changes = planner.revise_plan(state, plan, plan_reqs, ws, reasoner)
            if changes:
                plan = new_plan
                ws.plan_history.append(plan.to_dict())
                ws.research = research.run(state, plan.analyses, ws, already_done=ws.research)
                new = [c for c in planner.materialise(state, plan, ws) if c.id not in ws.candidates]
                for c in new:
                    ws.candidates[c.id] = c
                executor.execute(state, new, plan.simulation_tools, ws, reason=f"plan v{plan.version}")
                rev["actions"].append(f"Planner wrote plan v{plan.version}: " + "; ".join(changes)
                                      + f". This added {len(new)} new option(s).")
                progressed = progressed or bool(new)
            else:
                rev["actions"].append("Planner had no further alternatives")
        ws.revisions.append(rev)
        if not progressed:
            ws.record(ORCH, "revision", "no_progress",
                      "revision produced no new work; using the last Critic review")
            break

    ws.plan = plan.to_dict()
    # 6. Select + finalise.
    ws.recommendation = ops.finalize(state, ws, reasoner)
    ws.selected_id = ws.recommendation.get("selected_id")
    return _result(ws, cfg, reasoner, snapshot, question_key)


def _candidate_rows(ws: WorkflowState) -> list:
    if not ws.critic_reports:
        return []
    rep = ws.critic_reports[-1]
    acc = [c for c in rep["accepted"]]
    best_t = min((ws.simulations[c].remaining_time_s for c in acc), default=None)
    rows = []
    for cid, cand in ws.candidates.items():
        v = rep["verdicts"].get(cid, {"status": "not_reviewed", "violations": [], "integrity": []})
        sim = ws.simulations.get(cid)
        pos = ws.positions.get(cid)
        rows.append({
            "id": cid, "label": cand.label, "family": cand.family, "origin": cand.origin,
            "stops": len(cand.stops), "first_pit_lap": cand.first_pit_lap,
            "status": "SELECTED" if cid == ws.selected_id else v["status"].upper(),
            "remaining_time_s": round(sim.remaining_time_s, 3) if sim else None,
            "delta_to_best_s": round(sim.remaining_time_s - best_t, 3) if sim and best_t is not None else None,
            "lap_time_total_s": round(sim.lap_time_total_s, 3) if sim else None,
            "pit_loss_total_s": round(sim.pit_loss_total_s, 3) if sim else None,
            "final_compound": sim.stints[-1]["compound"] if sim else None,
            "final_stint_age": sim.stints[-1]["end_age"] if sim else None,
            "traffic_laps": sim.traffic_laps if sim else None,
            "position_in_group": pos["position_in_group"] if pos else None,
            "issue_list": [x["message"] for x in v["violations"]] + [m[:1].upper() + m[1:] + "." for m in v["integrity"]],
            "issues": " ".join([x["message"] for x in v["violations"]]
                               + [m[:1].upper() + m[1:] + "." for m in v["integrity"]]),
            "_rank": rank_key(sim.remaining_time_s, cand) if sim else (float("inf"),),
        })
    rows.sort(key=lambda r: (r["status"] not in ("SELECTED", "ACCEPTED"), r["_rank"]))
    for r in rows:
        r.pop("_rank")
    return rows


def _result(ws: WorkflowState, cfg: ModelConfig, reasoner, raw, question_key) -> Dict[str, Any]:
    st = ws.race_state
    return {
        "analysis_id": ws.analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_fingerprint": input_fingerprint(raw, question_key),
        "question_key": question_key, "question": QUESTIONS.get(question_key, question_key),
        "model_mode": reasoner.mode, "model_label": reasoner.label,
        "ai": {"provider": getattr(reasoner, "provider", "rules"), "model": getattr(reasoner, "model", "rules"),
               "short": getattr(reasoner, "short", "Rules"),
               "calls": list(getattr(reasoner, "calls", []) or []),
               "used": any(c.get("ok") for c in (getattr(reasoner, "calls", []) or []))},
        "status": ws.status if ws.status != "running" else "error",
        "trigger_event": ws.trigger_event,
        "data_source": raw.get("data_source", "synthetic") if isinstance(raw, dict) else None,
        "data_source_label": raw.get("data_source_label", "") if isinstance(raw, dict) else None,
        "input_snapshot": raw, "input_issues": ws.input_issues,
        "decision_point": ({"completed_laps": st.completed_laps, "total_laps": st.total_laps,
                            "remaining_laps": st.remaining_laps} if st else None),
        "plan": ws.plan, "plan_history": ws.plan_history, "research": ws.research,
        "candidates": _candidate_rows(ws),
        "candidate_defs": {k: v.to_dict() for k, v in ws.candidates.items()},
        "simulations": {k: {**v.summary(), "laps": v.laps} for k, v in ws.simulations.items()},
        "rivals": {k: {kk: vv for kk, vv in v.items() if kk != "trace"} for k, v in ws.rivals.items()},
        "positions": ws.positions,
        "critic_reports": ws.critic_reports, "revisions": ws.revisions, "retry_count": ws.retry_count,
        "max_revisions": cfg.max_revisions,
        "selected_id": ws.selected_id, "recommendation": ws.recommendation, "log": ws.log,
    }

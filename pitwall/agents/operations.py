"""Pit Wall Operations Agent: turns the validated, calculated result into the recommendation,
trade-offs, explicit-rule risk category and a clearly simulated team-radio message."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_CONFIG, RISK_RULE_TEXT, ModelConfig
from ..models import RaceState, Strategy
from ..simulation import RULE_NAMES, effective_max_stint, rank_key
from ..workflow_state import WorkflowState
from .ai_brief import dumps, race_brief

AGENT = "Pit Wall Operations"


def action_label(state: RaceState, s: Strategy) -> str:
    if not s.stops:
        return "Stay out to the flag"
    first = s.stops[0]
    if len(s.stops) > 1:
        second = s.stops[1]
        return (f"Two-stop: box lap {first.lap} for {first.compound.title()}, "
                f"then lap {second.lap} for {second.compound.title()}")
    if first.lap == state.first_lap:
        return f"Box this lap for {first.compound.title()} tyres"
    return f"Stay out, then box lap {first.lap} for {first.compound.title()} tyres"


def radio_mismatch(text: str, rec: Dict[str, Any], ws: WorkflowState) -> Optional[str]:
    """Cheap consistency check of an AI-written radio message against the chosen plan."""
    import re
    t = text.lower()
    stops = rec.get("stops") or []
    first = ws.race_state.first_lap if ws.race_state else None
    laps = {int(x) for x in re.findall(r"\blap\s+(\d{1,3})\b", t)}
    if laps - {s["lap"] for s in stops}:
        return f"mentions lap {sorted(laps - {s['lap'] for s in stops})[0]}, not in the plan"
    if re.search(r"box,? box|box this lap|box now|pit this lap|pit now", t) and (not stops or stops[0]["lap"] != first):
        return "says box now, but the plan doesn't stop this lap"
    if stops and re.search(r"stay out to the (end|flag)|no more stops", t):
        return "says no more stops, but the plan has a stop"
    comps = {c.lower() for c in ("soft", "medium", "hard", "intermediate", "wet")}
    said = {c for c in comps if re.search(rf"\b{c}s?\b", t)}
    planned = {s["compound"].lower() for s in stops}
    if said and planned and not (said & planned):
        return "names a different tyre"
    return None


class OperationsAgent:
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def select(self, ws: WorkflowState) -> Optional[str]:
        report = ws.critic_reports[-1]
        acc = [ws.candidates[c] for c in report["accepted"]]
        if not acc:
            return None
        best = min(acc, key=lambda c: rank_key(ws.simulations[c.id].remaining_time_s, c))
        ws.record(AGENT, "agent", "select_strategy",
                  f"{best.id} has the lowest estimated remaining time among {len(acc)} validated candidates "
                  f"({ws.simulations[best.id].remaining_time_s:.3f} s)")
        return best.id

    def risk(self, state: RaceState, sel: Strategy, margin: Optional[float], ws: WorkflowState) -> Dict:
        cfg = self.cfg
        last = ws.simulations[sel.id].stints[-1]
        life = effective_max_stint(last["compound"], state.deg_multiplier, cfg)
        frac = last["end_age"] / life if life else 1.0
        depends = (bool(state.neutral_ahead()) or state.rain_arrival() is not None
                   or state.dries_after() is not None)
        reasons = []
        if margin is not None and margin < 0:
            reasons.append(f"a valid option {-margin:.2f} s faster exists (chosen for lower risk)")
            margin = None if frac <= cfg.tyre_life_risk_fraction else margin
            if frac <= cfg.tyre_life_risk_fraction:
                return {"category": "MEDIUM", "reasons": reasons + (["depends on forecast/Safety Car timing"]
                                                                    if depends else []),
                        "final_stint_life_used": round(frac, 3),
                        "rule": RISK_RULE_TEXT.format(h=cfg.risk_margin_high_s, f=cfg.tyre_life_risk_fraction,
                                                      l=cfg.risk_margin_low_s)}
        if margin is not None and 0 <= margin < cfg.risk_margin_high_s:
            reasons.append(f"margin {margin:.2f} s < {cfg.risk_margin_high_s} s")
        if frac > cfg.tyre_life_risk_fraction:
            reasons.append(f"final stint uses {frac:.0%} of safe tyre life")
        if reasons:
            cat = "HIGH"
        else:
            if margin is None:
                reasons.append("only feasible option")
            elif margin < cfg.risk_margin_low_s:
                reasons.append(f"margin {margin:.2f} s < {cfg.risk_margin_low_s} s")
            if depends:
                reasons.append("depends on forecast/Safety Car timing")
            cat = "MEDIUM" if reasons else "LOW"
            if not reasons:
                reasons.append(f"margin {margin:.2f} s ≥ {cfg.risk_margin_low_s} s, tyre life {frac:.0%}")
        return {"category": cat, "reasons": reasons, "final_stint_life_used": round(frac, 3),
                "rule": RISK_RULE_TEXT.format(h=cfg.risk_margin_high_s, f=cfg.tyre_life_risk_fraction,
                                              l=cfg.risk_margin_low_s)}

    def finalize(self, state: RaceState, ws: WorkflowState, reasoner) -> Dict[str, Any]:
        sel_id = self.select(ws)
        if sel_id is None:
            rec = self._review_required(state, ws)
        else:
            ai = self._ai_choice(state, ws, reasoner, sel_id) if getattr(reasoner, "mode", "") == "ai" else None
            chosen = ai["choice"] if ai else sel_id
            rec = self._recommend(state, ws, chosen)
            if ai:
                self._apply_ai(rec, ai, ws)
            else:
                rec["decided_by"] = "Rule: the fastest option that passed every check"
        ws.record(AGENT, "agent", "finalize", f"{rec['status']}: {rec['headline']}")
        return rec

    # ------------------------------------------------------------------ AI decision
    def shortlist(self, ws: WorkflowState) -> List[Strategy]:
        acc = sorted((ws.candidates[c] for c in ws.critic_reports[-1]["accepted"]),
                     key=lambda c: rank_key(ws.simulations[c.id].remaining_time_s, c))
        if not acc:
            return []
        best = ws.simulations[acc[0].id].remaining_time_s
        return [c for c in acc if ws.simulations[c.id].remaining_time_s - best <= self.cfg.ai_choice_window_s
                ][:self.cfg.ai_shortlist_max]

    def _ai_choice(self, state: RaceState, ws: WorkflowState, reasoner, fastest_id: str) -> Optional[Dict]:
        if not reasoner.available():
            ws.record(AGENT, "ai", "ai_unavailable", f"AI model unavailable ({reasoner.last_error}); "
                      "the rule picks the fastest valid option.")
            return None
        short = self.shortlist(ws)
        best_t = ws.simulations[short[0].id].remaining_time_s
        opts = []
        for c in short:
            sim = ws.simulations[c.id]
            last = sim.stints[-1]
            life = effective_max_stint(last["compound"], state.deg_multiplier, self.cfg)
            pos = ws.positions.get(c.id) or {}
            opts.append({"id": c.id, "plan": action_label(state, c),
                         "seconds_slower_than_fastest": round(sim.remaining_time_s - best_t, 2),
                         "stops": len(c.stops), "final_tyre": last["compound"],
                         "final_tyre_life_used": f"{last['end_age'] / life:.0%}" if life else "n/a",
                         "laps_stuck_in_traffic": sim.traffic_laps,
                         "projected_place_in_group": pos.get("position_in_group"),
                         "relies_on_forecast_or_safety_car": c.family in ("event_aligned", "ai_plan")
                         and bool(state.neutral_ahead() or state.rain_arrival())})
        system = ("You are Pit Wall Operations, the agent that makes the final strategy call for an F1 team. "
                  "Every option below already passed the Critic's safety rules and was timed by the simulator. "
                  "Pick ONE option id. Prefer the fastest unless a slower one (all are within "
                  f"{self.cfg.ai_choice_window_s:g} s) is clearly safer, e.g. less tyre wear, no traffic, fewer "
                  "stops or less reliance on a forecast. Never invent numbers. Reply with JSON only.")
        user = (f"RACE STATE: {dumps(race_brief(state, self.cfg))}\nOPTIONS: {dumps(opts)}\n"
                'JSON shape: {"choice": "<option id>", "why": "one sentence", '
                '"explanation": ["2-3 short plain-English sentences for the race engineer"], '
                f'"radio": "short team-radio message to the driver ({state.target.code}), max 25 words"}}')
        data = reasoner.ask_json(system, user, purpose="Operations: choose the call", max_tokens=700)
        if data is None:
            ws.record(AGENT, "ai", "ai_failed", f"AI model didn't answer usably ({reasoner.last_error}); "
                      "the rule picks the fastest valid option.")
            return None
        choice = str(data.get("choice", "")).strip()
        ids = {c.id for c in short}
        if choice not in ids:
            ws.record(AGENT, "ai", "ai_choice_rejected",
                      f"AI chose '{choice[:40]}', which is not on the Critic-approved shortlist "
                      f"({', '.join(sorted(ids))}); the rule picks the fastest valid option instead.")
            choice = fastest_id
            data["why"] = "fallback: AI choice was not on the approved shortlist"
        cost = ws.simulations[choice].remaining_time_s - ws.simulations[fastest_id].remaining_time_s
        ws.record(AGENT, "ai", "ai_choose",
                  f"AI model ({reasoner.short}) chose {ws.candidates[choice].label}"
                  + (f", {cost:.2f} s slower than the fastest" if cost > 1e-6 else ", the fastest option")
                  + f" from a shortlist of {len(short)}. Reason: {str(data.get('why', '')).strip()[:200]}")
        return {"choice": choice, "why": str(data.get("why", "")).strip()[:300], "model": reasoner.short,
                "explanation": [str(x).strip()[:300] for x in (data.get("explanation") or []) if str(x).strip()][:3],
                "radio": " ".join(str(data.get("radio", "")).split())[:220], "shortlist": [c.id for c in short],
                "time_cost_s": round(cost, 3), "fastest_id": fastest_id}

    def _apply_ai(self, rec: Dict[str, Any], ai: Dict[str, Any], ws: WorkflowState) -> None:
        if rec["status"] != "recommended":
            return
        rec["ai"] = ai
        rec["llm_used"] = True
        rec["decided_by"] = (f"AI model ({ai['model']}) chose from {len(ai['shortlist'])} Critic-approved option(s)"
                             + (f", accepting {ai['time_cost_s']:.2f} s over the fastest" if ai["time_cost_s"] > 0 else ""))
        if ai["explanation"]:
            rec["llm_summary"] = " ".join(ai["explanation"])
        problem = radio_mismatch(ai["radio"], rec, ws) if ai["radio"] else "empty"
        if not problem:
            rec["radio"] = {"label": "Simulated team radio written by the AI model, not a real message",
                            "text": ai["radio"], "template": rec["radio"]["text"]}
        elif ai["radio"]:
            ws.record(AGENT, "ai", "ai_radio_rejected", f"AI radio message didn't match the plan ({problem}); "
                      "the template message is used instead.")
        if ai["time_cost_s"] > 0:
            fastest = ws.candidates[ai["fastest_id"]]
            rec["summary"].insert(1, f"The AI chose this over the fastest valid option ({fastest.label}, "
                                     f"{ai['time_cost_s']:.2f} s quicker): {ai['why']}")

    # ------------------------------------------------------------------
    def _review_required(self, state: RaceState, ws: WorkflowState) -> Dict[str, Any]:
        rep = ws.critic_reports[-1]
        names = RULE_NAMES
        codes = Counter(v["code"] for x in rep["verdicts"].values() for v in x["violations"])
        integ = sum(1 for x in rep["verdicts"].values() if x["integrity"])
        detail = [f"{names.get(c, c.lower())} ({c}) ×{n}" for c, n in codes.most_common()]
        if integ:
            detail.append(f"failed number checks (NUMERICAL_INTEGRITY) ×{integ}")
        examples = [v["message"] for x in rep["verdicts"].values() for v in x["violations"]][:3]
        reason = (f"No feasible, validated strategy after {ws.retry_count} revision(s) "
                  f"(limit {self.cfg.max_revisions}). Rejected for: {', '.join(detail) or 'no reason recorded'}."
                  + (" Example: " + " ".join(examples[:2]) if examples else ""))
        ws.status = "review_required"
        return {"status": "review_required", "headline": "Review required", "reason": reason,
                "action": None, "selected_id": None, "trade_offs": [], "numbers": {},
                "risk": None, "summary": [reason],
                "radio": {"label": "Simulated team radio, not a real message",
                          "text": f"{state.target.code}, stand by. Strategy under review, no call yet."},
                "llm_used": False}

    def _recommend(self, state: RaceState, ws: WorkflowState, sel_id: str) -> Dict[str, Any]:
        sel = ws.candidates[sel_id]
        sims = ws.simulations
        acc = sorted((ws.candidates[c] for c in ws.critic_reports[-1]["accepted"]),
                     key=lambda c: rank_key(sims[c.id].remaining_time_s, c))
        best_t = sims[sel_id].remaining_time_s
        others = [c for c in acc if c.id != sel_id]
        runner = others[0] if others else None
        # rounded once, so every place that shows the margin prints the same digits
        margin = round(sims[runner.id].remaining_time_s - best_t, 3) if runner else None
        trade: List[str] = []
        for fam, name in (("stay_out", "Stay out"), ("pit_now", "Box this lap"), ("delay", "Delayed stop"),
                          ("event_aligned", "Stop timed to the event"), ("two_stop", "Two stops"),
                          ("earliest_open", "Next open pit lap"), ("ai_plan", "AI-proposed plan")):
            if fam == sel.family:
                continue
            fam_c = [c for c in ws.candidates.values() if c.family == fam]
            if not fam_c:
                continue
            ok = [c for c in fam_c if c.id in ws.critic_reports[-1]["accepted"]]
            if ok:
                b = min(ok, key=lambda c: rank_key(sims[c.id].remaining_time_s, c))
                trade.append(f"{name} ({b.label}): {sims[b.id].remaining_time_s - best_t:.2f} s slower")
            else:
                v = ws.critic_reports[-1]["verdicts"][fam_c[0].id]
                msg = (v["violations"] or [{"message": "; ".join(v["integrity"]) or v["status"]}])[0]["message"]
                trade.append(f"{name}: rejected. {msg}")
        sim = sims[sel_id]
        last = sim.stints[-1]
        numbers = {"remaining_time_s": round(best_t, 3), "lap_time_total_s": round(sim.lap_time_total_s, 3),
                   "pit_loss_total_s": round(sim.pit_loss_total_s, 3), "stops": len(sel.stops),
                   "margin_to_next_best_s": round(margin, 3) if margin is not None else None,
                   "next_best": runner.label if runner else None,
                   "final_compound": last["compound"], "final_stint_age_at_flag": last["end_age"],
                   "traffic_laps": sim.traffic_laps}
        pos = ws.positions.get(sel_id)
        if pos:
            numbers["projected_position_in_group"] = pos["position_in_group"]
            numbers["group_size"] = pos["group_size"]
        risk = self.risk(state, sel, margin, ws)
        action = action_label(state, sel)
        summary = [f"Estimated {best_t:.2f} s to the flag: {sim.lap_time_total_s:.2f} s of lapping "
                   f"plus {sim.pit_loss_total_s:.2f} s in the pit lane.",
                   (f"{margin:.2f} s faster than the next-best valid option ({runner.label})." if runner and margin >= 0
                    else (f"{-margin:.2f} s slower than the fastest valid option ({runner.label})." if runner
                          else "It is the only option that passed every check."))]
        if pos:
            summary.append(f"Projected P{pos['position_in_group']} of the {pos['group_size']} simulated cars, "
                           f"by total time (overtaking not modelled).")
        summary.append(f"Final tyres: {last['compound'].title()}, {last['end_age']} laps old at the flag.")
        summary.append(f"Risk {risk['category'].lower()}: {'; '.join(risk['reasons'])}.")
        ws.status = "recommended"
        return {"status": "recommended", "headline": action, "action": action, "selected_id": sel_id,
                "selected_label": sel.label, "family": sel.family,
                "stops": [{"lap": s.lap, "compound": s.compound} for s in sel.stops],
                "numbers": numbers, "trade_offs": trade, "risk": risk, "summary": summary,
                "radio": {"label": "Simulated team radio, not a real message",
                          "text": self._radio(state, sel, margin)}, "reason": None, "llm_used": False}

    def _radio(self, state: RaceState, sel: Strategy, margin: Optional[float]) -> str:
        code = state.target.code
        m = f" Worth {margin:.1f} seconds." if margin is not None and margin > 0 else ""
        if not sel.stops:
            return f"{code}, stay out, stay out. Manage the {state.target.compound.lower()}s to the flag.{m}"
        s = sel.stops[0]
        if s.lap == state.first_lap:
            return f"{code}, box, box. Box this lap for {s.compound.lower()}s.{m}"
        return (f"{code}, extend. Plan is box lap {s.lap} for {s.compound.lower()}s, "
                f"{s.lap - state.completed_laps} laps from now.{m}")

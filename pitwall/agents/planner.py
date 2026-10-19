"""Planner Agent: turns the question + race state into an executable plan.

The plan's `analyses`, `simulation_tools` and `candidate_spec` are consumed directly by the
orchestrator, Research Agent and Executor; revisions change what is actually run next.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import DEFAULT_CONFIG, QUESTIONS, ModelConfig
from ..models import RaceState, Strategy
from ..strategies import CandidateSpec, build_candidates
from ..workflow_state import WorkflowState
from .ai_brief import PLANNER_SCHEMA, PLANNER_SYSTEM, dumps, merge_specs, plan_from_ai, race_brief, results_digest

AGENT = "Planner"


@dataclass
class Plan:
    version: int
    question_key: str
    question: str
    analyses: List[str]
    simulation_tools: List[str]
    spec: CandidateSpec
    rationale: List[str] = field(default_factory=list)
    objective: str = "minimise estimated remaining race time among feasible, validated strategies"
    tie_break: str = ("equal time to 1 ms → fewer stops → later first stop → strategy id")
    author: str = "rules"          # "rules" or the AI model that wrote this version

    def to_dict(self) -> Dict:
        return {"version": self.version, "question_key": self.question_key, "author": self.author,
                "question": self.question, "objective": self.objective,
                "tie_break": self.tie_break, "analyses": list(self.analyses),
                "simulation_tools": list(self.simulation_tools),
                "candidate_spec": self.spec.to_dict(), "rationale": list(self.rationale)}


class PlannerAgent:
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    # ---------------------------------------------------------------- initial plan
    def create_plan(self, state: RaceState, question_key: str, ws: WorkflowState, reasoner=None) -> Plan:
        plan = self._rule_plan(state, question_key)
        if _ai_on(reasoner):
            if question_key not in AI_QUESTIONS:
                ws.record(AGENT, "ai", "rule_plan_kept",
                          "This question fixes which options to compare, so the AI model isn't needed to plan them.")
            else:
                ai_plan = self._ai_plan(state, question_key, ws, reasoner, plan)
                if ai_plan is not None:
                    return ai_plan
        ws.record(AGENT, "agent", "create_plan",
                  f"Plan v1: {len(plan.analyses)} analyses; " + " ".join(plan.rationale),
                  analyses=plan.analyses, tools=plan.simulation_tools, spec=plan.spec.to_dict())
        return plan

    def _ai_plan(self, state: RaceState, question_key: str, ws: WorkflowState, reasoner, rule_plan: Plan
                 ) -> Optional[Plan]:
        if not reasoner.available():
            ws.record(AGENT, "ai", "ai_unavailable", f"AI model unavailable ({reasoner.last_error}); "
                      "the rule-based planner wrote the plan instead.")
            return None
        user = (f"QUESTION: {QUESTIONS[question_key]}\nRACE STATE: {dumps(race_brief(state, self.cfg))}\n"
                f"Reply with JSON in this shape:\n{PLANNER_SCHEMA}")
        data = reasoner.ask_json(PLANNER_SYSTEM, user, purpose="Planner: write the plan", max_tokens=1200)
        if data is None:
            ws.record(AGENT, "ai", "ai_failed", f"AI model didn't return a usable plan ({reasoner.last_error}); "
                      "the rule-based planner wrote the plan instead.")
            return None
        spec, analyses, rationale, notes = plan_from_ai(data, state, self.cfg)
        if spec is None:
            ws.record(AGENT, "ai", "ai_failed", "AI plan unusable (" + "; ".join(notes) + "); rule-based plan used.")
            return None
        keep = []
        if not spec.include_stay_out:
            keep.append("stay out")
        if not spec.include_pit_now:
            keep.append("box this lap")
        spec.include_stay_out = spec.include_pit_now = True   # reference points for every comparison
        why = [f"AI: {r}" for r in rationale] or ["AI: (no reasons given)"]
        if keep:
            notes.append("kept " + " and ".join(keep) + " as reference options")
        why += [f"Checked: {n}" for n in notes]
        plan = Plan(1, question_key, QUESTIONS[question_key], analyses, list(rule_plan.simulation_tools), spec,
                    why, author=reasoner.short)
        ws.record(AGENT, "ai", "create_plan",
                  f"Plan v1 written by the AI model ({reasoner.short}): " + " · ".join(why),
                  analyses=analyses, spec=spec.to_dict())
        return plan

    def _rule_plan(self, state: RaceState, question_key: str) -> Plan:
        cfg = self.cfg
        why: List[str] = []
        has_rain = bool(state.wet_runs()) or state.track_condition == "wet"
        sc_ahead = bool(state.neutral_ahead())
        has_event = has_rain or sc_ahead or state.deg_multiplier != 1.0 or bool(state.events)

        analyses = ["track_status", "tyre_analysis"]
        if state.competitors:
            analyses.append("gap_analysis")
        if has_rain:
            analyses.append("weather_analysis")
            why.append("Rain is on track or forecast, so a weather analysis is included.")
        else:
            why.append("Dry with no rain forecast, so the weather analysis is skipped.")

        tools = ["generate_candidates", "simulate_strategy"]
        if state.competitors and question_key != "compound_choice":
            tools.insert(1, "project_rivals")
            tools.append("project_positions")
            why.append("Other cars are listed, so their strategies, traffic and finishing order are modelled.")

        spec = CandidateSpec()
        event_laps = self._event_laps(state)
        if question_key == "compound_choice":
            spec = CandidateSpec(include_stay_out=False, include_pit_now=True,
                                 compounds=[c for c, n in state.target.available_sets.items() if n > 0])
            why.append("Tyre question: test a stop this lap with every available tyre type.")
        elif question_key == "pit_now_vs_stay_out":
            spec = CandidateSpec(include_stay_out=True, include_pit_now=True)
            why.append("Yes/no question: test only 'box this lap' against 'stay out'.")
        else:
            if question_key == "event_response" and not has_event:
                why.append("No race event is active, so a full strategy review is run instead.")
            spec = CandidateSpec(include_stay_out=True, include_pit_now=True,
                                 delays=[cfg.short_delay_laps], event_laps=event_laps,
                                 wet_transition=state.rain_arrival() is not None,
                                 return_to_slicks=self._dries(state))
            why.append(f"Full review: test stopping this lap, stopping {cfg.short_delay_laps} laps later and "
                       "not stopping" + (", plus stops timed to the race event" if event_laps or spec.wet_transition
                                         else "") + ".")

        return Plan(1, question_key, QUESTIONS[question_key], analyses, tools, spec, why)

    @staticmethod
    def _dries(state: RaceState) -> bool:
        arrival = state.rain_arrival()
        return state.dries_after() is not None or bool(arrival and arrival["end"] < state.total_laps)

    @staticmethod
    def _period_open_lap(state: RaceState, p: Dict) -> Optional[int]:
        laps = [lap for lap in range(max(p["start"], state.first_lap), min(p["end"], state.total_laps - 1) + 1)
                if lap not in state.pit_closed_laps]
        return laps[0] if laps else None

    def _event_laps(self, state: RaceState) -> List[Tuple[int, str]]:
        laps = []
        for p in state.neutral_ahead():
            lap = self._period_open_lap(state, p)
            name = "Safety Car" if p["kind"] == "SC" else "Virtual Safety Car"
            if lap is not None and lap != state.first_lap:
                laps.append((lap, f"stop under the {name} (reduced pit loss)"))
        return laps

    def materialise(self, state: RaceState, plan: Plan, ws: WorkflowState) -> List[Strategy]:
        cands = build_candidates(state, state.target, plan.spec, self.cfg, origin=f"plan_v{plan.version}")
        ws.record(AGENT, "tool", "generate_candidates",
                  f"{len(cands)} options from plan v{plan.version}: " + "; ".join(c.label for c in cands),
                  ids=[c.id for c in cands])
        return cands

    # ---------------------------------------------------------------- revision
    def revise_plan(self, state: RaceState, plan: Plan, requests: List[Dict],
                    ws: WorkflowState, reasoner=None) -> Tuple[Plan, List[str]]:
        codes = set()
        for r in requests:
            codes.add(r["code"])
            codes.update(r.get("violation_codes", []))
        spec, ai_changes, author = copy.deepcopy(plan.spec), [], "rules"
        if _ai_on(reasoner) and plan.question_key in AI_QUESTIONS and reasoner.available():
            user = (f"QUESTION: {plan.question}\nRACE STATE: {dumps(race_brief(state, self.cfg))}\n"
                    f"YOUR CURRENT PLAN: {dumps(_spec_summary(plan.spec))}\n"
                    f"CRITIC FEEDBACK AND RESULTS SO FAR: {dumps(results_digest(ws))}\n"
                    "Revise the plan to answer the Critic. Return the plans to ADD (nothing is removed), "
                    f"as JSON in this shape:\n{PLANNER_SCHEMA}")
            data = reasoner.ask_json(PLANNER_SYSTEM, user, purpose=f"Planner: revise plan v{plan.version}",
                                     max_tokens=1200)
            ai_spec = plan_from_ai(data, state, self.cfg)[0] if data is not None else None
            if ai_spec is None:
                ws.record(AGENT, "ai", "ai_failed", f"AI revision unusable ({reasoner.last_error or 'bad reply'}); "
                          "applying the rule-based fixes only.")
            else:
                spec, added = merge_specs(spec, ai_spec)
                ai_changes = [f"AI {a}" for a in added]
                author = reasoner.short
                ws.record(AGENT, "ai", "revise_plan", "AI model's changes: " + ("; ".join(added) or "nothing new"))
        spec, rule_changes = self._rule_fixes(state, spec, codes)
        changes = ai_changes + rule_changes
        if not changes:
            ws.record(AGENT, "revision", "revise_plan",
                      "No further alternatives can address the Critic's requests", codes=sorted(codes))
            return plan, []
        new_plan = Plan(plan.version + 1, plan.question_key, plan.question, list(plan.analyses),
                        list(plan.simulation_tools), spec,
                        plan.rationale + [f"v{plan.version + 1}: {c}" for c in changes],
                        author=author if ai_changes else plan.author)
        if spec.wet_transition and "weather_analysis" not in new_plan.analyses:
            new_plan.analyses.append("weather_analysis")
            changes.append("scheduled weather analysis")
        ws.record(AGENT, "revision", "revise_plan", f"Plan v{new_plan.version}: " + "; ".join(changes),
                  codes=sorted(codes))
        return new_plan, changes

    def _rule_fixes(self, state: RaceState, spec: CandidateSpec, codes: set) -> Tuple[CandidateSpec, List[str]]:
        """Deterministic minimum fixes for the Critic's request codes (also applied after an AI revision)."""
        cfg = self.cfg
        changes: List[str] = []

        def setflag(name: str, msg: str):
            if not getattr(spec, name):
                setattr(spec, name, True)
                changes.append(msg)

        # Weather fixes first: switching to wet tyres often also resolves stint-length issues,
        # so two-stop candidates are only added once no weather fix is pending.
        weather_fix = False
        if codes & {"SLICKS_ON_WET", "MISSING_WET_TRANSITION"} and (state.rain_arrival() or state.dries_after()):
            n = len(changes)
            if state.rain_arrival():
                setflag("wet_transition", "added stops onto INTERMEDIATE/WET around the rain")
            if self._dries(state):
                setflag("return_to_slicks", "added return-to-slicks stops when the rain ends")
            weather_fix = len(changes) > n
        if "STINT_TOO_LONG" in codes and not weather_fix:
            setflag("two_stop", "added two-stop candidates (one stop exceeds safe tyre life)")
        if "PIT_LANE_CLOSED" in codes or "PIT_NOW_BLOCKED" in codes:
            setflag("shift_closed_to_open", "moved stops on closed pit laps to the next open lap")
        if "COMPOUND_UNAVAILABLE" in codes and spec.compounds:
            spec.compounds = []
            changes.append("restricted compounds to sets actually available")
        if codes & {"TWO_COMPOUND_RULE"} and not spec.include_pit_now:
            spec.include_pit_now = True
            changes.append("added box-now candidates to satisfy the two-compound rule")
        if "MISSING_SC_OPTION" in codes:
            for p in state.neutral_ahead():
                lap = self._period_open_lap(state, p)
                if lap is None:
                    continue
                name = "Safety Car" if p["kind"] == "SC" else "Virtual Safety Car"
                entry = (lap, f"stop under the {name} (reduced pit loss)")
                if lap == state.first_lap and not spec.include_pit_now:
                    spec.include_pit_now = True
                    changes.append(f"added box-this-lap (L{lap}) under the {name}")
                elif lap != state.first_lap and all(lap != x for x, _ in spec.event_laps):
                    spec.event_laps.append(entry)
                    changes.append(f"added a stop on lap {lap} under the {name}")
        if "NO_FEASIBLE_CANDIDATE" in codes and not changes:
            for d in (cfg.short_delay_laps, 2 * cfg.short_delay_laps):
                if d not in spec.delays:
                    spec.delays.append(d)
                    changes.append(f"widened timing search: +{d}-lap delay")
            if not spec.two_stop:
                spec.two_stop = True
                changes.append("added two-stop candidates as a last alternative")

        return spec, changes


AI_QUESTIONS = ("full_strategy", "event_response")


def _ai_on(reasoner) -> bool:
    return reasoner is not None and getattr(reasoner, "mode", "mock") == "ai"


def _spec_summary(spec: CandidateSpec) -> Dict:
    d = spec.to_dict()
    d["event_laps"] = [{"lap": l, "why": w} for l, w in spec.event_laps]
    d["custom_plans"] = [{"stops": [{"lap": l, "tyre": c} for l, c in p["stops"]]} for p in spec.custom_plans]
    return {k: v for k, v in d.items() if v not in (False, [], None)}

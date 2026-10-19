"""Pit Wall Assistant: a chat interface to the agents.

Deterministic routing (works without an AI model): each question is matched to an intent and answered
from the current analysis. 'What if…' questions are turned into a modified copy of the inputs and
the full five-agent workflow is re-run (never saved, the real inputs are untouched).
With an AI model (Groq / Gemini / Ollama) switched on, the model rewrites the exact answers in plain English,
answers unmatched questions from a compact summary of the analysis, and turns unusual scenario wording into a
form the scenario reader understands. What-ifs run all five agents with the same brain.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import DEFAULT_CONFIG, QUESTIONS, ModelConfig
from ..llm import MockReasoner
from ..orchestrator import run_analysis
from .. import scenario as scenario_mod
from ..validation import parse_race_state, validate_race_input

AGENT = "Pit Wall Assistant"

SCENARIO_TRIGGER = (r"what if|what happens if|suppose|imagine|scenario|simulate|run (it|this|that|the agents)|"
                    r"let'?s say|if (there'?s|there is|it|a|the)\b|lap \d+ ?(of|/) ?\d+|\d+ laps? (to go|left)|"
                    r"\b(vsc|virtual|safety car|sc|rain\w*)\b[^?]*\b(laps? \d+|next lap|in \d+ laps?)")

GLOSSARY = {
    "undercut": "Undercut: pitting before the car ahead so your fresh tyres gain enough time to be ahead "
                "once they stop.",
    "overcut": "Overcut: staying out longer than a rival, hoping clean air or their slow out-lap gets you ahead.",
    "safety car": "Safety Car: the field drives slowly behind a pace car after an incident. A pit stop then costs "
                  "about half the usual time, so teams often box.",
    "vsc": "Virtual Safety Car (VSC): everyone must slow to a set delta without a pace car. Stops are cheaper "
           "too. This model treats it like a Safety Car period.",
    "virtual safety car": "Virtual Safety Car (VSC): everyone must slow to a set delta without a pace car. "
                          "Stops are cheaper too. This model treats it like a Safety Car period.",
    "pit loss": "Pit loss: the time lost by driving through the pit lane and changing tyres, compared with "
                "staying on track. It is counted once per stop.",
    "degradation": "Degradation (tyre wear): how much slower a tyre gets each lap it is used. Softs are fastest "
                   "but wear quickest; hards are slower but last longer.",
    "stint": "Stint: the laps driven on one set of tyres, between pit stops.",
    "compound": "Compound: the tyre type. Soft, Medium and Hard are for dry tracks; Intermediate and Wet are for rain.",
    "intermediate": "Intermediate: a grooved tyre for a damp or drying track. Full Wets are for heavy rain.",
    "two-compound rule": "Two-compound rule: in a dry race every car must use at least two different dry "
                         "tyre types. Using wet tyres waives it.",
    "box": "Box: team-radio word for 'pit this lap'.",
    "cliff": "Tyre cliff: the age after which a tyre suddenly loses grip and wear speeds up sharply.",
    "fuel": "Fuel effect: cars get faster as fuel burns off (about 0.06 s per lap in real-race replays here).",
    "drs": "DRS: a flap on the rear wing that opens on straights to help overtaking. This model doesn't simulate it.",
}

FAMILY_WORDS = [
    ("two_stop", r"\b(two|2)[ -]?stops?\b|second stop"),
    ("pit_now", r"\b(pit|box|stop)(ping)? (now|this lap|immediately|right now)\b|\bnow\b.*\b(pit|box)\b"),
    ("stay_out", r"stay(ing)? out|no (more )?stops?|don'?t (pit|box|stop)|not (pit|box|stop)|to the (flag|end)"),
    ("delay", r"\b(later|wait|delay|extend|a few laps)\b"),
]
COMPOUND_WORDS = [("INTERMEDIATE", r"\binter(mediate)?s?\b"), ("WET", r"\b(full )?wets?\b(?! track)"),
                  ("SOFT", r"\bsofts?\b"), ("MEDIUM", r"\bmediums?\b"), ("HARD", r"\bhards?\b")]
FAMILY_NAME = {"pit_now": "boxing this lap", "stay_out": "staying out", "delay": "stopping a few laps later",
               "two_stop": "a two-stop plan", "event_aligned": "a stop timed to the event"}


@dataclass
class ChatContext:
    raw: Dict[str, Any]
    question_key: str
    result: Optional[Dict[str, Any]] = None
    stale: bool = False
    reality: Optional[Dict[str, Any]] = None
    reveal: Optional[Callable[[], Optional[Dict[str, Any]]]] = None
    is_real_race: bool = False


@dataclass
class ChatAnswer:
    text: str
    intent: str
    tools: List[str] = field(default_factory=list)
    source: str = "Rule-based answer (no AI model)"
    charts: Optional[Dict[str, Any]] = None        # data for the chat's scenario charts
    scenario_raw: Optional[Dict[str, Any]] = None  # race inputs of a chat scenario (to load into the app)

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": "assistant", "text": self.text, "intent": self.intent, "tools": self.tools,
             "source": self.source}
        if self.charts:
            d["charts"] = self.charts
        if self.scenario_raw:
            d["scenario_raw"] = self.scenario_raw
        return d


def scenario_charts(result: Dict[str, Any], state) -> Optional[Dict[str, Any]]:
    """Compact data for the plain-English scenario charts (timeline of the race + option gaps)."""
    if result["recommendation"]["status"] != "recommended":
        return None
    sel = result["selected_id"]
    nxt = next((c for c in result["candidates"] if c["status"] == "ACCEPTED"), None)
    plans = [{"name": "Agents' plan", "stints": result["simulations"][sel]["stints"]}]
    if nxt:
        plans.append({"name": "Next-best plan", "stints": result["simulations"][nxt["id"]]["stints"]})
    return {"first": state.first_lap, "total": state.total_laps,
            "neutral": [p for p in state.neutral_ahead()], "wet": state.wet_runs(), "plans": plans,
            "options": [{k: c[k] for k in ("label", "status", "remaining_time_s", "delta_to_best_s", "issues")}
                        for c in result["candidates"]]}


def _fmt_t(s: Optional[float]) -> str:
    if s is None:
        return "–"
    m, sec = divmod(float(s), 60)
    return f"{int(m)}:{sec:05.2f}"


class PitWallAssistant:
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG, reasoner=None):
        self.cfg = cfg
        self.reasoner = reasoner or MockReasoner()

    # ------------------------------------------------------------------ public
    def answer(self, question: str, ctx: ChatContext) -> ChatAnswer:
        q = " ".join((question or "").lower().split())
        if not q:
            return ChatAnswer("Ask me something about the race, for example *Why not pit now?*", "empty")
        ans = self._route(q, ctx)
        return self._maybe_llm(question, ans, ctx)

    # ------------------------------------------------------------------ routing
    def _route(self, q: str, ctx: ChatContext) -> ChatAnswer:
        if re.search(r"\b(hi|hello|hey|help|what can you|how do i use)\b", q) and len(q) < 40:
            return self._help(ctx)
        term = self._glossary_term(q)
        if term:
            return ChatAnswer(GLOSSARY[term], "glossary")
        if re.search(SCENARIO_TRIGGER, q):
            return self._what_if(q, ctx)
        if re.search(r"real(ly)?|actual(ly)?|in real life|what happened|team (did|do)|did the team|reality", q):
            return self._reality(ctx)
        if ctx.result is None or ctx.result.get("status") == "invalid_input":
            return ChatAnswer("There's no analysis to talk about yet. Press **Analyze Strategy** first, or ask me a "
                              "what-if (for example *What if it rains in 5 laps?*) and I'll run the agents for you.",
                              "no_result")
        fam, comps = self._options_in(q)
        if (fam or comps) and re.search(r"why|what about|how about|instead|better|compare|vs|versus|should|could|"
                                        r"would|can we|option|not\b", q):
            return self._why_not(fam, comps, ctx)
        for pattern, fn in (
            (r"critic|reject|rule|valid|check|revision|retry|sent back", self._critic),
            (r"agent|how (does|do) (it|you|this)|workflow|planner|executor|research agent|orchestrat|operations", self._agents),
            (r"risk|safe|confiden|sure|certain|probab", self._risk),
            (r"gap|position|ahead|behind|rival|competitor|undercut|overcut|traffic|finish|place|who", self._gaps),
            (r"rain|weather|wet|forecast", self._weather),
            (r"safety car|\bsc\b|\bvsc\b", self._safety_car),
            (r"tyre|tire|compound|wear|degrad|stint|age", self._tyres),
            (r"time|how long|margin|faster|slower|seconds|pit loss|numbers", self._numbers),
            (r"why|recommend|call|decision|what should|best|plan|explain|summary|summar", self._explain_call),
        ):
            if re.search(pattern, q):
                return fn(ctx)
        if fam or comps:
            return self._why_not(fam, comps, ctx)
        return self._fallback(ctx)

    # ------------------------------------------------------------------ helpers
    def _note(self, ctx: ChatContext) -> str:
        if ctx.result and ctx.stale:
            return (f"\n\n_Note: this is about analysis {ctx.result['analysis_id']}; your inputs changed since then, "
                    "so press Analyze Strategy to refresh._")
        return ""

    def _glossary_term(self, q: str) -> Optional[str]:
        if not re.search(r"what('?s| is| are| does)|meaning|define|definition|explain the term", q):
            return None
        for term in sorted(GLOSSARY, key=len, reverse=True):
            if re.search(r"\b" + re.escape(term) + r"s?\b", q):
                if term in ("safety car",) and re.search(r"what if|if there", q):
                    return None
                return term
        return None

    def _options_in(self, q: str) -> Tuple[Optional[str], List[str]]:
        fam = next((f for f, pat in FAMILY_WORDS if re.search(pat, q)), None)
        comps = [c for c, pat in COMPOUND_WORDS if re.search(pat, q)]
        return fam, comps

    # ------------------------------------------------------------------ intents
    def _help(self, ctx: ChatContext) -> ChatAnswer:
        ex = ["Why this call?", "Why not pit now?", "What about hard tyres?", "What did the Critic reject?",
              "What if a Safety Car comes out next lap?", "What if it rains in 5 laps?", "What is an undercut?"]
        if ctx.is_real_race:
            ex.append("What really happened?")
        return ChatAnswer("I'm the pit-wall assistant. I answer from the current analysis, run *what-if* checks "
                          "through all five agents, and explain racing terms. Try:\n\n"
                          + "\n".join(f"- *{e}*" for e in ex), "help")

    def _explain_call(self, ctx: ChatContext) -> ChatAnswer:
        rec = ctx.result["recommendation"]
        if rec["status"] != "recommended":
            return ChatAnswer(f"**No call was made.** {rec['reason']}" + self._note(ctx), "explain_call")
        lines = [f"**{rec['headline']}.**", ""] + [f"- {x}" for x in rec["summary"]]
        if rec["trade_offs"]:
            lines += ["", "Alternatives:"] + [f"- {x}" for x in rec["trade_offs"][:3]]
        return ChatAnswer("\n".join(lines) + self._note(ctx), "explain_call")

    def _why_not(self, fam: Optional[str], comps: List[str], ctx: ChatContext) -> ChatAnswer:
        res = ctx.result
        cands = res["candidates"]
        defs = res["candidate_defs"]
        sel = res["selected_id"]

        def matches(c):
            d = defs[c["id"]]
            ok_f = fam is None or d["family"] == fam or (fam == "pit_now" and d["stops"] and
                                                       d["stops"][0]["lap"] == res["decision_point"]["completed_laps"] + 1)
            ok_c = not comps or any(s["compound"] in comps for s in d["stops"])
            return ok_f and ok_c
        hits = [c for c in cands if matches(c)]
        what = " and ".join(x for x in (FAMILY_NAME.get(fam, "") if fam else "",
                                        "/".join(c.title() for c in comps) + " tyres" if comps else "") if x)
        if not hits:
            if res["question_key"] == "full_strategy":
                why = ("The Planner only adds two-stop plans when the Critic finds no valid one-stop plan, and here "
                       "one-stop plans passed." if fam == "two_stop" else
                       "It wasn't among the options this plan generated (they depend on the race state).")
                return ChatAnswer(f"The agents didn't test {what or 'that option'}. {why} Ask a what-if such as "
                                  "*What if tyre wear is 1.8x?* to see when it becomes the call." + self._note(ctx),
                                  "why_not")
            return ChatAnswer(f"The Planner didn't test {what or 'that option'} for the question "
                              f"“{QUESTIONS.get(res['question_key'], '')}”. Switch the question to "
                              "*What is the best strategy for the rest of the race?* to test more options."
                              + self._note(ctx), "why_not")
        if any(c["id"] == sel for c in hits):
            c = next(c for c in hits if c["id"] == sel)
            return ChatAnswer(f"**{c['label']}** is the recommendation: it was the fastest option that passed "
                              f"every check ({_fmt_t(c['remaining_time_s'])} to the flag)." + self._note(ctx), "why_not")
        valid = [c for c in hits if c["status"] == "ACCEPTED"]
        lines = []
        if valid:
            b = valid[0]
            lines.append(f"**{b['label']}** passed every check but is **{b['delta_to_best_s']:.2f} s slower** than the "
                         f"call ({_fmt_t(b['remaining_time_s'])} vs {_fmt_t(b['remaining_time_s'] - b['delta_to_best_s'])}). "
                         f"It spends {b['pit_loss_total_s']:.1f} s in the pit lane and finishes on "
                         f"{b['final_compound'].title()} tyres {b['final_stint_age']} laps old.")
        rejected = [c for c in hits if c["status"] == "REJECTED"]
        for c in rejected[: (1 if valid else 2)]:
            lines.append(f"**{c['label']}** was rejected by the Critic. {c['issues']}")
        return ChatAnswer("\n\n".join(lines) + self._note(ctx), "why_not", ["candidate lookup"])

    def _risk(self, ctx: ChatContext) -> ChatAnswer:
        rec = ctx.result["recommendation"]
        if rec["status"] != "recommended":
            return self._explain_call(ctx)
        r = rec["risk"]
        return ChatAnswer(f"Risk is **{r['category'].lower()}**: {'; '.join(r['reasons'])}.\n\nThis is a fixed rule, "
                          f"not a probability. {r['rule']}" + self._note(ctx), "risk")

    def _gaps(self, ctx: ChatContext) -> ChatAnswer:
        res = ctx.result
        g = res["research"].get("gap_analysis")
        lines = list(g["summary"]) if g else ["Gap analysis wasn't part of this plan."]
        pos = res["recommendation"].get("numbers", {}).get("projected_position_in_group")
        if pos:
            lines.append(f"With the recommended plan, the projected finish is P{pos} of "
                         f"{res['recommendation']['numbers']['group_size']} simulated cars (overtaking isn't modelled).")
        if res.get("rivals"):
            lines.append("Rivals are assumed to run their own fastest valid plan: "
                         + "; ".join(f"{c}: {r['strategy_label']}" for c, r in res["rivals"].items()) + ".")
        return ChatAnswer("\n".join(f"- {x}" for x in lines) + self._note(ctx), "gaps")

    def _weather(self, ctx: ChatContext) -> ChatAnswer:
        w = ctx.result["research"].get("weather_analysis")
        if not w:
            return ChatAnswer("There's no rain in this race state, so the Planner skipped the weather analysis. "
                              "Try *What if it rains in 5 laps?* and I'll re-run the agents." + self._note(ctx), "weather")
        return ChatAnswer("\n".join(f"- {x}" for x in w["summary"]) + self._note(ctx), "weather")

    def _safety_car(self, ctx: ChatContext) -> ChatAnswer:
        t = ctx.result["research"].get("track_status", {})
        lines = list(t.get("summary", []))
        lines.append("Try *What if a Safety Car comes out on lap 25?* to see how the call changes.")
        return ChatAnswer("\n".join(f"- {x}" for x in lines) + self._note(ctx), "safety_car")

    def _tyres(self, ctx: ChatContext) -> ChatAnswer:
        t = ctx.result["research"].get("tyre_analysis")
        lines = list(t["summary"]) if t else []
        n = ctx.result["recommendation"].get("numbers", {})
        if n.get("final_compound"):
            lines.append(f"The recommended plan finishes on {n['final_compound'].title()} tyres, "
                         f"{n['final_stint_age_at_flag']} laps old at the flag.")
        return ChatAnswer("\n".join(f"- {x}" for x in lines) + self._note(ctx), "tyres")

    def _numbers(self, ctx: ChatContext) -> ChatAnswer:
        rec = ctx.result["recommendation"]
        if rec["status"] != "recommended":
            return self._explain_call(ctx)
        n = rec["numbers"]
        m = n["margin_to_next_best_s"]
        return ChatAnswer(
            f"- Time to the flag: **{_fmt_t(n['remaining_time_s'])}** ({n['lap_time_total_s']:.1f} s lapping + "
            f"{n['pit_loss_total_s']:.1f} s in the pit lane)\n"
            + (f"- Margin to the next-best valid option: **{m:.2f} s** ({n['next_best']})\n" if m is not None
               else "- It was the only valid option.\n")
            + f"- Stops from here: {n['stops']}; traffic laps after rejoining: {n['traffic_laps']}" + self._note(ctx),
            "numbers")

    def _critic(self, ctx: ChatContext) -> ChatAnswer:
        res = ctx.result
        last = res["critic_reports"][-1]
        lines = [f"The Critic ran {len(res['critic_reports'])} review round(s): {len(last['accepted'])} options valid, "
                 f"{len(last['rejected'])} rejected."]
        examples = [c for c in res["candidates"] if c["status"] == "REJECTED"][:3]
        for c in examples:
            lines.append(f"- **{c['label']}**: {(c.get('issue_list') or [c['issues']])[0]}")
        if res["revisions"]:
            for rv in res["revisions"]:
                lines.append(f"- Revision {rv['revision']}: " + " ".join(rv["actions"]))
        else:
            lines.append("- No revisions were needed this time.")
        return ChatAnswer("\n".join(lines) + self._note(ctx), "critic")

    def _agents(self, ctx: ChatContext) -> ChatAnswer:
        res = ctx.result
        sims = sum(1 for e in res["log"] if e["action"] == "simulate_strategy")
        return ChatAnswer(
            "Five agents share one workspace:\n"
            f"- **Planner** chose {len(res['plan']['analyses']) if res.get('plan') else 0} analyses and the option types "
            f"to test (plan v{len(res['plan_history'])}).\n"
            f"- **Race Research** ran: {', '.join(res['research'])}.\n"
            f"- **Strategy Executor** ran {sims} lap-by-lap simulations.\n"
            f"- **Strategy Critic** checked rules and numbers ({res['retry_count']} revision(s)).\n"
            "- **Pit Wall Operations** picked the fastest valid option, wrote the radio call and saved the record.\n\n"
            "I'm a sixth helper: I answer questions and run what-ifs through those five." + self._note(ctx), "agents")

    def _fallback(self, ctx: ChatContext) -> ChatAnswer:
        return ChatAnswer("I didn't understand that one. I can answer questions like *Why this call?*, *Why not "
                          "pit now?*, *What about soft tyres?*, *What did the Critic reject?*, *What if it rains in "
                          "5 laps?* or *What is a Safety Car?*", "fallback")

    # ------------------------------------------------------------------ reality
    def _reality(self, ctx: ChatContext) -> ChatAnswer:
        if not ctx.is_real_race:
            return ChatAnswer("This scenario is made up, so there's no real race to compare with. Load a real race "
                              "on Race Control (Real races 2019–2026), analyse it, then ask me again.", "reality")
        if ctx.result is None:
            return ChatAnswer("Press **Analyze Strategy** first. I only reveal the real race after the agents have "
                              "made their own call, so hindsight can't influence it.", "reality")
        cmp = ctx.reality
        tools = []
        if cmp is None and ctx.reveal is not None:
            cmp = ctx.reveal()
            tools.append("revealed the real race (laps after the decision point)")
        if cmp is None:
            return ChatAnswer("I couldn't load what happened next in this race. See the message in the "
                              "Race Control page.", "reality", tools)
        lines = [f"**{cmp['verdict']['text']}**", "",
                 f"- Agents: {cmp['model_call']}", f"- Team: {cmp['team_call']} ({cmp['status']}, P{cmp['finish_position']})"]
        lines += [f"- {x}" for x in cmp["explanations"]]
        return ChatAnswer("\n".join(lines), "reality", tools)

    # ------------------------------------------------------------------ scenarios / what-ifs (run the agents)
    def _what_if(self, q: str, ctx: ChatContext, normalised: Optional[str] = None) -> ChatAnswer:
        sc = scenario_mod.parse(normalised or q, ctx.raw)
        if sc.empty and normalised is None:
            normalised = self._ai_normalise(q, ctx)
            if normalised:
                sc = scenario_mod.parse(normalised, ctx.raw)
        if sc.empty:
            return ChatAnswer(
                "Describe the race and I'll run it through all five agents. For example:\n\n"
                "- *After lap 20 of 56 on 18-lap-old mediums, VSC on laps 24–26, rain from lap 35 to 45*\n"
                "- *What if a Safety Car comes out next lap?*\n"
                "- *15 laps to go, raining until lap 50, tyre wear 1.5x*\n\n"
                "I understand the lap, tyres, Safety Cars, VSCs, rain, tyre wear, pit-stop time and a closed pit lane.",
                "what_if")
        base_issues = validate_race_input(ctx.raw)
        if base_issues:
            return ChatAnswer("The current inputs are invalid, so I can't run a scenario yet: "
                              + "; ".join(f"{i['field']}: {i['message']}" for i in base_issues[:3]), "what_if")
        new_raw = scenario_mod.apply(ctx.raw, sc)
        issues = validate_race_input(new_raw)
        if issues:
            return ChatAnswer("I can't run that scenario: " + "; ".join(
                f"{i['field'].replace('_', ' ')} {i['message']}" for i in issues[:3]) + ".", "what_if")
        tools = []
        base = ctx.result
        if base is None or ctx.stale or base.get("status") == "invalid_input":
            base = run_analysis(ctx.raw, ctx.question_key, cfg=self.cfg)
            tools.append("ran the five agents on your current inputs (baseline with rules, not saved)")
        what = run_analysis(new_raw, ctx.question_key, cfg=self.cfg, reasoner=self.reasoner,
                            trigger_event={"type": "chat_scenario", "description": "Chat scenario: " + "; ".join(sc.parts)})
        tools.append("ran the five agents on your scenario (not saved)")
        if normalised:
            tools.insert(0, f"AI model rewrote your scenario as: “{normalised}”")
        state = parse_race_state(new_raw, self.cfg)
        text = self._scenario_text(sc, what, base, state)
        ans = ChatAnswer(text, "what_if", tools)
        ans.charts = scenario_charts(what, state)
        ans.scenario_raw = new_raw
        return ans

    def _plain_plan(self, what: Dict[str, Any], state) -> List[str]:
        rec = what["recommendation"]
        sel = what["selected_id"]
        rows = {r["lap"]: r for r in what["simulations"][sel]["laps"]}
        lines = []
        prev = state.target.compound
        for s in rec["stops"]:
            lap, comp = s["lap"], s["compound"]
            row = rows.get(lap, {})
            kind = row.get("neutral")
            wet_next = state.condition_on_lap(min(lap + 1, state.total_laps)) == "wet"
            is_wet_tyre = comp in ("INTERMEDIATE", "WET")
            if kind:
                name = "Safety Car" if kind == "SC" else "VSC"
                lines.append(f"Pit on lap {lap} under the {name} for {comp.title()} tyres: the stop costs about "
                             f"{row.get('pit_loss_s', 0):.0f} s instead of {state.pit_loss_s:.0f} s.")
            elif is_wet_tyre and wet_next:
                lines.append(f"Switch to {comp.title()}s on lap {lap}, so the car is on wet tyres when the rain "
                             f"arrives (lap {lap + 1} is wet).")
            elif not is_wet_tyre and prev in ("INTERMEDIATE", "WET"):
                lines.append(f"Go back to {comp.title()} slicks on lap {lap} as the track dries.")
            else:
                lines.append(f"Pit on lap {lap} for {comp.title()} tyres: fresh tyres gain more than the "
                             f"{row.get('pit_loss_s', state.pit_loss_s):.0f} s the stop costs.")
            prev = comp
        if not rec["stops"]:
            lines.append("Stay out to the flag: the current tyres last, and a stop would cost more than it gains.")
        used = {s["lap"] for s in rec["stops"]}
        defs = what["candidate_defs"]
        for p in state.neutral_ahead():
            window = set(range(max(p["start"], state.first_lap), p["end"] + 1))
            if used & window:
                continue
            name = "Safety Car" if p["kind"] == "SC" else "VSC"
            tried = [c for c in what["candidates"] if defs[c["id"]]["stops"]
                     and defs[c["id"]]["stops"][0]["lap"] in window]
            if not tried:
                continue
            tried.sort(key=lambda c: c["remaining_time_s"] if c["remaining_time_s"] is not None else 1e9)
            ok = [c for c in tried if c["status"] in ("ACCEPTED", "SELECTED")]
            if ok:
                lines.append(f"Stopping under the {name} (laps {p['start']}–{p['end']}) was tested but is "
                             f"{ok[0]['delta_to_best_s']:.1f} s slower overall.")
            else:
                lines.append(f"Stopping under the {name} (laps {p['start']}–{p['end']}) was tested but rejected: "
                             f"{(tried[0].get('issue_list') or [tried[0]['issues']])[0]}")
        stay = next((c for c in what["candidates"] if c["family"] == "stay_out"), None)
        if rec["stops"] and stay and stay["status"] == "REJECTED":
            lines.append(f"Staying out isn't allowed: {(stay.get('issue_list') or [stay['issues']])[0]}")
        n = rec["numbers"]
        if n.get("margin_to_next_best_s") is not None:
            lines.append(f"Expected time to the flag: {_fmt_t(n['remaining_time_s'])}. The next-best plan "
                         f"({n['next_best']}) is {n['margin_to_next_best_s']:.1f} s slower.")
        else:
            lines.append(f"Expected time to the flag: {_fmt_t(n['remaining_time_s'])}. It was the only plan that "
                         "passed every check.")
        r = rec["risk"]
        lines.append(f"Risk: {r['category'].lower()} ({'; '.join(r['reasons'])}).")
        return lines

    def _scenario_text(self, sc, what: Dict[str, Any], base: Dict[str, Any], state) -> str:
        out = ["**The scenario I gave the agents**"] + [f"- {p}" for p in sc.parts]
        if sc.assumptions:
            out.append(f"- _Assumed: {'; '.join(sc.assumptions)}_")
        if sc.fresh:
            out.append("- _Other Safety Car / VSC / rain inputs were replaced by the ones above._")
        out.append("")
        rec = what["recommendation"]
        if rec["status"] != "recommended":
            out += [f"**What they decided: no safe plan.** {rec['reason']}"]
            return "\n".join(out)
        out.append(f"**What they decided: {rec['headline']}**")
        out += [f"- {x}" for x in self._plain_plan(what, state)]
        b = base["recommendation"]
        if b["status"] == "recommended" and b["headline"] != rec["headline"]:
            out += ["", f"**Compared with your current race inputs:** the call changes from *{b['headline']}*."]
        elif b["status"] == "recommended":
            out += ["", "**Compared with your current race inputs:** the call stays the same."]
        out += ["", f"_The agents tested {len(what['candidates'])} plans with {what['retry_count']} revision(s). "
                    "Not saved; your inputs are unchanged. Use the button below to load this scenario._"]
        return "\n".join(out)

    # ------------------------------------------------------------------ AI model
    def _context_json(self, ctx: ChatContext) -> str:
        res = ctx.result
        if not res or res.get("status") == "invalid_input":
            return "{}"
        rec = res["recommendation"]
        return json.dumps({
            "question": res["question"], "call": rec.get("headline"), "numbers": rec.get("numbers"),
            "summary": rec.get("summary"), "trade_offs": rec.get("trade_offs"),
            "risk": (rec.get("risk") or {}).get("category"),
            "options": [{k: c[k] for k in ("label", "status", "delta_to_best_s", "issues")} for c in res["candidates"][:10]],
            "research": {k: v["summary"] for k, v in res["research"].items()},
        }, default=str)[:6000]

    CHAT_SYSTEM = ("You are the chat assistant on PitWall AI, a Formula 1 race-strategy tool with five agents "
                   "(Planner, Race Research, Strategy Executor, Strategy Critic, Pit Wall Operations). Answer in "
                   "simple English for someone new to F1: 2-6 short sentences or a few bullets, markdown allowed. "
                   "Use ONLY the facts given. Keep every number, lap and tyre exactly as given; never invent "
                   "numbers. If the facts don't answer it, say so and suggest a question you can answer.")

    def _ai_on(self) -> bool:
        return getattr(self.reasoner, "mode", "mock") == "ai"

    def _ai_normalise(self, question: str, ctx: ChatContext) -> Optional[str]:
        """Ask the AI model to rewrite a free-text race scenario in the phrasing the scenario reader understands."""
        if not self._ai_on() or not self.reasoner.available():
            return None
        raw = ctx.raw
        system = ("Convert the user's Formula 1 race scenario into ONE line using only these phrases, and only for "
                  "things the user actually said: 'after lap A of B', 'on N-lap-old softs|mediums|hards|inters|wets', "
                  "'safety car on laps X-Y', 'vsc on laps X-Y', 'rain from lap X to Y', 'tyre wear 1.5x', "
                  "'pit stop costs N seconds', 'pit lane closed on laps X, Y'. Relative timings must be turned into "
                  "lap numbers using the current lap. Reply with JSON only: {\"is_scenario\": true|false, \"line\": \"...\"}")
        user = (f"CURRENT RACE: after lap {raw.get('completed_laps')} of {raw.get('total_laps')}, "
                f"tyres {raw['target'].get('compound')} aged {raw['target'].get('tyre_age_laps')} laps.\n"
                f"USER: {question}")
        data = self.reasoner.ask_json(system, user, purpose="Chat: read the scenario", max_tokens=300)
        if not data or not data.get("is_scenario"):
            return None
        line = " ".join(str(data.get("line", "")).split())[:300]
        return line or None

    def _maybe_llm(self, question: str, ans: ChatAnswer, ctx: ChatContext) -> ChatAnswer:
        r = self.reasoner
        if not self._ai_on():
            return ans
        if not r.available():
            ans.source = f"Rule-based answer (AI model unavailable: {r.last_error})"
            return ans
        if ans.intent in ("help", "glossary", "no_result", "empty"):
            return ans
        if ans.intent == "fallback":
            line = self._ai_normalise(question, ctx)
            if line and not scenario_mod.parse(line, ctx.raw).empty:
                return self._maybe_llm(question, self._what_if(question.lower(), ctx, normalised=line), ctx)
            text = r.write(self.CHAT_SYSTEM, f"FACTS (current analysis): {self._context_json(ctx)}\n"
                                             f"QUESTION: {question}", purpose="Chat: answer", max_tokens=500)
            if text:
                return ChatAnswer(text, "ai_answer", ans.tools, f"AI model ({r.short}), from this analysis only")
            ans.source = f"Rule-based answer (AI model failed: {r.last_error})"
            return ans
        facts = self._context_json(ctx)
        text = r.write(self.CHAT_SYSTEM,
                       f"QUESTION: {question}\nEXACT ANSWER FROM THE AGENTS' RESULTS (rewrite it simply, keep all "
                       f"numbers):\n{ans.text}\nMORE FACTS: {facts}", purpose="Chat: explain", max_tokens=600)
        if not text:
            ans.source = f"Rule-based answer (AI model failed: {r.last_error})"
            return ans
        out = ChatAnswer(text, ans.intent, ans.tools, f"AI model ({r.short}) explaining the agents' exact results")
        out.charts, out.scenario_raw = ans.charts, ans.scenario_raw
        return out

"""Agent-centred views: a live, step-by-step run and an expandable 'given / did / decided' pipeline."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import streamlit as st

from pitwall.config import QUESTIONS
from pitwall.orchestrator import run_analysis

from . import tables
from .help_text import h
from .styles import esc, fmt_race_time, section, tip

AGENTS = [
    {"key": "Planner", "name": "Planner", "icon": "🧭", "help": "agent_planner"},
    {"key": "Race Research", "name": "Race Research", "icon": "🔎", "help": "agent_research"},
    {"key": "Strategy Executor", "name": "Strategy Executor", "icon": "⚙️", "help": "agent_executor"},
    {"key": "Strategy Critic", "name": "Strategy Critic", "icon": "🛡️", "help": "agent_critic"},
    {"key": "Pit Wall Operations", "name": "Pit Wall Operations", "icon": "📋", "help": "agent_ops"},
]
ICON = {a["key"]: a["icon"] for a in AGENTS} | {"Orchestrator": "🔀"}


def md(x: str) -> None:
    st.markdown(x, unsafe_allow_html=True)


def _delay() -> float:
    try:
        return max(0.0, float(os.environ.get("PITWALL_LIVE_DELAY", "0.12")))
    except ValueError:
        return 0.12


# ====================================================================== live run
def _stages_html(active: str, done: set, run_id: str) -> str:
    cells = []
    for i, a in enumerate(AGENTS):
        cls = "active" if a["key"] == active else ("done" if a["key"] in done else "")
        state = "working…" if cls == "active" else ("done" if cls == "done" else "waiting")
        cells.append(f'<div class="pw-stage {cls}"><div class="nm">{a["icon"]} {esc(a["name"])}</div>'
                     f'<div class="st">{state}</div></div>')
    return f'<div class="pw-stages" data-run="{esc(run_id)}">{"".join(cells)}</div>'


def live_run(pending: Dict[str, Any], on_done, compact: bool = False) -> None:
    """Run the agents inside the page so each step appears as it happens, then hand the result back."""
    brain = getattr(pending.get("reasoner"), "short", "Rules")
    md(section("The agents are working", h("live"),
               sub=f"Agent brain: {brain}. Each line appears as an agent acts"
                   + (" (AI steps can take a few seconds each)." if brain != "Rules" else ".")
                   + " Turn off “Watch live” to run without the animation.", first=not compact))
    strip = st.empty()
    done: set = set()
    strip.markdown(_stages_html("Planner", done, "live"), unsafe_allow_html=True)
    delay = _delay()
    with st.status("Running the five agents…", expanded=not compact) as status:
        def progress(entry: Dict[str, Any]) -> None:
            agent = entry["agent"]
            if agent in ICON and agent != "Orchestrator":
                for a in AGENTS:  # agents before the active one in this pass are finished
                    if a["key"] == agent:
                        break
                    done.add(a["key"])
                strip.markdown(_stages_html(agent, done, "live"), unsafe_allow_html=True)
            what = tables.action_label(entry["action"])
            detail = entry["detail"] if len(entry["detail"]) < 170 else entry["detail"][:167] + "…"
            icon = "🧠" if entry.get("kind") == "ai" else ICON.get(agent, "•")
            st.markdown(f"{icon} **{agent}** · {what}" + (f" — {detail}" if detail else ""))
            if delay:
                time.sleep(delay)
        result = run_analysis(pending["raw"], pending["question"], trigger_event=pending.get("trigger"),
                              reasoner=pending.get("reasoner"), progress=progress)
        if result["status"] == "invalid_input":
            status.update(label="Stopped: the inputs are invalid", state="error")
        else:
            done.update(a["key"] for a in AGENTS)
            strip.markdown(_stages_html("", done, "live"), unsafe_allow_html=True)
            status.update(label=f"Done: {result['recommendation']['headline']}", state="complete")
    on_done(result)


# ====================================================================== pipeline (given / did / decided)
def _steps(result, agent: str) -> List[str]:
    return [f"{tables.action_label(e['action'])}" + (f": {e['detail']}" if e["detail"] else "")
            for e in result["log"] if e["agent"] == agent]


def _bul(items) -> str:
    items = [i for i in items if i]
    return "\n".join(f"- {i}" for i in items) if items else "- (nothing)"


def _block(given, did, decided, passed) -> str:
    return (f"**Given**\n{_bul(given)}\n\n**What it did**\n{_bul(did)}\n\n"
            f"**Decided**\n{_bul(decided)}\n\n**Passed on**\n{_bul(passed)}")


def agent_pipeline(result: Dict[str, Any]) -> None:
    snap = result["input_snapshot"]
    rec = result["recommendation"]
    t = snap["target"]
    plans = result["plan_history"]
    research = result["research"]
    reports = result["critic_reports"]
    last = reports[-1] if reports else {"accepted": [], "rejected": [], "revision_requests": []}
    cands = result["candidates"]
    sims = sum(1 for e in result["log"] if e["action"] == "simulate_strategy")

    md(section("Inside each agent", h("pipeline"),
               sub="Open any agent to see what it was given, what it did, what it decided and what it passed on. "
                   "Everything here comes from the real run log."))
    # Input check
    with st.expander(f"🛡️ Input check · {'inputs valid' if not result['input_issues'] else 'inputs invalid'}"):
        md(_block(
            [f"Race state: after lap {snap['completed_laps']} of {snap['total_laps']}, "
             f"{t.get('name', t['code'])} on {t['compound'].title()} tyres ({t['tyre_age_laps']} laps old)",
             f"{len(snap['competitors'])} other cars", f"Question: {result['question']}"],
            _steps(result, "Strategy Critic")[:1],
            ["Inputs are valid, the workflow can start" if not result["input_issues"] else
             "Inputs are invalid, so the workflow stopped"],
            ["Valid race state → Planner"]))
    if result["status"] == "invalid_input":
        return
    # Planner
    p1 = plans[0]
    rev_note = f", revised to v{len(plans)}" if len(plans) > 1 else ""
    who = f" · written by {p1.get('author')}" if p1.get("author", "rules") != "rules" else " · rule-based"
    with st.expander(f"🧭 Planner · plan v1{rev_note}{who}: {len(p1['analyses'])} analyses, "
                     f"{len(result['candidate_defs'])} options to test"):
        state_facts = [f"Question: “{QUESTIONS.get(result['question_key'], '')}”",
                       f"Track {snap['track_condition']}, tyre wear ×{snap['deg_multiplier']:g}"]
        tsum = research.get("track_status", {}).get("summary", [])[1:]
        state_facts += tsum[:3]
        if research.get("weather_analysis"):
            state_facts += research["weather_analysis"]["summary"][1:2]
        decided = [f"Plan v{p['version']}: analyses = {', '.join(a.replace('_', ' ') for a in p['analyses'])}"
                   for p in plans] + plans[-1]["rationale"]
        md(_block(state_facts, _steps(result, "Planner"), decided,
                  ["Analyses to run → Race Research", f"{len(result['candidate_defs'])} options → Strategy Executor"]))
    # Research
    with st.expander(f"🔎 Race Research · ran {len(research)} analyses"):
        findings = [x for f in research.values() for x in f["summary"]]
        md(_block([f"Analyses requested by the Planner: {', '.join(k.replace('_', ' ') for k in research)}"],
                  [s.split(":")[0] for s in _steps(result, "Race Research")], findings,
                  ["Findings → Strategy Executor and Pit Wall Operations"]))
    # Executor
    top = sorted([c for c in cands if c["remaining_time_s"] is not None], key=lambda c: c["remaining_time_s"])[:5]
    with st.expander(f"⚙️ Strategy Executor · {sims} lap-by-lap simulations"):
        given = [f"{len(cands)} options from the Planner", "Tools: lap-time simulator"
                 + (", rival projection, finishing-order projection" if result.get("rivals") else "")]
        did = [s for s in _steps(result, "Strategy Executor") if not s.startswith("Simulated an option")][:2]
        did.append(f"Simulated {sims} options lap by lap (each lap: base pace + tyres + weather + traffic + "
                   "fuel + Safety Car/VSC)")
        decided = [f"{c['label']}: {fmt_race_time(c['remaining_time_s'])} to the flag" for c in top]
        md(_block(given, did, decided, ["Times for every option → Strategy Critic"]))
    # Critic
    with st.expander(f"🛡️ Strategy Critic · {len(last['accepted'])} valid, {len(last['rejected'])} rejected, "
                     f"{result['retry_count']} revision(s)"):
        did = []
        for r in reports:
            reqs = "; ".join(f"asked the {q['target'].title()} to fix: {q['reason']}" for q in r["revision_requests"])
            did.append(f"Round {r['iteration']}: {len(r['accepted'])} valid, {len(r['rejected'])} rejected"
                       + (f". {reqs}" if reqs else ". No changes needed."))
        rejected = [c for c in cands if c["status"] == "REJECTED"][:4]
        decided = [f"Rejected {c['label']}: {(c.get('issue_list') or [c['issues']])[0]}" for c in rejected]
        decided.append(f"{len(last['accepted'])} option(s) passed every rule and number check")
        passed = ["Valid options → Pit Wall Operations"]
        if result["revisions"]:
            passed.insert(0, "Work sent back for revision: " + " | ".join(" ".join(rv["actions"])
                                                                           for rv in result["revisions"]))
        md(_block(["Every option and its simulated numbers",
                   "Rules: race window, open pit lane, tyre sets, safe tyre life, two-compound rule, "
                   "no slicks on a wet track, number checks"], did, decided, passed))
    # Operations
    with st.expander(f"📋 Pit Wall Operations · {rec['headline']}"):
        if rec["status"] == "recommended":
            n = rec["numbers"]
            decided = [f"Call: {rec['headline']}", rec.get("decided_by", ""),
                       f"Time to the flag {fmt_race_time(n['remaining_time_s'])}",
                       f"Risk: {rec['risk']['category'].lower()} ({'; '.join(rec['risk']['reasons'])})",
                       f"Team radio: “{rec['radio']['text']}”"]
        else:
            decided = ["Review required: " + rec["reason"]]
        rule = ("AI model picks from the Critic-approved options within 2 s of the fastest and explains why"
                if rec.get("ai") else "Rule: pick the lowest time to the flag; ties go to fewer stops, then the later stop")
        md(_block([f"{len(last['accepted'])} valid option(s) from the Critic", rule],
                  _steps(result, "Pit Wall Operations"), decided,
                  [f"Decision record {result['analysis_id']} saved to the history"]))
    with st.expander("📜 Full activity log (every step, in order)"):
        st.dataframe(tables.log_frame(result), hide_index=True, width="stretch", height=380,
                     column_config=tables.log_config())


# ====================================================================== pipeline diagram
def _badge(kind: str) -> str:
    cls, text = {"ai": ("ai", "🧠 AI model"), "rules": ("rule", "Rules"), "py": ("py", "Python tools"),
                 "ml": ("py", "Python + ML tyres")}[kind]
    return f'<span class="pw-badge {cls}">{text}</span>'


def pipeline_diagram(result: Dict[str, Any]) -> None:
    """Six-step flow (input check → five agents) with each step's status, brain and key output."""
    ok, warn, bad, best = "var(--pw-ok)", "var(--pw-warn)", "var(--pw-bad)", "var(--pw-best)"
    rec = result["recommendation"] or {}
    invalid = result["status"] == "invalid_input"
    plans = result.get("plan_history") or []
    p1 = plans[0] if plans else {}
    reports = result.get("critic_reports") or []
    last = reports[-1] if reports else {"accepted": [], "rejected": []}
    sims = sum(1 for e in result["log"] if e["action"] == "simulate_strategy")
    ml = bool((result["input_snapshot"].get("tyre_model") or {}).get("curves"))
    ai_plan = bool(p1) and p1.get("author", "rules") != "rules"
    nodes = [
        ("🛡️", "Input check", "rules", (bad if invalid else ok),
         "Inputs invalid, run stopped" if invalid else "All inputs valid"),
    ]
    if not invalid:
        nodes += [
            ("🧭", "Planner", "ai" if ai_plan else "rules", warn if len(plans) > 1 else ok,
             f"Plan v{len(plans)} · {len(result['candidate_defs'])} options"
             + ('<br><span class="x">written by the AI</span>' if ai_plan else "")),
            ("🔎", "Race Research", "py", ok, f"{len(result['research'])} analyses"),
            ("⚙️", "Strategy Executor", "ml" if ml else "py", ok,
             f"{sims} lap-by-lap simulations" + ('<br><span class="x">ML tyre model</span>' if ml else "")),
            ("🛡️", "Strategy Critic", "rules", warn if result["retry_count"] else ok,
             f"{len(last['accepted'])} valid · {len(last['rejected'])} rejected"),
            ("📋", "Pit Wall Operations", "ai" if rec.get("ai") else "rules",
             best if rec.get("status") == "recommended" else bad, esc(rec.get("headline", ""))),
        ]
    cells = []
    for i, (icon, name, brain, color, status) in enumerate(nodes):
        last = " last" if i == len(nodes) - 1 else ""
        cells.append(f'<div class="pw-node{last}" style="--i:{i};--c:{color}"><div class="bar"></div>'
                     f'<div class="top"><span class="ic">{icon}</span>{_badge(brain)}</div>'
                     f'<div class="nm">{esc(name)}</div><div class="st">{status}</div></div>')
    loop = ""
    if not invalid:
        n = result["retry_count"]
        loop = (f'<div class="pw-loop"><span>↺ Critic sent work back {n}× (max {result["max_revisions"]})</span></div>'
                if n else '<div class="pw-loop none"><span>No revision needed</span></div>')
    md(f'<div class="pw-flow" data-run="{esc(result["analysis_id"])}">{"".join(cells)}{loop}</div>')


def ai_calls(result: Dict[str, Any]) -> None:
    calls = (result.get("ai") or {}).get("calls") or []
    if not calls:
        return
    with st.expander(f"🧠 Calls to the AI model in this run ({len(calls)})"):
        st.dataframe([{"Step": c["purpose"], "Result": "✓ used" if c["ok"] else f"✕ {c['error']}",
                       "Seconds": c["latency_s"], "Tokens": c.get("tokens")} for c in calls],
                     hide_index=True, width="stretch")
        st.caption("Each call sends a short, factual race brief and gets JSON back. Every answer is checked "
                   "before it is used; anything unusable is replaced by the rules (see the activity log).")


def call_banner(result: Dict[str, Any]) -> None:
    rec = result["recommendation"]
    ok = rec["status"] == "recommended"
    md(f'<div class="pw-call reveal{"" if ok else " review"}" data-run="{esc(result["analysis_id"])}">'
       f'<div class="q">The agents\' answer to “{esc(result["question"])}”{tip(h("call"))}</div>'
       f'<div class="act">{esc(rec["headline"])}</div>'
       f'<div class="meta">{esc(rec.get("decided_by", ""))} · numbers and the radio call are on Race Control.'
       f'</div></div>')

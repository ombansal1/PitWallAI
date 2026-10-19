"""The eight pages of the app (each one job). Files in views/ call these functions."""
from __future__ import annotations

import streamlit as st

from pitwall.data_sources import fastf1_source as ff1src

from . import agents_view, components as c
from .help_text import h
from .model_lab import model_lab as _model_lab
from .state import finish_run, store
from .styles import esc, kpis, page_title, section, tip

ss = st.session_state


def md(x: str) -> None:
    st.markdown(x, unsafe_allow_html=True)


def _result():
    return st.session_state.get("last_result")


def _link_row(*pages) -> None:
    cols = st.columns(len(pages))
    for col, (path, label, icon) in zip(cols, pages):
        col.page_link(path, label=label, icon=icon)


# ====================================================================== Race Control
def race_control() -> None:
    md(page_title("Race Control", "Pick a moment in a race, let the five agents decide the pit-stop call, then "
                                  "compare it with what really happened."))
    with st.container(border=True):
        c.race_picker()
        c.race_summary()
    with st.container(border=True):
        c.ask_agents()
    res = _result()
    if res is None:
        st.info("Press **Analyze Strategy** to run the five agents. With **Watch live** on, you'll see each agent "
                "work on the Agents page.")
        return
    if res["status"] == "invalid_input":
        return
    md(section("The call", h("call")))
    c.call_card(res)
    c.call_details(res)
    c.reality_panel(res)
    md('<div class="pw-divider"></div>')
    _link_row(("views/agents.py", "See how the agents decided", ":material/hub:"),
              ("views/strategy_lab.py", "Compare every option", ":material/science:"),
              ("views/chat.py", "Ask the pit wall about it", ":material/forum:"))


# ====================================================================== Agents
def agents() -> None:
    md(page_title("Agents", "Five agents share one workspace: the Planner decides what to test, Race Research "
                            "measures, the Executor simulates, the Critic checks and can send work back, and Pit "
                            "Wall Operations makes the call."))
    pending = st.session_state.get("pending_run")
    if pending and pending.get("where", "agents") == "agents":
        st.session_state["pending_run"] = None      # clear first: a failure can never loop
        agents_view.live_run(pending, lambda r: finish_run(r, pending.get("message", "")))
        st.rerun()
    res = _result()
    if res is None:
        st.info("No analysis yet. Go to **Race Control** and press **Analyze Strategy** to watch the agents work.")
        st.page_link("views/race_control.py", label="Go to Race Control", icon=":material/sports_score:")
        _how_agents()
        return
    md(section("The pipeline for this decision", h("pipeline"), first=True,
               sub="Each box is one agent. The badge shows what does the thinking: the AI model, fixed rules, or "
                   "Python calculation."))
    agents_view.pipeline_diagram(res)
    agents_view.call_banner(res)
    agents_view.ai_calls(res)
    agents_view.agent_pipeline(res)


def _how_agents() -> None:
    md(section("How a decision is made", h("agents")))
    md('<ol class="pw-steps">'
       '<li><b>Critic checks the inputs.</b> Missing, impossible or non-numeric values stop the run.</li>'
       '<li><b>Planner</b> (AI model or rules) chooses which analyses to run and which pit-stop plans to test.</li>'
       '<li><b>Race Research</b> measures tyres, gaps, weather and track status.</li>'
       '<li><b>Strategy Executor</b> simulates every plan lap by lap (formula or ML tyre model).</li>'
       '<li><b>Strategy Critic</b> checks the racing rules and every number; it can send work back twice.</li>'
       '<li><b>Pit Wall Operations</b> (AI model or rules) picks the call, explains it and saves the record.</li>'
       '</ol>')


# ====================================================================== Chat
def chat() -> None:
    md(page_title("Ask the Pit Wall", "Ask about the call, compare options, or describe a whole scenario (lap, "
                                      "tyres, Safety Car, VSC, rain). Scenarios run through all five agents and "
                                      "come back in plain English with a chart."))
    c.chat_panel(_result())


# ====================================================================== Strategy Lab
def strategy_lab() -> None:
    md(page_title("Strategy Lab", "Throw race events at the agents and compare every option they tested."))
    pending = st.session_state.get("pending_run")
    with st.container(border=True):
        c.events_panel()
        if pending and pending.get("where") == "lab":
            st.session_state["pending_run"] = None
            agents_view.live_run(pending, lambda r: finish_run(r, pending.get("message", "")), compact=True)
            st.rerun()
    res = _result()
    if res is None or res["status"] == "invalid_input":
        st.info("Run an analysis on Race Control first (or apply an event above, which runs one).")
        return
    comp = st.session_state.get("comparison")
    if comp:
        md(section("What the event changed", h("compare")))
        tag = '<span class="tag chg">Call changed</span>' if comp["changed"] else \
            '<span class="tag same">Call unchanged</span>'
        md(f'<div class="pw-ba">{tag}<div class="row"><span>Cause</span><div>{esc(comp["event"].capitalize())}</div>'
           f'</div><div class="row"><span>Before</span><div>{esc(comp["previous_action"])}</div></div>'
           f'<div class="row"><span>Now</span><div>{esc(comp["current_action"])}</div></div>'
           f'<div class="why">{esc(comp["reason"])}</div></div>')
    md(section("Current call", h("call")))
    c.call_card(res)
    if res["recommendation"]["status"] == "recommended":
        c.strategy_charts(res)
    c.options_table(res)


# ====================================================================== Model Lab
def model_lab() -> None:
    md(page_title("Model Lab", "The AI model that drives the agents, the machine-learning tyre model trained on "
                               "real laps, and how the agents' calls compare with real teams."))
    _model_lab()


# ====================================================================== Race Setup
def setup() -> None:
    md(page_title("Race Setup", "Every input the agents use. Edit anything; results are marked out of date "
                                "until you analyse again. Use 0 for 'not set'."))
    c.inputs_editor()
    base = st.session_state["raw_base"]
    md(section("Where these inputs come from", h("provenance"), sub=base.get("data_source_label", "")))
    from . import tables
    st.dataframe(tables.provenance_frame(base.get("provenance", {})), hide_index=True, width="stretch",
                 column_config={"Type": st.column_config.TextColumn(
                     "Type", help="Retrieved = real data. Derived = calculated from real data. "
                                  "Assumption = synthetic value. User / event = entered here.")})
    for n in base.get("notes", []):
        st.caption(n)


# ====================================================================== Records
def records() -> None:
    md(page_title("Records", "Every analysis is saved automatically as a decision record, with its inputs, "
                             "checks and result. Real-race reveals are saved as separate reality checks."))
    s = store()
    recs = s.load_all()
    dec = [r for r in recs if r.get("record_type", "decision") == "decision"]
    rc = [r for r in recs if r.get("record_type") == "reality_check"]
    ok = sum(1 for r in dec if r.get("status") == "recommended")
    ai = sum(1 for r in dec if (r.get("ai") or {}).get("used"))
    match = sum(1 for r in rc if (r.get("verdict") or {}).get("code") == "match")
    md(kpis([("Decisions saved", str(len(dec)), "", "best", h("history")),
             ("Calls made", str(ok), f"{len(dec) - ok} needed review", "good", ""),
             ("Decided with the AI model", str(ai), "", "ai", ""),
             ("Real-race checks", str(len(rc)), f"{match} matched the team" if rc else "none yet", "", h("reality"))],
            cols=4))
    c.history_table()
    c.downloads(_result())
    with st.expander("Delete saved records"):
        st.caption("Deletes every decision record stored on this computer (data/decisions).")
        from .state import clear_history_cb
        st.button("Clear decision history", on_click=clear_history_cb, key="clear_hist_btn")


# ====================================================================== About
def about() -> None:
    md(page_title("About PitWall AI", "An agentic AI proof of concept for Formula 1 race strategy (college "
                                      "project). All numbers are estimates from a simplified model."))
    a, b = st.columns(2, gap="large")
    with a:
        md(section("Architecture", h("agents"), first=True))
        md('<div class="pw-formula">'
           '<b>Streamlit UI</b> (8 pages) → <b>Orchestrator</b> (shared workspace, routing, max 2 revisions)<br>'
           '→ 🧭 Planner <span class="pw-chip ai">AI model</span> → 🔎 Race Research <span class="pw-chip">Python</span>'
           ' → ⚙️ Executor <span class="pw-chip">simulator + ML tyres</span><br>'
           '→ 🛡️ Critic <span class="pw-chip">rules</span> ↺ back to Planner/Executor → 📋 Operations '
           '<span class="pw-chip ai">AI model</span> → decision record (JSON Lines)<br>'
           '<span style="color:var(--pw-muted)">AI model: Groq (OpenAI gpt-oss-120b), Gemini or Ollama, through one '
           'OpenAI-compatible client. ML: scikit-learn gradient boosting trained on 139k real laps.</span></div>')
        md(section("How a lap time is estimated", h("model")))
        md('<div class="pw-formula">Lap time = base pace + tyre effect + weather + traffic + fuel/track effect '
           '+ Safety Car / VSC slow-down<br>Tyre effect: formula (config.py) or the ML model\'s learned curve '
           'for that circuit<br>Pit-stop loss: normal under green, ×0.5 under a Safety Car, ×0.6 under a VSC<br>'
           'Time to the flag = all remaining lap times + pit-lane time for each stop</div>')
        md(section("Rules the Critic enforces", h("agent_critic")))
        md('<ul class="pw-list"><li>Stops inside the remaining race, never on a closed pit lane.</li>'
           '<li>A tyre set must be available for every stop.</li><li>No tyre past its safe life (shorter at high '
           'wear).</li><li>Two different dry tyre types in a dry race, unless wet tyres are used.</li>'
           '<li>No more than two laps on slicks on a wet track.</li><li>Every number finite, adding up exactly, '
           'each stop counted once.</li></ul>')
    with b:
        md(section("Real races vs FastF1", h("sources"), first=True))
        md('<ul class="pw-list"><li><b>Real races (built in):</b> 162 races from 2019 to Singapore 2026 are saved '
           'inside the project (about 14 MB). Real drivers, lap times, tyres, gaps and pit stops. Instant and '
           'offline. Source: the TracingInsights archive (MIT licence), itself made with FastF1.</li>'
           '<li><b>FastF1:</b> a Python library that downloads timing from F1\'s servers while the app runs. Same '
           'kind of data, but needs internet and <code>pip install fastf1</code>; useful for races newer than the '
           f'built-in set. It is {"installed" if ff1src.fastf1_available() else "not installed, so its option is hidden"}.'
           '</li><li><b>Example scenarios:</b> four made-up races with fictional drivers.</li></ul>')
        md(section("No hindsight (unless you ask)", h("reality")))
        md('<ul class="pw-list"><li>The race state uses only laps up to the decision point.</li>'
           '<li>After the agents decide, “Reveal what really happened” reads the rest of the race and scores the '
           'team\'s real strategy with the same model.</li><li>The optional hindsight switch gives the agents the '
           'real Safety Cars, VSCs and rain; it is labelled everywhere.</li></ul>')
        md(section("Honest limits", "What this proof of concept does not do."))
        md('<ul class="pw-list"><li>Simplified physics: no overtaking model, DRS, tyre temperature or damage.</li>'
           '<li>Plan shapes are limited (now, +3 laps, events, AI custom plans).</li>'
           '<li>Rain and Safety Cars are user-given forecasts or hindsight; they are never predicted.</li>'
           '<li>The AI model can be slow or rate-limited on free plans; the rules take over automatically.</li>'
           '<li>Not a live tool and not for real racing decisions.</li></ul>')
    md(section("Tyre formula parameters", "Used when the tyre model is set to Formula (pitwall/config.py)."))
    from pitwall.config import DEFAULT_CONFIG
    from . import tables
    st.dataframe(tables.compounds_frame(DEFAULT_CONFIG), hide_index=True, width="stretch",
                 column_config=tables.compounds_config())
    md(f'<p class="pw-side-note">Help bubbles {tip("Like this one: hover any ? for a plain-English explanation.")} '
       'appear across the site.</p>')

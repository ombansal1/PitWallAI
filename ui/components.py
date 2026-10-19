"""Reusable page pieces (read session state, draw widgets). Pages compose them in ui/pages.py."""
from __future__ import annotations

import json
from typing import Dict, List

import streamlit as st

from pitwall.config import COMPOUNDS, DEFAULT_CONFIG, QUESTIONS
from pitwall.data_sources import fastf1_source as ff1src
from pitwall.data_sources import replays as rp
from pitwall.data_sources.synthetic import list_scenarios
from pitwall.validation import validate_race_input

from . import charts, tables
from .help_text import h
from .state import (COMP_COLS, PERIOD_TYPES, ai_status, analyze_cb, apply_event_cb, build_raw, chat_clear_cb,
                    chat_submit_cb, chat_suggest_cb, editor_key, load_chat_scenario_cb, load_raw_into_widgets,
                    load_real_race_cb, periods_key, reality_status, replay_session, reveal_cb, store,
                    use_fallback_cb)
from .styles import esc, fmt_race_time, section, step_head, tip, tyre_chip

SOURCES = ["Example scenarios", "Real races (2019–2026)"]
FASTF1_SOURCE = "FastF1 download (optional)"


def md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


# ====================================================================== 1. race moment picker
@st.cache_data(show_spinner=False, ttl=600)
def _schedule(year: int):
    """Cached (result, error) so an offline failure is not retried on every rerun."""
    try:
        return ff1src.get_race_schedule(year), None
    except ff1src.DataSourceError as e:
        return None, str(e)


@st.cache_resource(show_spinner=False, max_entries=3)
def _session(year: int, rnd: int):
    return ff1src.load_session(year, rnd)  # FastF1 disk cache + this in-memory cache


def source_options() -> List[str]:
    return SOURCES + ([FASTF1_SOURCE] if ff1src.fastf1_available() else [])


def race_picker() -> None:
    ss = st.session_state
    opts = source_options()
    if ss.get("w_source") not in opts:
        ss["w_source"] = SOURCES[0]
    md(step_head(1, "Choose the race moment", h("data_source")))
    st.radio("Race data", opts, key="w_source", horizontal=True, label_visibility="collapsed",
             help=h("data_source"))
    if ss["w_source"] == SOURCES[0]:
        scen = {s["id"]: s["name"] for s in list_scenarios()}
        c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
        c1.selectbox("Scenario", list(scen), format_func=lambda k: scen[k], key="w_scenario",
                     on_change=_scenario_changed, help=h("scenario"))
        c2.caption("Made-up races with fictional drivers, useful to see how the agents react to rain, "
                   "Safety Cars or worn tyres.")
    elif ss["w_source"] == SOURCES[1]:
        _real_race_controls()
    else:
        _fastf1_controls()


def _scenario_changed():
    from .state import on_scenario_change
    on_scenario_change()


def _real_race_controls() -> None:
    try:
        index = rp.load_index()
    except ff1src.DataSourceError as e:
        st.error(str(e))
        return
    ss = st.session_state
    years = rp.seasons(index)
    c1, c2, c3 = st.columns([1, 2, 2])
    year = c1.selectbox("Season", years, key="rr_year", help="Championship year.")
    races = rp.races_for(year, index)
    labels = {r["id"]: f"{r['event'].replace(' Grand Prix', ' GP')} ({r['date'] or ''})" for r in races}
    if ss.get("rr_race") not in labels:   # season changed: start from its first race
        ss["rr_race"] = next(iter(labels))
    rid = c2.selectbox("Race", list(labels), format_func=lambda k: labels[k], key="rr_race",
                       help="Every race with complete timing data for this season.")
    try:
        sess = replay_session(rid)
    except ff1src.DataSourceError as e:
        st.error(str(e))
        return
    drivers = rp.driver_options(sess)
    names = {d["code"]: f"{d['name']} ({d['code']}), {d['team']}" if d["team"] else f"{d['name']} ({d['code']})"
             for d in drivers}
    if ss.get("rr_driver") not in names:
        ss["rr_driver"] = next(iter(names))
    c3.selectbox("Driver", list(names), format_func=lambda k: names[k], key="rr_driver",
                 help="Listed in the order they finished according to the timing data.")
    total = sess.total_laps
    if not 2 <= int(ss.get("rr_lap", 0)) <= total - 2:
        ss["rr_lap"] = max(2, total // 3)
    a, b, c = st.columns([3, 2, 1.3], vertical_alignment="bottom")
    a.slider("Decision point (after lap)", 2, max(3, total - 2), key="rr_lap", help=h("decision_point"))
    b.toggle("Hindsight: give the agents this race's real Safety Cars, VSCs and rain", key="rr_real_events",
             help=h("real_events"))
    c.button("Load this moment", type="primary", on_click=load_real_race_cb, width="stretch",
             help="Builds the race state from real laps up to the chosen lap.")
    st.caption(f"{len(index['races'])} real races built in (works offline). The agents only see laps up to "
               "your decision point; afterwards you can reveal what really happened.")


def _fastf1_controls() -> None:
    ss = st.session_state
    st.caption("Downloads a race from F1's timing servers with the FastF1 library (needs internet). The "
               "built-in real races are the same kind of data, already saved in the project.")
    c1, c2 = st.columns([1, 3])
    ss.setdefault("ff1_year", 2023)
    year = c1.number_input("Season", 2018, 2026, step=1, key="ff1_year", help="Championship year.")
    sched, err = _schedule(int(year))
    if err:
        st.error(err)
        st.button("Use synthetic fallback", on_click=use_fallback_cb)
        return
    if sched.empty:
        st.info("No completed races for this season.")
        return
    events = {int(r.RoundNumber): r.EventName for r in sched.itertuples()}
    rnd = c2.selectbox("Race", list(events), format_func=lambda k: f"Round {k}: {events[k]}", key="ff1_round")
    if st.button("Download race timing", key="ff1_load_btn", help="Downloads (or reads cached) lap timing."):
        ss["ff1_loaded"], ss["ff1_error"] = (int(year), int(rnd)), None
    if ss.get("ff1_error"):
        st.error(ss["ff1_error"])
        st.button("Use synthetic fallback", on_click=use_fallback_cb)
        return
    key = ss.get("ff1_loaded")
    if not key:
        return
    try:
        with st.spinner("Loading lap timing…"):
            session = _session(*key)
    except ff1src.DataSourceError as e:
        ss["ff1_loaded"], ss["ff1_error"] = None, str(e)  # don't retry automatically on reruns
        st.error(str(e))
        st.button("Use synthetic fallback", on_click=use_fallback_cb)
        return
    try:
        drivers = ff1src.session_drivers(session)
        total = int(session.total_laps)
    except Exception as e:
        st.error(f"Session data incomplete: {e}")
        st.button("Use synthetic fallback", on_click=use_fallback_cb)
        return
    a, b, c = st.columns([1, 3, 1.3], vertical_alignment="bottom")
    drv = a.selectbox("Driver", drivers, key="ff1_driver", help="The car you are making the decision for.")
    if not 2 <= int(ss.get("ff1_lap", 0)) <= total - 1:
        ss["ff1_lap"] = min(20, total - 1)
    lap = b.slider("Decision point (after lap)", 2, total - 1, key="ff1_lap", help=h("decision_point"))
    c.button("Use this moment", type="primary", on_click=_ff1_build, args=(session, drv, lap), width="stretch")


def _ff1_build(session, driver, lap):
    from .state import flash
    try:
        raw = ff1src.build_race_state(session, driver, int(lap), DEFAULT_CONFIG)
        raw["replay"]["round"] = int(st.session_state.get("ff1_round", 0))
        load_raw_into_widgets(raw)
        st.session_state["input_errors"] = []
        flash("success", f"Historical replay loaded: {raw['scenario_name']}.")
    except ff1src.DataSourceError as e:
        flash("error", f"FastF1 data couldn't be used: {e}. Choose another lap or driver, or the synthetic fallback.")


# ====================================================================== race tower
def race_summary() -> None:
    ss = st.session_state
    done, total = ss["w_completed_laps"], ss["w_total_laps"]
    pct = max(0, min(100, 100 * done / total)) if total else 0
    wet = ss["w_track_condition"] == "wet"
    raw = build_raw()
    neutral = ([{"kind": "SC", "start": raw["sc_start_lap"], "end": raw["sc_end_lap"]}] if raw.get("sc_start_lap") else [])
    neutral += [p for p in raw.get("neutral_periods") or [] if isinstance(p.get("start"), int)]
    neutral.sort(key=lambda p: p["start"])
    sc = ("None" if not neutral else f"{'SC' if neutral[0]['kind'] == 'SC' else 'VSC'}<small>L{neutral[0]['start']}–"
          f"{neutral[0]['end']}" + (f" +{len(neutral) - 1}" if len(neutral) > 1 else "") + "</small>")
    rains = ([(raw["rain_start_lap"], raw.get("rain_end_lap"))] if raw.get("rain_start_lap") else [])
    rains += [(p["start"], p["end"]) for p in raw.get("wet_periods") or [] if isinstance(p.get("start"), int)]
    rains.sort(key=lambda r: r[0])
    rain = ("None" if not rains else (f"L{rains[0][0]}–{rains[0][1]}" if rains[0][1] else f"L{rains[0][0]}+")
            + (f" <small>+{len(rains) - 1}</small>" if len(rains) > 1 else ""))
    tcode = ss["raw_base"]["target"]["code"]

    def cell(label, key, value, extra="", cls=""):
        return (f'<div class="pw-cell"><div class="k">{esc(label)}{tip(h(key))}</div>'
                f'<div class="v {cls}">{value}</div>{extra}</div>')
    md('<div class="pw-tower">'
       + cell("Lap", "lap", f'{done}<small>/ {total}</small>',
              f'<div class="pw-prog"><i style="--p:{pct:.1f}%"></i></div>')
       + cell("Laps left", "remaining", str(max(0, total - done)))
       + cell("Track", "track", "Wet" if wet else "Dry", cls="pw-v-wet" if wet else "")
       + cell("SC / VSC", "sc", sc, cls="pw-v-warn" if neutral else "")
       + cell("Rain", "rain", rain, cls="pw-v-wet" if rains else "")
       + cell("Tyre wear", "deg", f"{ss['w_deg']:g}×", cls="pw-v-warn" if ss["w_deg"] > 1 else "")
       + cell(f"{tcode} tyres", "tyre", f'{tyre_chip(ss["w_t_compound"])}{ss["w_t_age"]}<small>laps</small>')
       + '</div>')
    if ss.get("events_log"):
        md('<div class="pw-chips">' + "".join(f'<span class="pw-chip warn">Event: {esc(e["description"])}</span>'
                                                for e in ss["events_log"]) + '</div>')


# ====================================================================== 2. ask the agents
def ask_agents() -> None:
    st_ai = ai_status()
    md(step_head(2, "Ask the agents", h("agents")))
    c1, c2, c3, c4 = st.columns([3, 2.2, 1.6, 1.6], vertical_alignment="bottom")
    c1.selectbox("Question for the pit wall", list(QUESTIONS), format_func=lambda k: QUESTIONS[k],
                 key="w_question", help=h("question"))
    c2.radio("Tyre model", ["Formula", "ML model"], key="w_tyre_model", horizontal=True, help=h("tyre_model"))
    c3.toggle("Watch live", key="w_live", help=h("live_toggle"))
    c4.button("Analyze Strategy", type="primary", on_click=analyze_cb, width="stretch", help=h("analyze"))
    brain = (f'<span class="pw-chip ai">Agent brain: {esc(st_ai["text"])}</span>' if st_ai["on"] and st_ai["ready"]
             else f'<span class="pw-chip warn">Agent brain: {esc(st_ai["text"])}</span>' if st_ai["on"]
             else '<span class="pw-chip">Agent brain: rules (no AI model)</span>')
    md(f'<div class="pw-chips">{brain}{tip(h("ai_brain"))}</div>')
    if st.session_state.get("ml_error") and st.session_state.get("w_tyre_model") == "ML model":
        st.warning(f"ML tyre model unavailable: {st.session_state['ml_error']}. The formula is used.")


# ====================================================================== banners
def banners(result) -> None:
    from .state import brain_signature, current_fingerprint, result_brain
    ss = st.session_state
    errs = ss.get("input_errors") or []
    if errs:
        st.error("**Invalid inputs.** Nothing was analysed or saved. Fix these on the Race Setup page:\n"
                 + "\n".join(f"- **{i['field']}**: {i['message']}" for i in errs))
    if result is not None and (result["input_fingerprint"] != current_fingerprint()
                               or result_brain(result) != brain_signature()):
        st.warning("Inputs, question, tyre model or AI model changed since this analysis, so the results are "
                   "**out of date**. Press Analyze Strategy on Race Control to refresh.")


# ====================================================================== the call
def _stat(label: str, key: str, value: str, end: bool = False) -> str:
    return (f'<div class="pw-stat"><div class="k">{esc(label)}{tip(h(key), end=end)}</div>'
            f'<div class="v">{value}</div></div>')


def call_card(result) -> None:
    rec = result["recommendation"]
    ok = rec["status"] == "recommended"
    trig = result.get("trigger_event")
    meta = []
    if rec.get("decided_by"):
        cls = "ai" if rec.get("ai") else ""
        meta.append(f'<span class="pw-chip {cls}">{esc(rec["decided_by"])}</span>{tip(h("decided_by"))}')
    tm = (result["input_snapshot"].get("tyre_model") or {}).get("label")
    meta.append(f'<span class="pw-chip">Tyres: {esc(tm or "formula")}</span>')
    if trig:
        meta.append(f'<span class="pw-chip warn">{esc(trig["description"])}</span>')
    md(f'<div class="pw-call reveal{"" if ok else " review"}" data-run="{esc(result["analysis_id"])}">'
       f'<div class="q">Answer to “{esc(result["question"])}”{tip(h("call"))}</div>'
       f'<div class="act">{esc(rec["headline"])}</div>'
       f'<div class="meta">{"".join(meta)}<span>Record {esc(result["analysis_id"])}</span></div></div>')
    if not ok:
        st.error(rec["reason"])
        return
    n = rec["numbers"]
    m = n["margin_to_next_best_s"]
    margin = ('<small>only valid option</small>' if m is None else f'{m:.2f}<small> s</small>' if m >= 0
              else f'{m:.2f}<small> s vs fastest</small>')
    pos = (f'P{n["projected_position_in_group"]}<small> of {n["group_size"]}</small>'
           if n.get("projected_position_in_group") else "<small>not modelled</small>")
    md('<div class="pw-stats">'
       + _stat("Time to flag", "remaining_time", fmt_race_time(n["remaining_time_s"]))
       + _stat("Margin", "margin", margin)
       + _stat("Pit lane", "pit_loss", f'{n["pit_loss_total_s"]:.1f}<small> s</small>')
       + _stat("Finish", "position", pos, end=True) + '</div>')
    r = rec["risk"]
    md(f'<div class="pw-riskline"><span class="pw-risk {r["category"]}">{r["category"].title()} risk'
       f'</span>{tip(h("risk"))}<span>{esc("; ".join(r["reasons"]).capitalize())}</span></div>')


def call_details(result) -> None:
    rec = result["recommendation"]
    if rec["status"] != "recommended":
        return
    left, right = st.columns([3, 2], gap="medium")
    with left:
        if rec.get("ai") and rec["ai"].get("explanation"):
            md(f'<div class="pw-card ai"><h4>🧠 The AI model explains{tip(h("ai_brain"))}</h4>'
               + "".join(f"<p>{esc(x)}</p>" for x in rec["ai"]["explanation"])
               + (f'<p><b>Why this option:</b> {esc(rec["ai"]["why"])}</p>' if rec["ai"].get("why") else "")
               + "</div>")
        md(section("The numbers behind the call", h("why")))
        md('<ul class="pw-list">' + "".join(f"<li>{esc(x)}</li>" for x in rec["summary"]) + "</ul>")
        md(section("What the alternatives cost", h("tradeoffs")))
        items = rec["trade_offs"] or ["No other option types were tested for this question."]
        md('<ul class="pw-list alt">' + "".join(
            f'<li class="{"bad" if "rejected" in x else ""}">{esc(x)}</li>' for x in items) + "</ul>")
    with right:
        bars = "".join(f'<i style="--d:{i}"></i>' for i in range(7))
        md(f'<div class="pw-radio"><div class="hd"><span class="pw-wave">{bars}</span>'
           f'{esc(rec["radio"]["label"])}{tip(h("radio"), end=True)}</div>'
           f'<div class="txt">“{esc(rec["radio"]["text"])}”</div></div>')
        md(section("Before and after", h("compare")))
        comp = st.session_state.get("comparison")
        if comp:
            tag = '<span class="tag chg">Call changed</span>' if comp["changed"] else \
                '<span class="tag same">Call unchanged</span>'
            md(f'<div class="pw-ba">{tag}'
               f'<div class="row"><span>Cause</span><div>{esc(comp["event"].capitalize())}</div></div>'
               f'<div class="row"><span>Before</span><div>{esc(comp["previous_action"])} '
               f'<span style="color:var(--pw-muted)">({fmt_race_time(comp["previous_time_s"])})</span></div></div>'
               f'<div class="row"><span>Now</span><div>{esc(comp["current_action"])} '
               f'<span style="color:var(--pw-muted)">({fmt_race_time(comp["current_time_s"])})</span></div></div>'
               f'<div class="why">{esc(comp["reason"])}</div></div>')
        else:
            md('<p class="pw-side-note">Run another analysis or apply a race event in the Strategy Lab to compare '
               'decisions here.</p>')


# ====================================================================== what really happened
VERDICT_CLASS = {"match": "same", "timing": "chg", "tyres": "chg", "different": "chg", "none": "chg"}


def reality_panel(result) -> None:
    ss = st.session_state
    if not ss.get("replay_ctx"):
        return
    stat = reality_status()
    hindsight = bool((ss.get("replay_ctx") or {}).get("real_events"))
    sub = ("Hindsight mode: the agents were told the race's real Safety Cars, VSCs and rain; everything else came "
           "from laps up to the decision point." if hindsight else
           "The agents decided using only laps up to the decision point. Reveal the rest of the real race "
           "to see how their call compares with what the team did.")
    md(section("What really happened", h("reality"), sub=sub))
    if not stat["ok"]:
        if stat["reason"] == "mismatch":
            st.info("This analysis no longer matches the loaded race moment (the lap or driver was changed). "
                    "Load the moment again to compare with reality.")
        return
    if ss.get("reality_error"):
        st.error(ss["reality_error"])
    cmp = ss.get("reality")
    if not cmp or cmp.get("analysis_id") != result["analysis_id"]:
        st.button("Reveal what really happened", on_click=reveal_cb, type="primary",
                  help="Reads the laps after the decision point. They were never shown to the agents.")
        if stat.get("edited"):
            st.caption("You changed inputs or applied events after loading this race, so the comparison will mix "
                       "those changes with real events.")
        return
    t = cmp["times"]
    tag_cls = VERDICT_CLASS.get(cmp["verdict"]["code"], "chg")
    real_t = t["real_time_for_team_call_s"]
    err = cmp.get("model_error_s")
    md(f'<div class="pw-ba pw-reveal"><span class="tag {tag_cls}">{esc(cmp["verdict"]["text"].split(":")[0])}</span>'
       f'<div class="row"><span>Agents</span><div>{esc(cmp["model_call"])}</div></div>'
       f'<div class="row"><span>Team</span><div>{esc(cmp["team_call"])} '
       f'<span style="color:var(--pw-muted)">({esc(cmp["status"])}, P{cmp["finish_position"]})'
       f'</span></div></div><div class="why">{esc(cmp["verdict"]["text"])}</div></div>')
    md('<div class="pw-stats">'
       + _stat("Real time to flag", "real_time", fmt_race_time(real_t) if real_t else "<small>not comparable</small>")
       + _stat("Model's estimate", "model_est", fmt_race_time(t["model_estimate_for_team_call_s"])
               if t["model_estimate_for_team_call_s"] else "<small>–</small>")
       + _stat("Model error", "model_error", f'{err:+.1f}<small> s</small>' if err is not None else "<small>–</small>")
       + _stat("Agents vs team (model)", "model_gain",
               f'{cmp["model_gain_vs_team_s"]:+.1f}<small> s</small>' if cmp.get("model_gain_vs_team_s") is not None
               else "<small>–</small>", end=True) + '</div>')
    if cmp["events_after"]:
        md('<div class="pw-chips">' + "".join(f'<span class="pw-chip warn">{esc(k.capitalize())}: laps {esc(v)}</span>'
                                                for k, v in cmp["events_after"].items()) + "</div>")
    md('<ul class="pw-list">' + "".join(f"<li>{esc(x)}</li>" for x in cmp["explanations"]) + "</ul>")
    if cmp.get("lap_compare"):
        with st.expander("Where the model was right and wrong, lap by lap"):
            md(section("Real lap time minus model estimate", h("chart_reality"), first=True))
            st.altair_chart(charts.reality_chart(cmp), width="stretch")
    st.caption(cmp["note"] + " Model numbers are estimates from this simplified model.")


# ====================================================================== race events (Strategy Lab)
PRESETS = [("Safety Car next lap", "safety_car", {"ev_sc_until": 0, "ev_sc_dur": 4}, "🚨"),
           ("VSC next lap", "vsc", {"ev_vsc_until": 0, "ev_vsc_dur": 2}, "🟡"),
           ("Rain in 6 laps", "rain", {"ev_rain_until": 6, "ev_rain_dur": 0}, "🌧️"),
           ("Tyres wear 50% faster", "degradation", {"ev_deg_factor": 1.5}, "🔥")]


def _preset_cb(event_type: str, values: Dict) -> None:
    for k, v in values.items():
        st.session_state[k] = v
    apply_event_cb(event_type)


EVENT_DEFAULTS = {"ev_sc_until": 0, "ev_sc_dur": 4, "ev_vsc_until": 0, "ev_vsc_dur": 2, "ev_rain_until": 6,
                  "ev_rain_dur": 0, "ev_deg_factor": 1.5}


def events_panel() -> None:
    for k, v in EVENT_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    md(section("Inject a race event", h("events"),
               sub="One click adds the event at the current lap and re-runs all five agents, so you can see "
                   "how the call changes. Fine-tune the timing below.", first=True))
    cols = st.columns(4)
    for col, (label, kind, vals, icon) in zip(cols, PRESETS):
        col.button(f"{icon} {label}", key=f"preset_{kind}", on_click=_preset_cb, args=(kind, vals),
                   width="stretch", help="Apply this event now and re-run the agents.")
    with st.expander("Custom timing for each event"):
        a, v, b, c = st.columns(4)
        with a:
            md(f"<div class=\"pw-sub\">Safety Car{tip(h('ev_sc'))}</div>")
            st.number_input("Laps until it comes out (0 = next lap)", 0, 99, key="ev_sc_until")
            st.number_input("How many laps it stays out", 1, 10, key="ev_sc_dur")
            st.button("Apply Safety Car", on_click=apply_event_cb, args=("safety_car",), width="stretch")
        with v:
            md(f"<div class=\"pw-sub\">Virtual Safety Car{tip(h('ev_vsc'))}</div>")
            st.number_input("Laps until it starts (0 = next lap)", 0, 99, key="ev_vsc_until")
            st.number_input("How many laps it lasts", 1, 10, key="ev_vsc_dur")
            st.button("Apply VSC", on_click=apply_event_cb, args=("vsc",), width="stretch")
        with b:
            md(f"<div class=\"pw-sub\">Rain forecast{tip(h('ev_rain'))}</div>")
            st.number_input("Laps until the track is wet (1 = next lap)", 1, 99, key="ev_rain_until")
            st.number_input("Wet laps (0 = until the finish)", 0, 80, key="ev_rain_dur")
            st.button("Apply rain forecast", on_click=apply_event_cb, args=("rain",), width="stretch")
        with c:
            md(f"<div class=\"pw-sub\">Extra tyre wear{tip(h('ev_deg'))}</div>")
            st.number_input("Wear multiplier to apply", 1.1, 3.0, step=0.1, key="ev_deg_factor")
            st.button("Apply degradation increase", on_click=apply_event_cb, args=("degradation",),
                      width="stretch")
        st.caption("If you changed inputs since the last analysis, a baseline analysis runs first so the "
                   "before/after comparison is fair.")


# ====================================================================== options + charts
def strategy_charts(result) -> None:
    a, b = st.columns(2, gap="medium")
    with a:
        md(section("How much slower is each option?", h("chart_gap")))
        st.altair_chart(charts.delta_bar_chart(result), width="stretch")
        hidden = charts.bar_rows(result)[1]
        if hidden:
            st.caption(f"Hidden: {hidden} rejected option(s) far off the pace (see the table below).")
    with b:
        md(section("Where time is gained or lost", h("chart_trace")))
        st.altair_chart(charts.gap_trace_chart(result), width="stretch")
    with st.expander("Lap-by-lap breakdown of the recommended plan"):
        md(section("Every simulated lap", h("chart_laps"), first=True))
        st.altair_chart(charts.lap_time_chart(result), width="stretch")
        st.dataframe(tables.laps_frame(result), hide_index=True, width="stretch",
                     column_config=tables.laps_config())


def options_table(result) -> None:
    md(section("Every option the agents tested",
               "All candidate plans with their simulated times and the Critic's verdict. Hover any column "
               "heading for what it means.",
               sub=f"Valid options first, fastest at the top. Tie-break: "
                   f"{result['plan']['tie_break'] if result.get('plan') else 'n/a'}."))
    df = tables.candidates_frame(result)
    st.dataframe(tables.style_verdicts(df), hide_index=True, width="stretch",
                 column_config=tables.candidates_config(), height=min(600, 38 + 35 * len(df)))
    if result.get("rivals"):
        md(section("What the other cars are assumed to do", h("rivals")))
        st.dataframe(tables.rivals_frame(result), hide_index=True, width="stretch",
                     column_config=tables.rivals_config())


# ====================================================================== chat
SUGGESTIONS = ["Why this call?", "Why not pit now?", "What did the Critic reject?", "What is an undercut?"]


def scenario_suggestions() -> List[str]:
    ss = st.session_state
    done, total = int(ss["w_completed_laps"]), int(ss["w_total_laps"])
    left = total - done
    if left < 6:
        return ["What if a Safety Car comes out next lap?"]
    a, b = done + max(2, left // 4), min(total, done + max(4, left // 2))
    return ["What if a VSC comes out next lap?",
            f"Safety Car on laps {done + 2}–{done + 4} and rain from lap {a + 2} to {min(total, b + 4)}",
            f"After lap {done} of {total} on 15-lap-old mediums, VSC in 3 laps, tyre wear 1.5x"]


def chat_panel(result) -> None:
    ss = st.session_state
    sa = ai_status()
    who = (f'<span class="pw-chip ai">Answers by {esc(sa["text"].replace("AI: ", ""))}, using the agents\' '
           'exact results</span>' if sa["on"] and sa["ready"]
           else '<span class="pw-chip">Rule-based answers (no AI model is switched on)</span>')
    md(f'<div class="pw-chips">{who}{tip(h("chat"))}</div>')
    sugg = list(SUGGESTIONS)
    if ss.get("replay_ctx"):
        sugg.insert(1, "What really happened?")
    sugg += scenario_suggestions()
    for row in range(0, len(sugg), 4):
        cols = st.columns(4)
        for j, (col, q) in enumerate(zip(cols, sugg[row:row + 4])):
            col.button(q, key=f"sugg_{row + j}", on_click=chat_suggest_cb, args=(q,), width="stretch")
    box = st.container(height=560, border=True)
    with box:
        if not ss.get("chat"):
            st.caption("No messages yet. Pick a suggestion above or type a question below. Describe a scenario "
                       "(lap, tyres, Safety Car, VSC, rain…) and all five agents will run it.")
        for i, m in enumerate(ss.get("chat", [])):
            with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🏁"):
                st.markdown(m["text"])
                if m.get("charts"):
                    md(f'<div class="pw-sub">The race at a glance{tip(h("chart_timeline"))}</div>')
                    st.altair_chart(charts.scenario_timeline(m["charts"]), width="stretch")
                    with st.expander("How much slower was each plan?"):
                        st.altair_chart(charts.options_chart(m["charts"]["options"]), width="stretch")
                if m.get("scenario_raw"):
                    st.button("Load this scenario into the race inputs", key=f"load_sc_{i}",
                              on_click=load_chat_scenario_cb, args=(i,),
                              help="Copies the scenario into the inputs so you can run it live and save it.")
                if m["role"] == "assistant":
                    extra = [m.get("source", "")]
                    if m.get("tools"):
                        extra.append("Actions: " + "; ".join(m["tools"]))
                    st.caption(" · ".join(x for x in extra if x))
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    with c1:
        st.chat_input("Ask a question or describe a scenario, e.g. “VSC on laps 24–26 and rain from lap 35”",
                      key="chat_box", on_submit=chat_submit_cb)
    c2.button("Clear chat", on_click=chat_clear_cb, width="stretch", disabled=not ss.get("chat"))


# ====================================================================== race inputs editor
def _commit_periods():
    from .state import edited_periods
    ss = st.session_state
    ss["periods_df"] = edited_periods()
    ss["periods_editor_ver"] = ss.get("periods_editor_ver", 0) + 1


def _commit_competitors():
    from .state import edited_competitors
    ss = st.session_state
    ss["competitors_df"] = edited_competitors()
    ss["comp_editor_ver"] = ss.get("comp_editor_ver", 0) + 1


def inputs_editor() -> None:
    ss = st.session_state
    a, b, c = st.columns(3, gap="medium")
    with a:
        md(f"<div class=\"pw-sub\">Race{tip(h('decision_point'))}</div>")
        st.number_input("Total laps", 1, 150, step=1, key="w_total_laps", help=h("in_total"))
        st.number_input("Laps completed (decision point)", 0, 150, step=1, key="w_completed_laps", help=h("in_done"))
        st.number_input("Pit-stop time loss (s)", 0.0, 120.0, step=0.5, key="w_pit_loss", help=h("in_pit"))
        st.selectbox("Track right now", ["dry", "wet"], key="w_track_condition", help=h("in_track"),
                     format_func=str.title)
        st.number_input("Tyre wear multiplier", 0.1, 5.0, step=0.1, key="w_deg", help=h("in_deg"))
    with b:
        md(f"<div class=\"pw-sub\">Weather and Safety Car{tip(h('rain'))}</div>")
        st.number_input("Rain: first wet lap", 0, 150, step=1, key="w_rain_start", help=h("in_rain_start"))
        st.number_input("Rain: last wet lap", 0, 150, step=1, key="w_rain_end", help=h("in_rain_end"))
        st.number_input("Safety Car: first lap", 0, 150, step=1, key="w_sc_start", help=h("in_sc_start"))
        st.number_input("Safety Car: last lap", 0, 150, step=1, key="w_sc_end", help=h("in_sc_end"))
        st.text_input("Pit lane closed on laps", key="w_pit_closed", help=h("in_closed"), placeholder="e.g. 23, 24")
        md(f"<div class=\"pw-sub\">More Safety Car / VSC / rain periods{tip(h('in_periods'))}</div>")
        st.data_editor(ss["periods_df"], key=periods_key(), num_rows="dynamic", width="stretch", hide_index=True,
                       on_change=_commit_periods, column_config={
                           "Type": st.column_config.SelectboxColumn("Type", options=PERIOD_TYPES, required=True),
                           "First lap": st.column_config.NumberColumn("First lap", min_value=1, max_value=200, step=1),
                           "Last lap": st.column_config.NumberColumn("Last lap", min_value=1, max_value=200, step=1)})
    with c:
        t = ss["raw_base"]["target"]
        md(f"<div class=\"pw-sub\">Your car: {esc(t['code'])}, {esc(t.get('name', ''))}{tip(h('tyre'), end=True)}</div>")
        st.selectbox("Current tyres", list(COMPOUNDS), key="w_t_compound", help=h("in_compound"), format_func=str.title)
        st.number_input("Tyre age (laps)", 0, 150, step=1, key="w_t_age", help=h("in_age"))
        st.number_input("Base lap time (s)", 0.0, 400.0, step=0.05, format="%.3f", key="w_t_pace", help=h("in_pace"))
        st.number_input("Gap to leader (s)", 0.0, 1000.0, step=0.1, format="%.3f", key="w_t_gap", help=h("in_gap"))
        st.multiselect("Tyre types already used", list(COMPOUNDS), key="w_t_used", help=h("in_used"),
                       format_func=str.title)
    md(f"<div class=\"pw-sub\">New tyre sets left{tip(h('in_sets'))}</div>")
    cols = st.columns(len(COMPOUNDS))
    for col, comp in zip(cols, COMPOUNDS):
        col.number_input(comp.title(), 0, 10, step=1, key=f"w_t_sets_{comp}", help=f"Unused {comp.title()} sets.")
    md(section("Other cars in the simulation", h("in_comp"),
               sub="Gaps use the same leader reference as your car. Tyre sets come from the scenario."))
    C = st.column_config
    st.data_editor(ss["competitors_df"], key=editor_key(), num_rows="fixed", width="stretch", hide_index=True,
                   on_change=_commit_competitors, column_order=COMP_COLS, column_config={
                       "code": C.TextColumn("Driver", help="Three-letter driver code."),
                       "gap_to_leader_s": C.NumberColumn("Gap to leader (s)", format="%.3f", help=h("in_gap")),
                       "baseline_pace_s": C.NumberColumn("Base lap time (s)", format="%.3f", help=h("in_pace")),
                       "compound": C.SelectboxColumn("Tyres", options=list(COMPOUNDS), help=h("in_compound")),
                       "tyre_age_laps": C.NumberColumn("Tyre age (laps)", step=1, help=h("in_age")),
                       "compounds_used": C.TextColumn("Tyre types used", help="Comma-separated, e.g. MEDIUM, HARD.")})
    issues = validate_race_input(build_raw())
    if issues:
        st.error("**These inputs are invalid:**\n" + "\n".join(f"- **{i['field']}**: {i['message']}" for i in issues))
    else:
        st.success("All inputs are valid.")


# ====================================================================== records
def history_table() -> None:
    s = store()
    raw = s.summary_frame()
    md(section("Decision history", h("history"),
               sub=f"{len(raw)} saved record{'s' if len(raw) != 1 else ''}, newest first."))
    if raw.empty:
        md('<p class="pw-side-note">No decisions yet. Every completed analysis is saved here automatically.</p>')
    else:
        st.dataframe(tables.history_frame(raw), hide_index=True, width="stretch",
                     column_config=tables.history_config())


def downloads(result) -> None:
    s = store()
    raw = s.summary_frame()
    md(section("Downloads", "JSON keeps every detail of each record; CSV is a one-row-per-decision summary."))
    c = st.columns(4)
    c[0].download_button("All decisions (JSON)", s.to_json_bytes(), "pitwall_decisions.json", "application/json",
                         disabled=raw.empty, width="stretch", help="Full records, including inputs and checks.")
    c[1].download_button("All decisions (CSV)", s.to_csv_bytes(), "pitwall_decisions.csv", "text/csv",
                         disabled=raw.empty, width="stretch", help="One summary row per decision.")
    if result is not None:
        c[2].download_button("This analysis (JSON)", json.dumps(result, indent=2, default=str).encode(),
                             f"{result['analysis_id']}.json", "application/json", width="stretch",
                             help="Everything from the current analysis: plan, simulations, checks and log.")
        if result.get("candidates"):
            c[3].download_button("Options table (CSV)", tables.candidates_frame(result).to_csv(index=False).encode(),
                                 f"{result['analysis_id']}_options.csv", "text/csv", width="stretch",
                                 help="The strategy comparison table as shown.")

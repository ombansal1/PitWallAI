"""Streamlit session-state handling. All mutations happen inside button/selection callbacks,
which run exactly once per user action, so reruns never re-apply events or duplicate records."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from datetime import datetime, timezone

from pitwall.agents.assistant import ChatContext, PitWallAssistant
from pitwall.comparison import compare_results
from pitwall.config import COMPOUNDS
from pitwall.data_sources import fastf1_source as ff1src
from pitwall.data_sources.fastf1_source import DataSourceError, build_race_state
from pitwall.data_sources.replays import load_replay
from pitwall.data_sources.synthetic import load_scenario
from pitwall.events import EventError, apply_event
from pitwall.llm import DEFAULT_PROVIDER, PROVIDERS, MockReasoner, find_api_key, make_reasoner
from pitwall.ml.tyre_model import MLUnavailable, tyre_model_payload
from pitwall.orchestrator import input_fingerprint, run_analysis
from pitwall.reality import (RealityError, actual_outcome, apply_real_events, compare_with_reality,
                             real_events_after)
from pitwall.records import DecisionStore, build_decision_record, build_reality_record
from pitwall.validation import validate_race_input

COMP_COLS = ["code", "gap_to_leader_s", "baseline_pace_s", "compound", "tyre_age_laps", "compounds_used"]
PERIOD_COLS = ["Type", "First lap", "Last lap"]
PERIOD_TYPES = ["Safety Car", "VSC", "Rain"]


def periods_frame(raw: Dict[str, Any]) -> pd.DataFrame:
    rows = [{"Type": "Safety Car" if p["kind"] == "SC" else "VSC", "First lap": int(p["start"]),
             "Last lap": int(p["end"])} for p in raw.get("neutral_periods") or []]
    rows += [{"Type": "Rain", "First lap": int(p["start"]), "Last lap": int(p["end"])}
             for p in raw.get("wet_periods") or []]
    return pd.DataFrame(rows, columns=PERIOD_COLS)


def store() -> DecisionStore:
    return DecisionStore()


def flash(level: str, msg: str) -> None:
    st.session_state.setdefault("flash", []).append((level, msg))


# ------------------------------------------------------------------ loading race states
def load_raw_into_widgets(raw: Dict[str, Any]) -> None:
    ss = st.session_state
    ss["raw_base"] = copy.deepcopy(raw)
    ss["replay_ctx"] = copy.deepcopy(raw.get("replay"))  # None for synthetic scenarios
    ss["reality"] = None
    ss["reality_error"] = None
    ss["events_log"] = list(raw.get("events", []))
    ss["w_total_laps"] = int(raw["total_laps"])
    ss["w_completed_laps"] = int(raw["completed_laps"])
    ss["w_pit_loss"] = float(raw["pit_loss_s"])
    ss["w_track_condition"] = raw["track_condition"]
    ss["w_rain_start"] = int(raw.get("rain_start_lap") or 0)
    ss["w_rain_end"] = int(raw.get("rain_end_lap") or 0)
    ss["w_sc_start"] = int(raw.get("sc_start_lap") or 0)
    ss["w_sc_end"] = int(raw.get("sc_end_lap") or 0)
    ss["w_deg"] = float(raw["deg_multiplier"])
    ss["w_pit_closed"] = ", ".join(str(x) for x in raw.get("pit_closed_laps", []))
    t = raw["target"]
    ss["w_t_compound"] = t["compound"]
    ss["w_t_age"] = int(t["tyre_age_laps"])
    ss["w_t_pace"] = float(t["baseline_pace_s"])
    ss["w_t_gap"] = float(t["gap_to_leader_s"])
    ss["w_t_used"] = [c for c in t.get("compounds_used", []) if c in COMPOUNDS]
    for c in COMPOUNDS:
        ss[f"w_t_sets_{c}"] = int(t["available_sets"].get(c, 0))
    ss["competitors_df"] = pd.DataFrame([{
        "code": c["code"], "gap_to_leader_s": float(c["gap_to_leader_s"]),
        "baseline_pace_s": float(c["baseline_pace_s"]), "compound": c["compound"],
        "tyre_age_laps": int(c["tyre_age_laps"]), "compounds_used": ", ".join(c.get("compounds_used", []))}
        for c in raw["competitors"]], columns=COMP_COLS)
    ss["comp_editor_ver"] = ss.get("comp_editor_ver", 0) + 1
    ss["periods_df"] = periods_frame(raw)
    ss["periods_editor_ver"] = ss.get("periods_editor_ver", 0) + 1


def init_state() -> None:
    ss = st.session_state
    if ss.get("initialized"):
        return
    ss["initialized"] = True
    ss["w_source"] = "Example scenarios"
    ss["w_scenario"] = "dry_normal"
    ss["w_question"] = "full_strategy"
    ss["w_tyre_model"] = "Formula"
    init_ai()
    ss["last_result"] = None
    ss["previous_result"] = None
    ss["comparison"] = None
    ss["input_errors"] = []
    ss["chat"] = []
    ss["w_live"] = True
    ss["pending_run"] = None
    load_raw_into_widgets(load_scenario("dry_normal"))


def on_scenario_change() -> None:
    try:
        load_raw_into_widgets(load_scenario(st.session_state["w_scenario"]))
        st.session_state["input_errors"] = []
        flash("info", "Scenario loaded. Previous results are kept but marked out of date until you re-analyse.")
    except Exception as e:  # corrupt/missing bundled file
        flash("error", f"Could not load scenario: {e}")


# ------------------------------------------------------------------ building the input
def _parse_laps(text: str):
    text = (text or "").strip()
    if not text:
        return []
    try:
        return [int(x) for x in text.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        return text  # left invalid on purpose: validation reports it, nothing is guessed


def editor_key() -> str:
    return f"comp_editor_{st.session_state.get('comp_editor_ver', 0)}"


def edited_competitors() -> pd.DataFrame:
    """Base competitor table with the data editor's pending edits applied (works in callbacks)."""
    df = st.session_state["competitors_df"].copy()
    delta = st.session_state.get(editor_key()) or {}
    for idx, changes in (delta.get("edited_rows") or {}).items():
        for col, val in changes.items():
            if col in df.columns:
                df[col] = df[col].astype(object)
                df.at[int(idx), col] = val
    return df


def periods_key() -> str:
    return f"periods_editor_{st.session_state.get('periods_editor_ver', 0)}"


def edited_periods() -> pd.DataFrame:
    """Extra-periods table with pending editor changes (edits, added and deleted rows) applied."""
    df = st.session_state.get("periods_df", pd.DataFrame(columns=PERIOD_COLS)).copy().astype(object)
    delta = st.session_state.get(periods_key()) or {}
    for idx, changes in (delta.get("edited_rows") or {}).items():
        for col, val in changes.items():
            if col in df.columns:
                df.at[int(idx), col] = val
    keep = [i for i in range(len(df)) if i not in set(delta.get("deleted_rows") or [])]
    df = df.iloc[keep]
    added = [{c: r.get(c) for c in PERIOD_COLS} for r in (delta.get("added_rows") or [])]
    if added:
        df = pd.concat([df, pd.DataFrame(added, columns=PERIOD_COLS)], ignore_index=True)
    return df.reset_index(drop=True)


def _period_lists(df: pd.DataFrame):
    neutral, wet = [], []
    for _, r in df.iterrows():
        t, a, b = r.get("Type"), r.get("First lap"), r.get("Last lap")
        clean = lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else (  # noqa: E731
            int(v) if isinstance(v, (int, float)) and float(v) == int(v) else v)
        a, b = clean(a), clean(b)
        if t is None and a is None and b is None:
            continue  # empty new row
        if t == "Rain":
            wet.append({"start": a, "end": b})
        else:
            neutral.append({"kind": "VSC" if t == "VSC" else ("SC" if t == "Safety Car" else t), "start": a, "end": b})
    return neutral, wet


def build_raw() -> Dict[str, Any]:
    ss = st.session_state
    base = ss["raw_base"]
    raw = copy.deepcopy(base)
    nz = lambda v: int(v) if v else None  # noqa: E731  (0 in the UI = not set)
    raw.update({
        "total_laps": ss["w_total_laps"], "completed_laps": ss["w_completed_laps"],
        "pit_loss_s": ss["w_pit_loss"], "track_condition": ss["w_track_condition"],
        "rain_start_lap": nz(ss["w_rain_start"]), "rain_end_lap": nz(ss["w_rain_end"]),
        "sc_start_lap": nz(ss["w_sc_start"]), "sc_end_lap": nz(ss["w_sc_end"]),
        "deg_multiplier": ss["w_deg"], "pit_closed_laps": _parse_laps(ss["w_pit_closed"]),
        "events": list(ss.get("events_log", [])),
    })
    neutral, wet = _period_lists(edited_periods())
    raw["neutral_periods"], raw["wet_periods"] = neutral, wet
    raw.pop("tyre_model", None)
    if ss.get("w_tyre_model") == "ML model":
        circuit = base.get("track_name") if base.get("data_source") in ("replay", "fastf1") else None
        try:
            raw["tyre_model"] = _ml_payload(circuit)
        except MLUnavailable as e:
            ss["ml_error"] = str(e)
    t = raw["target"]
    t.update({"compound": ss["w_t_compound"], "tyre_age_laps": ss["w_t_age"],
              "baseline_pace_s": ss["w_t_pace"], "gap_to_leader_s": ss["w_t_gap"],
              "compounds_used": list(ss["w_t_used"]),
              "available_sets": {c: ss[f"w_t_sets_{c}"] for c in COMPOUNDS}})
    df = edited_competitors()
    comps = []
    for i, row in df.reset_index(drop=True).iterrows():
        b = base["competitors"][i] if i < len(base["competitors"]) else {}
        used = [x.strip().upper() for x in str(row.get("compounds_used") or "").split(",") if x.strip()]

        def num(v):
            return None if v is None or (isinstance(v, float) and pd.isna(v)) else (
                v.item() if hasattr(v, "item") else v)
        comps.append({**b, "code": str(row["code"]).strip().upper() if row["code"] is not None else "",
                      "gap_to_leader_s": num(row["gap_to_leader_s"]),
                      "baseline_pace_s": num(row["baseline_pace_s"]),
                      "compound": row["compound"], "tyre_age_laps": num(row["tyre_age_laps"]),
                      "compounds_used": used})
    raw["competitors"] = comps
    return raw


# ------------------------------------------------------------------ AI model ("agent brain")
AI_CHOICES = ["rules", "groq", "gemini", "ollama"]
AI_LABEL = {"rules": "Rules only (no AI model)", **{k: v.label for k, v in PROVIDERS.items()}}


def _dotenv_path() -> Path:
    return Path(os.environ.get("PITWALL_DOTENV") or Path(__file__).resolve().parent.parent / ".env")


def _secrets() -> Dict[str, Any]:
    try:
        return dict(st.secrets)
    except Exception:          # no secrets.toml: that's fine
        return {}


def stored_key(provider: str):
    """(key, where) from secrets.toml / environment / .env; never from the page."""
    if provider not in PROVIDERS:
        return None, "not needed"
    return find_api_key(provider, _secrets(), _dotenv_path())


def init_ai() -> None:
    """Start with the AI model when a key is configured (Groq first), otherwise with the rules."""
    ss = st.session_state
    choice = os.environ.get("PITWALL_AI_PROVIDER", "").strip().lower()
    if choice not in AI_CHOICES:
        choice = next((p for p in ("groq", "gemini") if stored_key(p)[0]), "rules")
    ss["ai_provider"] = choice
    ss["ai_model"] = (PROVIDERS[choice] if choice in PROVIDERS else PROVIDERS[DEFAULT_PROVIDER]).default_model
    ss["ai_key_input"] = ""
    ss["ai_ollama_url"] = "http://localhost:11434"
    ss["ai_test"] = None


def on_ai_provider_change() -> None:
    ss = st.session_state
    p = ss.get("ai_provider", "rules")
    if p in PROVIDERS:
        ss["ai_model"] = PROVIDERS[p].default_model
    ss["ai_key_input"] = ""
    ss["ai_test"] = None


def ai_key():
    """(key, where): typed on the AI page (this browser session only) or stored configuration."""
    ss = st.session_state
    p = ss.get("ai_provider", "rules")
    typed = (ss.get("ai_key_input") or "").strip()
    if typed:
        return typed, "typed on the Model Lab page (kept for this session only)"
    return stored_key(p)


def reasoner():
    ss = st.session_state
    p = ss.get("ai_provider", "rules")
    if p not in PROVIDERS:
        return MockReasoner()
    base = os.environ.get("PITWALL_AI_BASE_URL") or None     # advanced: any OpenAI-compatible endpoint
    if p == "ollama" and not base:
        base = (ss.get("ai_ollama_url") or "http://localhost:11434").rstrip("/") + "/v1"
    return make_reasoner(p, model=ss.get("ai_model") or None, api_key=ai_key()[0], base_url=base)


def ai_status() -> Dict[str, Any]:
    """Short status for the header and the Model Lab page."""
    ss = st.session_state
    p = ss.get("ai_provider", "rules")
    if p not in PROVIDERS:
        return {"on": False, "ready": True, "text": "Rules (no AI model)", "detail": "The agents use their rules."}
    key, where = ai_key()
    model = ss.get("ai_model") or PROVIDERS[p].default_model
    if PROVIDERS[p].key_env and not key:
        return {"on": True, "ready": False, "text": f"AI: {model} · key missing",
                "detail": f"No API key for {PROVIDERS[p].label}. Add one on the Model Lab page."}
    test = ss.get("ai_test") or {}
    return {"on": True, "ready": True, "text": f"AI: {model}", "where": where,
            "tested": test.get("ok") if test.get("model") == model and test.get("provider") == p else None,
            "detail": f"{PROVIDERS[p].label}, key {where}." if PROVIDERS[p].key_env else PROVIDERS[p].label}


def ai_test_cb() -> None:
    ss = st.session_state
    r = reasoner()
    out = r.test_connection() if hasattr(r, "test_connection") else {"ok": True, "message": "Rules need no connection."}
    ss["ai_test"] = {**out, "provider": ss.get("ai_provider"), "model": ss.get("ai_model")}


def brain_signature() -> str:
    ss = st.session_state
    p = ss.get("ai_provider", "rules")
    return "rules" if p not in PROVIDERS else f"{p}:{ss.get('ai_model')}"


def result_brain(result: Dict[str, Any]) -> str:
    ai = result.get("ai") or {}
    return "rules" if ai.get("provider", "rules") in ("rules", None) else f"{ai['provider']}:{ai['model']}"


def current_fingerprint() -> str:
    return input_fingerprint(build_raw(), st.session_state["w_question"])


# ------------------------------------------------------------------ running analyses
def _handle_result(result: Dict[str, Any]) -> bool:
    ss = st.session_state
    if result["status"] == "invalid_input":
        ss["input_errors"] = result["input_issues"]
        flash("error", "Inputs are invalid — nothing was analysed or recorded. See the errors below.")
        return False
    ss["input_errors"] = []
    store().append(build_decision_record(result))  # idempotent by analysis_id
    prev = ss.get("last_result")
    ss["previous_result"] = prev
    ss["last_result"] = result
    ss["comparison"] = compare_results(prev, result) if prev else None
    ss["reality"], ss["reality_error"] = None, None   # a reality check belongs to one analysis
    return True


WIDGET_PREFIXES = ("w_", "ev_", "rr_", "ff1_", "ai_", "ml_try_")


@st.cache_data(show_spinner=False, max_entries=64)
def _ml_payload(circuit):
    return tyre_model_payload(circuit)


def keep_widget_state() -> None:
    """Multipage app: a widget's value is forgotten when its page isn't shown. Re-assigning the value at the
    start of every run keeps race inputs and settings when you switch pages."""
    ss = st.session_state
    for k in list(ss.keys()):
        if isinstance(k, str) and k.startswith(WIDGET_PREFIXES) and not k.endswith(("_btn", "_test")):
            try:
                ss[k] = ss[k]
            except Exception:   # e.g. a button state; nothing to keep
                pass


def _protect_inputs() -> None:
    """The live run restarts the page before the input widgets are drawn; some Streamlit versions
    then forget those widgets' values. Commit table edits and keep a copy that restore_inputs() puts back."""
    ss = st.session_state
    ss["competitors_df"] = edited_competitors()
    ss["comp_editor_ver"] = ss.get("comp_editor_ver", 0) + 1
    ss["periods_df"] = edited_periods()
    ss["periods_editor_ver"] = ss.get("periods_editor_ver", 0) + 1
    ss["_kept_inputs"] = {k: ss[k] for k in list(ss.keys())
                          if isinstance(k, str) and k.startswith(WIDGET_PREFIXES) and not k.endswith("_btn")}


def restore_inputs() -> None:
    """Put back any input value dropped during a live-run restart (no-op when nothing is missing)."""
    ss = st.session_state
    for k, v in (ss.pop("_kept_inputs", None) or {}).items():
        if k not in ss:
            ss[k] = v


def _queue_or_run(raw: Dict[str, Any], trigger=None, message: str = "", where: str = "agents") -> None:
    """With 'watch live' on, the run happens in the page (see app.py) so steps appear as they happen."""
    ss = st.session_state
    if ss.get("w_live", True):
        ss["pending_run"] = {"raw": raw, "question": ss["w_question"], "trigger": trigger,
                             "reasoner": reasoner(), "message": message, "where": where}
        _protect_inputs()
        return
    result = run_analysis(raw, ss["w_question"], trigger_event=trigger, reasoner=reasoner())
    finish_run(result, message)


def finish_run(result: Dict[str, Any], message: str = "") -> None:
    if _handle_result(result):
        flash("success", message.format(id=result["analysis_id"]) if message else
              f"Analysis {result['analysis_id']} completed and recorded.")


def analyze_cb() -> None:
    _queue_or_run(build_raw())


def apply_event_cb(event_type: str) -> None:
    ss = st.session_state
    raw = build_raw()
    issues = validate_race_input(raw)
    if issues:
        ss["input_errors"] = issues
        flash("error", "Fix the input errors before applying an event.")
        return
    if event_type == "safety_car":
        ev = {"type": "safety_car", "laps_until": int(ss["ev_sc_until"]), "duration_laps": int(ss["ev_sc_dur"])}
    elif event_type == "vsc":
        ev = {"type": "vsc", "laps_until": int(ss["ev_vsc_until"]), "duration_laps": int(ss["ev_vsc_dur"])}
    elif event_type == "rain":
        ev = {"type": "rain", "laps_until": int(ss["ev_rain_until"]), "duration_laps": int(ss["ev_rain_dur"])}
    else:
        ev = {"type": "degradation", "factor": float(ss["ev_deg_factor"])}
    try:
        new_raw, record = apply_event(raw, ev)
    except EventError as e:
        flash("error", f"Event not applied: {e}")
        return
    q = ss["w_question"]
    last = ss.get("last_result")
    if last is None or last.get("input_fingerprint") != input_fingerprint(raw, q):
        # Make sure "previous" describes the same inputs, just before the event.
        base = run_analysis(raw, q, trigger_event={"type": "baseline",
                                                   "description": "automatic pre-event baseline"},
                            reasoner=reasoner())
        _handle_result(base)
    # Write the event's effect back to the editable controls (same decision point).
    ss["w_sc_start"] = int(new_raw.get("sc_start_lap") or 0)
    ss["w_sc_end"] = int(new_raw.get("sc_end_lap") or 0)
    ss["w_rain_start"] = int(new_raw.get("rain_start_lap") or 0)
    ss["w_rain_end"] = int(new_raw.get("rain_end_lap") or 0)
    ss["w_deg"] = float(new_raw["deg_multiplier"])
    ss["periods_df"] = periods_frame(new_raw)
    ss["periods_editor_ver"] = ss.get("periods_editor_ver", 0) + 1
    ss["events_log"] = new_raw["events"]
    ss["raw_base"]["provenance"] = new_raw["provenance"]
    _queue_or_run(build_raw(), trigger=record, where="lab",
                  message=f"Event applied: {record['description']}. Workflow re-run ({{id}}).")


def clear_history_cb() -> None:
    store().clear()
    flash("info", "Decision history cleared.")


def use_fallback_cb() -> None:
    raw = load_scenario("dry_normal")
    raw["data_source"] = "synthetic_fallback"
    raw["data_source_label"] = "Synthetic fallback (FastF1 data unavailable)"
    load_raw_into_widgets(raw)
    st.session_state["w_source"] = "Example scenarios"
    flash("warning", "Using the synthetic fallback scenario.")


# ------------------------------------------------------------------ real races
@st.cache_resource(show_spinner=False, max_entries=6)
def replay_session(race_id: str):
    return load_replay(race_id)


@st.cache_resource(show_spinner=False, max_entries=3)
def fastf1_session(year: int, rnd: int):
    return ff1src.load_session(year, rnd)


def session_for(ctx: Dict[str, Any]):
    if ctx.get("source") == "replay":
        return replay_session(ctx["race_id"])
    return fastf1_session(int(ctx["year"]), int(ctx["round"]))


def load_real_race_cb() -> None:
    ss = st.session_state
    rid, drv, lap = ss.get("rr_race"), ss.get("rr_driver"), ss.get("rr_lap")
    try:
        sess = replay_session(rid)
        raw = build_race_state(sess, drv, int(lap), source="replay")
    except DataSourceError as e:
        flash("error", f"Couldn't use that moment: {e}.")
        return
    raw["replay"]["race_id"] = rid
    hindsight = bool(ss.get("rr_real_events"))
    if hindsight:
        raw = apply_real_events(raw, real_events_after(sess, int(lap)))
    load_raw_into_widgets(raw)
    ss["input_errors"] = []
    if hindsight:
        flash("warning", f"Loaded {raw['scenario_name']}. Hindsight mode: the agents are given the real Safety "
                         "Cars, VSCs and rain from the rest of the race. Press Analyze Strategy.")
    else:
        flash("success", f"Loaded {raw['scenario_name']}. Press Analyze Strategy. The agents only see laps up to "
                         f"lap {lap}.")


def reality_status() -> Dict[str, Any]:
    """Can the current analysis be checked against the real race? Returns {ok, reason}."""
    ss = st.session_state
    ctx, res = ss.get("replay_ctx"), ss.get("last_result")
    if not ctx:
        return {"ok": False, "reason": "synthetic"}
    if res is None or res.get("status") == "invalid_input":
        return {"ok": False, "reason": "no_result"}
    snap = res["input_snapshot"]
    if (snap.get("replay") or {}).get("decision_lap") != ctx.get("decision_lap") \
            or int(snap.get("completed_laps", -1)) != int(ctx.get("decision_lap", -2)) \
            or (snap.get("target") or {}).get("code") != ctx.get("driver"):
        return {"ok": False, "reason": "mismatch"}
    return {"ok": True, "reason": "", "edited": bool(snap.get("events")) or res["input_fingerprint"] != current_fingerprint()}


def reveal_reality(quiet: bool = False):
    """Compute the hindsight comparison for the current analysis (reads laps AFTER the decision)."""
    ss = st.session_state
    stat = reality_status()
    if not stat["ok"]:
        return None
    res, ctx = ss["last_result"], ss["replay_ctx"]
    if ss.get("reality") and ss["reality"].get("analysis_id") == res["analysis_id"]:
        return ss["reality"]
    try:
        sess = session_for(ctx)
        group = [c["code"] for c in res["input_snapshot"]["competitors"]]
        out = actual_outcome(sess, ctx["driver"], int(ctx["decision_lap"]), group)
        cmp = compare_with_reality(res, out)
    except (DataSourceError, RealityError, KeyError, ValueError) as e:
        ss["reality_error"] = f"Couldn't load what happened next: {e}"
        return None
    ss["reality"], ss["reality_error"] = cmp, None
    store().append(build_reality_record(cmp, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    if not quiet:
        flash("info", f"Real race revealed: {cmp['verdict']['text']}")
    return cmp


def reveal_cb() -> None:
    reveal_reality()


# ------------------------------------------------------------------ chat
def _ask(question: str) -> None:
    ss = st.session_state
    question = (question or "").strip()
    if not question:
        return
    res = ss.get("last_result")
    stale = bool(res) and res.get("input_fingerprint") != current_fingerprint()
    rstat = reality_status()
    ctx = ChatContext(raw=build_raw(), question_key=ss["w_question"], result=res, stale=stale,
                      reality=ss.get("reality"), reveal=(lambda: reveal_reality(quiet=True)) if rstat["ok"] else None,
                      is_real_race=bool(ss.get("replay_ctx")))
    answer = PitWallAssistant(reasoner=reasoner()).answer(question, ctx)
    ss.setdefault("chat", [])
    ss["chat"].append({"role": "user", "text": question})
    msg = answer.to_dict()
    msg["about"] = res["analysis_id"] if res else None
    ss["chat"].append(msg)


def chat_submit_cb() -> None:
    _ask(st.session_state.get("chat_box", ""))


def chat_suggest_cb(question: str) -> None:
    _ask(question)


def chat_clear_cb() -> None:
    st.session_state["chat"] = []


def load_chat_scenario_cb(index: int) -> None:
    ss = st.session_state
    try:
        raw = ss["chat"][index]["scenario_raw"]
    except (IndexError, KeyError):
        return
    load_raw_into_widgets(raw)
    ss["input_errors"] = []
    flash("success", "Scenario loaded into the race inputs. Press Analyze Strategy to run it live and save the decision.")

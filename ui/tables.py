"""Turns raw workflow output into readable tables (plain headings, plain values, header tooltips)."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from .help_text import HELP
from .styles import fmt_race_time

FAMILY = {"pit_now": "Box this lap", "delay": "Box a few laps later", "stay_out": "No more stops",
          "event_aligned": "Timed to an event", "two_stop": "Two stops", "earliest_open": "Next open pit lap",
          "ai_plan": "Proposed by the AI"}
VERDICT = {"SELECTED": "★ Recommended", "ACCEPTED": "✓ Valid", "REJECTED": "✕ Rejected",
           "INTEGRITY_FAILED_RERUN": "↻ Re-run requested", "NOT_SIMULATED": "– Not simulated"}
AGENT_ICON = {"Planner": "🧭 Planner", "Race Research": "🔎 Race Research",
              "Strategy Executor": "⚙️ Executor", "Strategy Critic": "🛡️ Critic",
              "Pit Wall Operations": "📋 Operations", "Orchestrator": "🔀 Orchestrator"}
KIND = {"agent": "Decision", "tool": "Tool run", "validation": "Check", "revision": "Revision", "llm": "LLM call",
        "ai": "AI model"}
ACTION = {"start": "Started workflow", "validate_inputs": "Checked the inputs", "create_plan": "Created the plan",
          "generate_candidates": "Generated options", "track_status": "Analysed track status",
          "tyre_analysis": "Analysed tyres", "gap_analysis": "Analysed gaps to other cars",
          "weather_analysis": "Analysed weather", "research_complete": "Finished research",
          "project_rivals": "Projected rival strategies", "simulate_strategy": "Simulated an option",
          "review": "Reviewed all options", "route_to_planner": "Sent back to the Planner",
          "route_to_executor": "Sent back to the Executor", "revise_plan": "Revised the plan",
          "retry_limit_reached": "Hit the revision limit", "no_progress": "Stopped revising",
          "select_strategy": "Picked the fastest valid option", "finalize": "Wrote the recommendation",
          "ollama_rewrite": "Rewrote summary with Ollama", "ollama_unavailable": "Ollama unavailable",
          "ollama_failed": "Ollama call failed", "rule_plan_kept": "Kept the rule plan",
          "ai_unavailable": "AI model unavailable", "ai_failed": "AI answer unusable, rules used",
          "ai_choose": "AI model made the call", "ai_choice_rejected": "Refused the AI's choice", "ai_radio_rejected": "Replaced the AI's radio message",
          "tyre_model": "Loaded the ML tyre model"}
SOURCE = {"synthetic": "Synthetic scenario", "synthetic_fallback": "Synthetic fallback",
          "fastf1": "FastF1 replay", "replay": "Real race replay"}


def origin_label(origin: str) -> str:
    if origin == "plan_v1":
        return "First plan"
    if origin.startswith("plan_v"):
        n = int(origin.split("v")[-1])
        return f"Revision {n - 1} (plan v{n})"
    return origin


def action_label(action: str) -> str:
    for prefix, text in (("rejected: ", "Rejected "), ("integrity_failed_rerun: ", "Flagged bad numbers in "),
                         ("accepted after re-run: ", "Accepted after re-run: ")):
        if action.startswith(prefix):
            return text + action[len(prefix):]
    return ACTION.get(action, action.replace("_", " ").capitalize())


def _help_cfg(**cols: Dict[str, Any]) -> Dict[str, Any]:
    return cols


def candidates_frame(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in result["candidates"]:
        tyre = (f"{r['final_compound'].title()}, {r['final_stint_age']} laps"
                if r["final_compound"] else "–")
        rows.append({
            "Verdict": VERDICT.get(r["status"], r["status"].title()),
            "Strategy": r["label"],
            "Time to flag": fmt_race_time(r["remaining_time_s"]),
            "Gap to recommended": r["delta_to_best_s"],
            "Why rejected / notes": r["issues"] or ("Passed every check" if r["status"] in ("SELECTED", "ACCEPTED") else ""),
            "Type": FAMILY.get(r["family"], r["family"]),
            "Proposed by": origin_label(r["origin"]),
            "Stops": r["stops"],
            "Tyres at the flag": tyre,
            "Finish (sim. group)": f"P{r['position_in_group']}" if r["position_in_group"] else "–",
            "Lapping time": r["lap_time_total_s"],
            "Pit-lane time": r["pit_loss_total_s"],
            "Traffic laps": r["traffic_laps"],
        })
    return pd.DataFrame(rows)


def candidates_config() -> Dict[str, Any]:
    C = st.column_config
    return {
        "Verdict": C.TextColumn("Verdict", help=HELP["t_verdict"], width=130),
        "Strategy": C.TextColumn("Strategy", help=HELP["t_strategy"], width="medium"),
        "Type": C.TextColumn("Type", help=HELP["t_type"]),
        "Proposed by": C.TextColumn("Proposed by", help=HELP["t_origin"]),
        "Stops": C.NumberColumn("Stops", help=HELP["t_stops"], format="%d", width="small"),
        "Time to flag": C.TextColumn("Time to flag", help=HELP["t_time"]),
        "Gap to recommended": C.NumberColumn("Gap to recommended", help=HELP["t_gap"], format="%+.2f s"),
        "Lapping time": C.NumberColumn("Lapping time", help=HELP["t_driving"], format="%.1f s"),
        "Pit-lane time": C.NumberColumn("Pit-lane time", help=HELP["t_pit"], format="%.1f s"),
        "Tyres at the flag": C.TextColumn("Tyres at the flag", help=HELP["t_finish_tyre"]),
        "Traffic laps": C.NumberColumn("Traffic laps", help=HELP["t_traffic"], format="%d", width="small"),
        "Finish (sim. group)": C.TextColumn("Finish (sim. group)", help=HELP["t_pos"], width="small"),
        "Why rejected / notes": C.TextColumn("Why rejected / notes", help=HELP["t_notes"], width=420),
    }


def style_verdicts(df: pd.DataFrame):
    colors = {"★": "#B37BFF", "✓": "#3DD68C", "✕": "#FF6464", "↻": "#F2C14E"}

    def paint(v):
        c = colors.get(str(v)[:1])
        return f"color:{c}; font-weight:700" if c else ""
    try:
        return df.style.map(paint, subset=["Verdict"])
    except AttributeError:  # pandas < 2.1
        return df.style.applymap(paint, subset=["Verdict"])


def rivals_frame(result: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Driver": r["code"], "Assumed strategy": r["strategy_label"],
        "Why this strategy": "Its own fastest valid option (model assumption)"
        if r["assumption"].startswith("own") else "No valid option found, so staying out is assumed",
        "Time to flag": fmt_race_time(r["remaining_time_s"]),
        "Finish vs leader now": r["finish_time_ref_s"],
    } for r in result["rivals"].values()])


def rivals_config() -> Dict[str, Any]:
    C = st.column_config
    return {"Driver": C.TextColumn("Driver", help="Three-letter driver code."),
            "Assumed strategy": C.TextColumn("Assumed strategy", help=HELP["rivals"]),
            "Why this strategy": C.TextColumn("Why this strategy", width="large"),
            "Time to flag": C.TextColumn("Time to flag", help=HELP["t_time"]),
            "Finish vs leader now": C.NumberColumn(
                "Finish time (s, leader ref.)", format="%.1f",
                help="Current gap to the leader plus the simulated time to the flag. Lower finishes ahead.")}


def laps_frame(result: Dict[str, Any]) -> pd.DataFrame:
    laps = result["simulations"][result["selected_id"]]["laps"]
    return pd.DataFrame([{
        "Lap": r["lap"], "Tyre": r["compound"].title(), "Tyre age": r["tyre_age"],
        "Track": "Wet" if r["condition"] == "wet" else "Dry", "Safety Car": "Yes" if r["safety_car"] else "",
        "Base pace": r["baseline_s"], "Tyre effect": r["tyre_adj_s"], "Weather effect": r["weather_adj_s"],
        "Safety Car slow-down": r["sc_adj_s"], "Traffic": r["traffic_adj_s"], "Lap time": r["lap_time_s"],
        "Pit-lane time": r["pit_loss_s"], "Race clock": fmt_race_time(r["cumulative_ref_s"]),
    } for r in laps])


def laps_config() -> Dict[str, Any]:
    C = st.column_config
    s = lambda name, h: C.NumberColumn(name, help=h, format="%.3f")  # noqa: E731
    return {
        "Lap": C.NumberColumn("Lap", format="%d", width="small"),
        "Tyre age": C.NumberColumn("Tyre age", help="Laps already done on this set at the start of the lap.",
                                   format="%d"),
        "Base pace": s("Base pace", "Fresh-soft, dry-track lap time for this driver."),
        "Tyre effect": s("Tyre effect", "Compound pace difference plus wear for the tyre's age."),
        "Weather effect": s("Weather effect", "Extra time from a wet track or the wrong tyre for the conditions."),
        "Safety Car slow-down": s("Safety Car slow-down", "Extra time while lapping behind the Safety Car."),
        "Traffic": s("Traffic", "Time lost behind another car after rejoining from the pits."),
        "Lap time": s("Lap time", "Sum of the columns to the left."),
        "Pit-lane time": C.NumberColumn("Pit-lane time", help=HELP["t_pit"], format="%.2f"),
        "Race clock": C.TextColumn("Race clock", help="Elapsed time on the leader's clock at the end of the lap."),
    }


def log_frame(result: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Step": e["step"], "Agent": AGENT_ICON.get(e["agent"], e["agent"]), "Kind": KIND.get(e["kind"], e["kind"]),
        "What happened": action_label(e["action"]), "Details": e["detail"],
    } for e in result["log"]])


def log_config() -> Dict[str, Any]:
    C = st.column_config
    return {"Step": C.NumberColumn("Step", format="%d", width="small"),
            "Agent": C.TextColumn("Agent", help="Which agent (or the orchestrator) acted."),
            "Kind": C.TextColumn("Kind", width="small",
                                 help="Decision = agent choice; Tool run = Python tool executed; Check = "
                                      "Critic validation; Revision = work sent back; LLM call = Ollama used."),
            "What happened": C.TextColumn("What happened"),
            "Details": C.TextColumn("Details", width="large")}


def history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame({
        "When (UTC)": df["timestamp_utc"].astype(str).str.replace("T", " ").str.replace("+00:00", "", regex=False),
        "Scenario": df["scenario"], "After lap": df["decision_after_lap"],
        "Trigger": df["event"].fillna("Manual analysis"),
        "Outcome": df["status"].map({"recommended": "Recommended", "review_required": "Review required"}),
        "Call": df["recommendation"], "Time to flag": df["remaining_time_s"].map(fmt_race_time),
        "Margin": df["margin_s"], "Risk": df["risk"].fillna("–").str.title(), "Revisions": df["revisions"],
        "Question": df["question"], "Data": df["source"].map(lambda s: SOURCE.get(s, s)),
        "Record ID": df["analysis_id"],
    })
    return out.iloc[::-1].reset_index(drop=True)


def history_config() -> Dict[str, Any]:
    C = st.column_config
    return {"After lap": C.NumberColumn("After lap", format="%d", help=HELP["decision_point"]),
            "Trigger": C.TextColumn("Trigger", help="The event that caused this analysis, if any."),
            "Margin": C.NumberColumn("Margin", format="%.2f s", help=HELP["margin"]),
            "Risk": C.TextColumn("Risk", help=HELP["risk"]),
            "Revisions": C.NumberColumn("Revisions", format="%d", help="Times the Critic sent work back."),
            "Record ID": C.TextColumn("Record ID", help="Unique ID of the saved decision record.")}


def provenance_frame(prov: Dict[str, str]) -> pd.DataFrame:
    def kind(v: str) -> str:
        v = v.lower()
        if "synthetic" in v or "assum" in v:
            return "Assumption"
        if "derived" in v:
            return "Derived"
        if "retrieved" in v:
            return "Retrieved"
        return "User / event"
    return pd.DataFrame([{"Input": k.replace("_", " ").replace(" s", " (s)").capitalize(),
                          "Type": kind(v), "Where it comes from": v[:1].upper() + v[1:]}
                         for k, v in prov.items()])


def compounds_frame(cfg) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, p in cfg.compounds.items():
        rows.append({"Tyre": name.title(), "Use": "Dry" if p.is_dry else "Wet",
                     "Pace": p.pace_offset_s, "Wear/lap": p.deg_s_per_lap,
                     "Cliff": p.cliff_age_laps, "Safe life": p.max_stint_laps})
    return pd.DataFrame(rows)


def compounds_config() -> Dict[str, Any]:
    C = st.column_config
    return {"Use": C.TextColumn("Use", help="Track condition the tyre is made for."),
            "Pace": C.NumberColumn("Pace", format="+%.2f s", help="Pace difference of a fresh tyre vs a fresh soft."),
            "Wear/lap": C.NumberColumn("Wear/lap", format="%.3f s", help="Time lost per lap of tyre age at 1.0× wear."),
            "Cliff": C.NumberColumn("Cliff", format="%d laps", help="Age where wear speeds up sharply."),
            "Safe life": C.NumberColumn("Safe life", format="%d laps",
                                        help="Longest safe stint at 1.0× wear; shorter when wear is higher.")}

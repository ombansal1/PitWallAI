"""'What really happened' — compare the agents' call with the real race (hindsight check).

Hindsight separation: the agents only ever receive the race state built from laps up to the
decision lap. `actual_outcome` reads the later laps and is called only AFTER an analysis is
complete, when the user asks to reveal the result. Nothing here feeds back into the agents.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .config import COMPOUNDS, DEFAULT_CONFIG, ModelConfig
from .models import PitStop, Strategy
from .rivals import project_rivals
from .simulation import check_strategy_constraints, simulate_strategy
from .strategies import _label as strategy_label
from .validation import parse_race_state


class RealityError(ValueError):
    pass


def _secs(td) -> Optional[float]:
    return None if td is None or pd.isna(td) else float(pd.Timedelta(td).total_seconds())


def _ranges(laps: List[int]) -> str:
    laps = sorted(set(int(x) for x in laps))
    if not laps:
        return ""
    out, start, prev = [], laps[0], laps[0]
    for x in laps[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        out.append(f"{start}" if start == prev else f"{start}–{prev}")
        if x is not None:
            start = prev = x
    return ", ".join(out)


def _condition(status: str) -> str:
    s = str(status or "1")
    if "5" in s:
        return "red flag"
    if "4" in s:
        return "Safety Car"
    if "6" in s or "7" in s:
        return "Virtual Safety Car"
    return "green"


def _runs(laps_list: List[int]) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for lap in sorted(set(laps_list)):
        if out and out[-1]["end"] == lap - 1:
            out[-1]["end"] = lap
        else:
            out.append({"start": lap, "end": lap})
    return out


def leader_status_by_lap(laps: pd.DataFrame) -> Dict[int, str]:
    """Track status per lap number, taken from the car leading that lap (covers the whole race)."""
    out: Dict[int, str] = {}
    for lap_no, grp in laps.groupby("LapNumber"):
        lead = grp[grp["Position"] == 1] if "Position" in grp.columns else grp.iloc[0:0]
        rows = lead if len(lead) else grp
        out[int(lap_no)] = "".join(sorted(set("".join(str(x) for x in rows["TrackStatus"] if x))))
    return out


def real_events_after(session, decision_lap: int) -> Dict[str, Any]:
    """HINDSIGHT: the Safety Car / VSC / red-flag and wet periods that really happened after the
    decision lap. Only used when the user explicitly asks the agents to be given them."""
    laps = session.laps
    status = leader_status_by_lap(laps)
    sc, vsc, red = [], [], []
    for lap_no, st in status.items():
        if lap_no <= decision_lap:
            continue
        c = _condition(st)
        (sc if c == "Safety Car" else vsc if c == "Virtual Safety Car" else red if c == "red flag" else []).append(lap_no)
    neutral = ([{"kind": "SC", **r} for r in _runs(sc + red)] + [{"kind": "VSC", **r} for r in _runs(vsc)])
    comp = laps[laps["LapNumber"] > decision_lap].dropna(subset=["Compound"])
    share = comp.groupby("LapNumber")["Compound"].agg(
        lambda s: s.astype(str).str.upper().isin(["INTERMEDIATE", "WET"]).mean())
    wet = _runs([int(l) for l, v in share.items() if v > 0.5])
    return {"neutral": sorted(neutral, key=lambda p: p["start"]), "wet": wet,
            "red_flag_laps": _ranges(red)}


def apply_real_events(raw: Dict[str, Any], events: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a real-race state with the real future events given to the agents
    (clearly labelled as hindsight)."""
    import copy
    new = copy.deepcopy(raw)
    done, total = int(raw["completed_laps"]), int(raw["total_laps"])
    new["sc_start_lap"] = new["sc_end_lap"] = None          # replace the 3-lap assumption with reality
    new["neutral_periods"] = [p for p in events["neutral"] if p["end"] >= done + 1 and p["start"] <= total]
    wet = [dict(r) for r in events["wet"] if r["end"] >= done + 1]
    new["rain_start_lap"] = new["rain_end_lap"] = None
    new["wet_periods"] = []
    if raw["track_condition"] == "wet":
        if wet and wet[0]["start"] == done + 1:
            first = wet.pop(0)
            new["rain_end_lap"] = first["end"] if first["end"] < total else None
        else:
            new["track_condition"] = "dry"   # the track was dry from the next lap on
    if wet and new["track_condition"] == "dry":
        first = wet.pop(0)
        new["rain_start_lap"], new["rain_end_lap"] = first["start"], (first["end"] if first["end"] < total else None)
    new["wet_periods"] = wet
    parts = [f"{'Safety Car' if p['kind'] == 'SC' else 'VSC'} laps {p['start']}–{p['end']}" for p in new["neutral_periods"]]
    parts += [f"rain laps {r['start']}–{r['end']}" for r in events["wet"] if r["end"] >= done + 1]
    note = ("HINDSIGHT: the agents were given the real race's events after the decision lap: "
            + ("; ".join(parts) if parts else "none (no Safety Car, VSC or rain happened)") + ".")
    if events.get("red_flag_laps"):
        note += f" Red flag on laps {events['red_flag_laps']} is modelled as a Safety Car."
    new["notes"] = list(raw.get("notes", [])) + [note]
    new["provenance"] = {**raw.get("provenance", {}),
                         "safety_car / VSC / rain": "HINDSIGHT: taken from the real race after the decision lap"}
    new["scenario_name"] = raw["scenario_name"] + " (with the real race's events)"
    new["data_source_label"] = raw["data_source_label"] + "; real Safety Cars/VSC/rain given to the agents (hindsight)"
    if isinstance(new.get("replay"), dict):
        new["replay"]["real_events"] = True
    return new


def completed_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Laps that were actually completed: a crash lap can carry a session time but has neither a
    lap time nor a position."""
    done = laps["Time"].notna() & (laps["LapTime"].notna() | laps["Position"].notna())
    return laps[done]


def classification(laps: pd.DataFrame) -> List[Dict[str, Any]]:
    """Order of the timing data: most laps first, then earliest crossing of the line.
    Post-race penalties are not included."""
    last = (completed_laps(laps).sort_values("LapNumber").groupby("Driver").tail(1)
            .sort_values(["LapNumber", "Time"], ascending=[False, True]))
    winner_laps = int(last["LapNumber"].max())
    finish_t = last[last["LapNumber"] == winner_laps]["Time"].min()
    out = []
    for pos, (_, r) in enumerate(last.iterrows(), start=1):
        n = int(r["LapNumber"])
        # A car that crosses the line after the winner finished is still running (lapped), not retired.
        running = n == winner_laps or (pd.notna(r["Time"]) and r["Time"] >= finish_t - pd.Timedelta(seconds=1))
        status = ("Finished" if n == winner_laps else f"Finished, +{winner_laps - n} lap{'s' if winner_laps - n > 1 else ''}"
                  if running else f"Retired after lap {n}")
        out.append({"pos": pos, "code": r["Driver"], "laps": n, "status": status, "time": r["Time"]})
    return out


def actual_outcome(session, driver: str, decision_lap: int, group: Optional[List[str]] = None) -> Dict[str, Any]:
    """What the team actually did after `decision_lap`, and how it ended (reads FUTURE laps)."""
    laps = session.laps
    d = laps[laps["Driver"] == driver].sort_values("LapNumber").reset_index(drop=True)
    at = d[d["LapNumber"] == decision_lap]
    if at.empty or pd.isna(at.iloc[0]["Time"]):
        raise RealityError(f"No timing for {driver} on lap {decision_lap}.")
    t0 = at.iloc[0]["Time"]
    after = d[d["LapNumber"] >= decision_lap].reset_index(drop=True)
    stops = []
    for i in range(1, len(after)):
        prev, cur = after.iloc[i - 1], after.iloc[i]
        new_stint = pd.notna(cur.get("Stint")) and pd.notna(prev.get("Stint")) and cur["Stint"] > prev["Stint"]
        if not new_stint:
            continue
        comp = str(cur.get("Compound") or "").upper()
        stops.append({"lap": int(prev["LapNumber"]), "compound": comp if comp in COMPOUNDS else None,
                      "under": _condition(prev.get("TrackStatus"))})
    cls = classification(laps)
    me = next((c for c in cls if c["code"] == driver), None)
    winner_laps = cls[0]["laps"] if cls else int(d["LapNumber"].max())
    finished_lead_lap = bool(me and me["laps"] == winner_laps)
    real_time = (_secs(me["time"] - t0) if me and finished_lead_lap else None)
    group_pos = None
    if group:
        order = [c["code"] for c in cls if c["code"] in set(group) | {driver}]
        group_pos = {"position": order.index(driver) + 1 if driver in order else None, "size": len(order),
                     "order": order}
    # Race events after the decision (from the leader's laps, which cover the whole race).
    events: Dict[str, List[int]] = {"Safety Car": [], "Virtual Safety Car": [], "red flag": []}
    for lap_no, st in leader_status_by_lap(laps).items():
        c = _condition(st)
        if lap_no > decision_lap and c in events:
            events[c].append(int(lap_no))
    comp_after = laps[laps["LapNumber"] > decision_lap].dropna(subset=["Compound"])
    wet_share = comp_after.groupby("LapNumber")["Compound"].agg(
        lambda s: s.astype(str).str.upper().isin(["INTERMEDIATE", "WET"]).mean())
    wet_laps = [int(lap_no) for lap_no, share in wet_share.items() if share > 0.5]
    laps_after = [{"lap": int(r["LapNumber"]), "lap_time_s": _secs(r["LapTime"]),
                   "compound": str(r.get("Compound") or "").upper() or None,
                   "condition": _condition(r.get("TrackStatus")),
                   "pit_in": pd.notna(r.get("PitInTime")), "pit_out": pd.notna(r.get("PitOutTime"))}
                  for _, r in d[d["LapNumber"] > decision_lap].iterrows()]
    return {
        "driver": driver, "decision_lap": int(decision_lap), "race_laps": int(winner_laps),
        "stops": stops,
        "strategy_label": strategy_label([PitStop(s["lap"], s["compound"] or "UNKNOWN") for s in stops]),
        "status": me["status"] if me else "Unknown", "finish_position": me["pos"] if me else None,
        "field_size": len(cls), "finished_lead_lap": finished_lead_lap, "real_time_to_flag_s": real_time,
        "group": group_pos,
        "events_after": {k: _ranges(v) for k, v in events.items() if v} | (
            {"wet track": _ranges(wet_laps)} if wet_laps else {}),
        "laps_after": laps_after,
        "winner": cls[0]["code"] if cls else None,
        "note": "Finishing order comes from the timing data; post-race penalties are not included.",
    }


def _verdict(model_stops: List[Dict], real_stops: List[Dict]) -> Dict[str, str]:
    if not model_stops and not real_stops:
        return {"code": "match", "text": "Match: both stayed out to the flag."}
    if len(model_stops) != len(real_stops):
        n_m, n_r = len(model_stops), len(real_stops)
        return {"code": "different", "text": f"Different strategy: from this point the agents planned {n_m} "
                                             f"stop{'s' if n_m != 1 else ''}, the team made {n_r}."}
    m, r = model_stops[0], real_stops[0]
    dl = r["lap"] - m["lap"]
    same_comp = (m["compound"] == r["compound"])
    if same_comp and abs(dl) <= 2:
        return {"code": "match", "text": f"Match: the team also boxed on lap {r['lap']} for "
                                         f"{(r['compound'] or 'unknown').title()} tyres."}
    if same_comp:
        when = "later" if dl > 0 else "earlier"
        return {"code": "timing", "text": f"Same tyres, different timing: the team boxed {abs(dl)} laps {when} "
                                          f"(lap {r['lap']} vs lap {m['lap']})."}
    return {"code": "tyres", "text": f"Different tyre choice: the team fitted {(r['compound'] or 'unknown').title()}, "
                                     f"the agents chose {m['compound'].title()}."}


def compare_with_reality(result: Dict[str, Any], outcome: Dict[str, Any],
                         cfg: ModelConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Score the agents' call against the real race. All 'model' numbers are model estimates."""
    rec = result["recommendation"]
    raw = result["input_snapshot"]
    state = parse_race_state(raw, cfg)
    model_stops = (rec.get("stops") or []) if rec["status"] == "recommended" else None
    out: Dict[str, Any] = {"analysis_id": result["analysis_id"], "driver": outcome["driver"],
                           "decision_lap": outcome["decision_lap"],
                           "model_call": rec["headline"], "team_call": outcome["strategy_label"],
                           "real_stops": outcome["stops"], "model_stops": model_stops,
                           "events_after": outcome["events_after"], "status": outcome["status"],
                           "finish_position": outcome["finish_position"], "field_size": outcome["field_size"]}
    out["verdict"] = (_verdict(model_stops, outcome["stops"]) if model_stops is not None
                      else {"code": "none", "text": "The agents made no call (review required)."})

    # Re-score the team's real strategy with the same model and assumptions.
    usable = [s for s in outcome["stops"] if s["compound"] in COMPOUNDS]
    team = Strategy("TEAM_ACTUAL", outcome["strategy_label"], "actual",
                    [PitStop(s["lap"], s["compound"]) for s in usable])
    rivals = project_rivals(state, cfg) if state.competitors else {}
    traces = {c: r["trace"] for c, r in rivals.items()} or None
    in_window = all(state.first_lap <= s.lap <= state.total_laps - 1 for s in team.stops)
    team_sim = simulate_strategy(state, state.target, team, cfg, rival_traces=traces) \
        if in_window and len(usable) == len(outcome["stops"]) else None
    rules = check_strategy_constraints(state, state.target, team, cfg) if team_sim else []
    out["team_strategy_rule_notes"] = [v["message"] for v in rules]
    model_t = rec["numbers"]["remaining_time_s"] if rec["status"] == "recommended" else None
    team_model_t = round(team_sim.remaining_time_s, 3) if team_sim else None
    real_t = outcome["real_time_to_flag_s"]
    out["times"] = {"model_estimate_for_agents_call_s": model_t,
                    "model_estimate_for_team_call_s": team_model_t,
                    "real_time_for_team_call_s": round(real_t, 3) if real_t is not None else None}
    if model_t is not None and team_model_t is not None:
        out["model_gain_vs_team_s"] = round(team_model_t - model_t, 3)  # + = model expected its call faster
    if team_sim and real_t is not None:
        err = real_t - team_sim.remaining_time_s
        out["model_error_s"] = round(err, 3)
        out["model_error_pct"] = round(100 * err / real_t, 2)
        # Where did the model miss? Split by real lap condition.
        model_by_lap = {r["lap"]: r["lap_time_s"] + r["pit_loss_s"] for r in team_sim.laps}
        split: Dict[str, float] = {}
        for lap in outcome["laps_after"]:
            if lap["lap_time_s"] is None or lap["lap"] not in model_by_lap:
                continue
            # neutralised laps first (a stop under the Safety Car is mostly a Safety Car effect)
            key = (f"{lap['condition']} laps" if lap["condition"] != "green" else
                   "pit-stop laps" if (lap["pit_in"] or lap["pit_out"]) else "green-flag laps")
            split[key] = split.get(key, 0.0) + (lap["lap_time_s"] - model_by_lap[lap["lap"]])
        out["error_by_condition_s"] = {k: round(v, 2) for k, v in split.items()}
        out["lap_compare"] = [{"lap": lap["lap"], "real_s": lap["lap_time_s"], "model_s": model_by_lap.get(lap["lap"]),
                               "condition": lap["condition"], "pit": bool(lap["pit_in"] or lap["pit_out"])}
                              for lap in outcome["laps_after"]]
    proj = result.get("positions", {}).get(result.get("selected_id") or "", {})
    out["position"] = {"projected_in_group": proj.get("position_in_group"),
                       "real_in_group": (outcome.get("group") or {}).get("position"),
                       "group_size": (outcome.get("group") or {}).get("size")}
    out["explanations"] = _explain(out, outcome)
    out["note"] = outcome["note"]
    return out


def _explain(cmp: Dict[str, Any], outcome: Dict[str, Any]) -> List[str]:
    lines = []
    ev = outcome["events_after"]
    if ev:
        lines.append("After the decision lap the race had: " + "; ".join(f"{k} on laps {v}" for k, v in ev.items())
                     + ". The agents couldn't know this at the decision point.")
    else:
        lines.append("No Safety Car, red flag or rain after the decision lap, so this is a fair test of the model.")
    err = cmp.get("model_error_s")
    split = cmp.get("error_by_condition_s", {})
    if err is not None:
        side = "slower" if err > 0 else "faster"
        lines.append(f"The team's real strategy took {abs(err):.1f} s {side} than the model estimated for that "
                     f"same strategy ({abs(cmp['model_error_pct']):.1f}%).")
        big = sorted(split.items(), key=lambda kv: -abs(kv[1]))[:2]
        if big:
            lines.append("Biggest differences: " + "; ".join(f"{k} {v:+.1f} s" for k, v in big) + ".")
    elif outcome["status"] != "Finished":
        lines.append(f"Real time can't be compared: the car's result was '{outcome['status']}'.")
    if cmp.get("model_gain_vs_team_s") is not None:
        g = cmp["model_gain_vs_team_s"]
        if abs(g) < 0.05:
            lines.append("The model rated the agents' call and the team's call the same.")
        else:
            lines.append(f"By the model's own numbers, the agents' call was {abs(g):.1f} s "
                         f"{'faster' if g > 0 else 'slower'} than what the team did (a model estimate, not a fact).")
    if cmp.get("team_strategy_rule_notes"):
        lines.append("The model's rules would have rejected the team's strategy: "
                     + cmp["team_strategy_rule_notes"][0] + " Real teams often run longer than these generic limits.")
    p = cmp["position"]
    if p["projected_in_group"] and p["real_in_group"]:
        lines.append(f"Finishing place among the simulated cars: projected P{p['projected_in_group']}, "
                     f"real P{p['real_in_group']} of {p['group_size']}.")
    return lines

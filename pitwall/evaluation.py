"""Agents vs real teams: run the five agents at real race moments and compare with what the team did.

Moments: for each built-in race, the drivers classified P1, P5 and P10, a few laps before their first real
pit stop (the moment a strategist is deciding when to box). The agents see only laps up to that point.
Each moment is run twice: with the hand-written tyre formula and with the ML tyre model, so the two
tyre models can be compared on real decisions, not just on lap times.

Used by scripts/evaluate_agents.py (writes data/eval/agents_vs_reality.json, shown in the Model Lab).
Runs with the rule-based agents (no AI model), so the results are exactly repeatable. Note: the shipped
ML tyre model is trained on every season, so for 2019-2024 its numbers here are partly in-sample; read
the 2025-2026 seasons (or the Model card's unseen-season test) for a fair comparison.
"""
from __future__ import annotations

import copy
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .data_sources.fastf1_source import DataSourceError, build_race_state
from .data_sources.replays import load_index, load_replay
from .ml.tyre_model import MLUnavailable, tyre_model_payload
from .orchestrator import run_analysis
from .reality import RealityError, actual_outcome, classification, compare_with_reality

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "agents_vs_reality.json"
POSITIONS = (1, 5, 10)
LEAD_LAPS = 3
VERDICT_TEXT = {"match": "Same call (tyre and lap within 2)", "timing": "Same tyre, different lap",
                "tyres": "Different tyre", "different": "Different number of stops", "none": "No call"}


def _first_stop(session, driver: str) -> Optional[int]:
    laps = session.laps
    d = laps[laps["Driver"] == driver].sort_values("LapNumber")
    prev = None
    for _, r in d.iterrows():
        if prev is not None and r["Stint"] == r["Stint"] and prev["Stint"] == prev["Stint"] \
                and r["Stint"] > prev["Stint"]:
            return int(prev["LapNumber"])
        prev = r
    return None


def race_moments(session, positions=POSITIONS, lead: int = LEAD_LAPS) -> List[Dict[str, Any]]:
    cls = classification(session.laps)
    out = []
    for p in positions:
        if p > len(cls):
            continue
        drv = cls[p - 1]["code"]
        stop = _first_stop(session, drv)
        if stop is None or stop - lead < 5:
            continue
        out.append({"driver": drv, "finish": p, "lap": stop - lead, "real_first_stop": stop})
    return out


def evaluate(race_ids: Optional[Iterable[str]] = None, progress=None) -> Dict[str, Any]:
    say = progress or (lambda msg: None)
    ids = list(race_ids) if race_ids is not None else [r["id"] for r in load_index()["races"]]
    rows: List[Dict[str, Any]] = []
    skipped: Counter = Counter()
    for n, rid in enumerate(ids, 1):
        sess = load_replay(rid)
        year = int(rid[:4])
        for m in race_moments(sess):
            try:
                raw = build_race_state(sess, m["driver"], m["lap"], source="replay")
                group = [c["code"] for c in raw["competitors"]]
                outcome = actual_outcome(sess, m["driver"], m["lap"], group)
            except (DataSourceError, RealityError) as e:
                skipped[type(e).__name__] += 1
                continue
            for model in ("formula", "ml"):
                r = copy.deepcopy(raw)
                if model == "ml":
                    try:
                        r["tyre_model"] = tyre_model_payload(raw["track_name"])
                    except MLUnavailable:
                        continue
                res = run_analysis(r)
                cmp = compare_with_reality(res, outcome)
                ms, rs = cmp["model_stops"] or [], cmp["real_stops"]
                rows.append({
                    "race": f"{year} {raw['track_name']}", "year": year, "driver": m["driver"],
                    "finish": m["finish"], "decision_lap": m["lap"], "tyre_model": model,
                    "verdict": cmp["verdict"]["code"], "agents_call": cmp["model_call"], "team_call": cmp["team_call"],
                    "agents_first_stop": ms[0]["lap"] if ms else None,
                    "agents_compound": ms[0]["compound"] if ms else None,
                    "team_first_stop": rs[0]["lap"] if rs else None, "team_compound": rs[0]["compound"] if rs else None,
                    "stops_agents": len(ms), "stops_team": len(rs),
                    "safety_car_after": any(k in cmp["events_after"] for k in ("Safety Car", "Virtual Safety Car",
                                                                                 "red flag")),
                    "wet_after": "wet track" in cmp["events_after"],
                    "model_error_s": cmp.get("model_error_s"),
                })
        if n % 10 == 0 or n == len(ids):
            say(f"{n}/{len(ids)} races, {len(rows)} runs")
    return {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": f"Drivers finishing P{', P'.join(map(str, POSITIONS))}, {LEAD_LAPS} laps before their first "
                      "real pit stop; rule-based agents (repeatable); each moment run with both tyre models.",
            "summary": {m: summarise([r for r in rows if r["tyre_model"] == m]) for m in ("formula", "ml")},
            "skipped": dict(skipped), "rows": rows}


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    n = len(rows)
    verdicts = Counter(r["verdict"] for r in rows)
    both = [r for r in rows if r["agents_first_stop"] is not None and r["team_first_stop"] is not None]
    lap_err = [abs(r["agents_first_stop"] - r["team_first_stop"]) for r in both]
    err = [abs(r["model_error_s"]) for r in rows if r["model_error_s"] is not None]
    green = [r for r in rows if not r["safety_car_after"]]
    clean = [abs(r["model_error_s"]) for r in green if not r.get("wet_after") and r["model_error_s"] is not None]
    return {
        "runs": n,
        "verdicts": {k: verdicts.get(k, 0) for k in VERDICT_TEXT},
        "same_call_pct": round(100 * verdicts.get("match", 0) / n, 1),
        "same_tyre_pct": round(100 * sum(1 for r in both if r["agents_compound"] == r["team_compound"])
                               / max(1, len(both)), 1),
        "same_stop_count_pct": round(100 * sum(1 for r in rows if r["stops_agents"] == r["stops_team"]) / n, 1),
        "median_stop_lap_difference": statistics.median(lap_err) if lap_err else None,
        "same_call_pct_no_safety_car": round(100 * sum(1 for r in green if r["verdict"] == "match")
                                             / max(1, len(green)), 1),
        "median_time_error_s": round(statistics.median(err), 2) if err else None,
        # Fair tyre-model comparison: races that stayed green and dry after the decision (no SC/VSC/red/rain).
        "median_time_error_clean_s": round(statistics.median(clean), 2) if clean else None,
        "clean_runs": len(clean),
        "by_season": {str(y): round(100 * sum(1 for r in rows if r["year"] == y and r["verdict"] == "match")
                                    / max(1, sum(1 for r in rows if r["year"] == y)), 1)
                      for y in sorted({r["year"] for r in rows})},
        "clean_time_error_by_season": {
            str(y): round(statistics.median(v), 1) for y in sorted({r["year"] for r in rows})
            if (v := [abs(r["model_error_s"]) for r in rows if r["year"] == y and r["model_error_s"] is not None
                      and not r["safety_car_after"] and not r.get("wet_after")])},
    }


def save(result: Dict[str, Any], path: Path = EVAL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def load(path: Path = EVAL_PATH) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

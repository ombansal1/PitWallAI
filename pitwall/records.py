"""Decision records: one structured record per completed analysis, stored as JSON Lines.
Appends are idempotent by analysis_id, so UI reruns cannot create duplicates."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "decisions" / "decision_log.jsonl"


def build_decision_record(result: Dict[str, Any]) -> Dict[str, Any]:
    rec = result.get("recommendation") or {}
    return {
        "analysis_id": result["analysis_id"],
        "timestamp_utc": result["created_at"],
        "question": result["question"],
        "model_mode": result["model_label"],
        "data_source": {"source": result.get("data_source"), "label": result.get("data_source_label"),
                        "provenance": (result.get("input_snapshot") or {}).get("provenance", {})},
        "input_fingerprint": result["input_fingerprint"],
        "input_snapshot": result["input_snapshot"],
        "triggering_event": result.get("trigger_event"),
        "plan_versions": result.get("plan_history"),
        "candidates": result.get("candidates"),
        "critic_decisions": [{"iteration": r["iteration"], "accepted": r["accepted"], "rejected": r["rejected"],
                              "revision_requests": [{k: q[k] for k in ("target", "code", "reason")}
                                                    for q in r["revision_requests"]]}
                             for r in result.get("critic_reports", [])],
        "revisions": result.get("revisions"),
        "retry_count": result.get("retry_count"),
        "status": result["status"],
        "selected_strategy": result.get("selected_id"),
        "final_recommendation": {k: rec.get(k) for k in
                                 ("headline", "summary", "numbers", "trade_offs", "risk", "reason", "radio",
                                  "llm_used", "llm_summary") if k in rec},
    }


def default_log_path() -> Path:
    """PITWALL_DECISION_LOG overrides the location (used by tests)."""
    return Path(os.environ.get("PITWALL_DECISION_LOG", DEFAULT_LOG_PATH))


def build_reality_record(cmp: Dict[str, Any], timestamp_utc: str) -> Dict[str, Any]:
    """Hindsight check for one analysis. Stored next to (never inside) the decision record."""
    return {"record_type": "reality_check", "analysis_id": f"{cmp['analysis_id']}-RC",
            "for_analysis": cmp["analysis_id"], "timestamp_utc": timestamp_utc,
            "verdict": cmp["verdict"], "agents_call": cmp["model_call"], "team_call": cmp["team_call"],
            "times": cmp["times"], "model_error_s": cmp.get("model_error_s"),
            "events_after_decision": cmp["events_after"], "explanations": cmp["explanations"]}


class DecisionStore:
    def __init__(self, path: os.PathLike = None):
        self.path = Path(path) if path is not None else default_log_path()

    def load_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a corrupted line rather than failing the UI
        return out

    def ids(self) -> set:
        return {r.get("analysis_id") for r in self.load_all()}

    def append(self, record: Dict[str, Any]) -> bool:
        """Append once; returns False if this analysis_id is already stored."""
        if record["analysis_id"] in self.ids():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return True

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    # ---- exports
    def summary_frame(self) -> pd.DataFrame:
        rows = []
        records = self.load_all()
        checks = {r.get("for_analysis"): r for r in records if r.get("record_type") == "reality_check"}
        for r in records:
            if r.get("record_type") == "reality_check":
                continue
            fr = r.get("final_recommendation") or {}
            snap = r.get("input_snapshot") or {}
            rows.append({
                "analysis_id": r.get("analysis_id"), "timestamp_utc": r.get("timestamp_utc"),
                "scenario": snap.get("scenario_name"), "source": (r.get("data_source") or {}).get("source"),
                "decision_after_lap": snap.get("completed_laps"), "question": r.get("question"),
                "event": (r.get("triggering_event") or {}).get("description"),
                "status": r.get("status"), "recommendation": fr.get("headline"),
                "remaining_time_s": (fr.get("numbers") or {}).get("remaining_time_s"),
                "margin_s": (fr.get("numbers") or {}).get("margin_to_next_best_s"),
                "risk": (fr.get("risk") or {}).get("category"), "revisions": r.get("retry_count"),
                "real_race_check": ((checks.get(r.get("analysis_id")) or {}).get("verdict") or {}).get("text"),
            })
        return pd.DataFrame(rows)

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.load_all(), indent=2, default=str).encode("utf-8")

    def to_csv_bytes(self) -> bytes:
        buf = io.StringIO()
        self.summary_frame().to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

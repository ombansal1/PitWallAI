"""Bundled synthetic scenarios (fictional drivers/circuit; no downloads required)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

SCENARIO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "scenarios"
SCENARIO_ORDER = ["dry_normal", "safety_car", "rain_approaching", "high_degradation"]


class ScenarioNotFound(FileNotFoundError):
    pass


def list_scenarios(directory: Path = SCENARIO_DIR) -> List[Dict[str, str]]:
    found = {p.stem: p for p in directory.glob("*.json")}
    order = [s for s in SCENARIO_ORDER if s in found] + sorted(set(found) - set(SCENARIO_ORDER))
    out = []
    for sid in order:
        with found[sid].open(encoding="utf-8") as f:
            out.append({"id": sid, "name": json.load(f).get("scenario_name", sid)})
    return out


def load_scenario(scenario_id: str, directory: Path = SCENARIO_DIR) -> Dict[str, Any]:
    path = directory / f"{scenario_id}.json"
    if not path.exists():
        raise ScenarioNotFound(f"scenario '{scenario_id}' not found in {directory}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)

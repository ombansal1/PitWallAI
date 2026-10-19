"""Race events. Each event returns an updated copy of the race-state dict at the SAME
decision point (no lap advancement); the original dict is never mutated."""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

from .config import VALIDATION_BOUNDS

EVENT_TYPES = ("safety_car", "vsc", "rain", "degradation")


class EventError(ValueError):
    pass


def _int(event: Dict[str, Any], key: str, default=None) -> int:
    v = event.get(key, default)
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
        raise EventError(f"{key} must be a whole number (got {v!r})")
    return int(v)


def apply_event(raw: Dict[str, Any], event: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    etype = event.get("type")
    if etype not in EVENT_TYPES:
        raise EventError(f"unknown event type {etype!r}")
    new = copy.deepcopy(raw)
    done, total = int(raw["completed_laps"]), int(raw["total_laps"])

    if etype in ("safety_car", "vsc"):
        name = "Safety Car" if etype == "safety_car" else "Virtual Safety Car"
        start = done + 1 + _int(event, "laps_until", 0)
        dur = _int(event, "duration_laps", 4 if etype == "safety_car" else 2)
        if dur < 1:
            raise EventError(f"{name} duration must be at least 1 lap")
        if start > total:
            raise EventError(f"{name} would start after the final lap ({total})")
        end = min(total, start + dur - 1)
        main_free = raw.get("sc_start_lap") is None or (raw.get("sc_end_lap") or 0) < done + 1
        if etype == "safety_car" and main_free:
            new["sc_start_lap"], new["sc_end_lap"] = start, end
        else:  # a second Safety Car, or any VSC, is added to the list of periods
            new["neutral_periods"] = list(raw.get("neutral_periods") or []) + [
                {"kind": "SC" if etype == "safety_car" else "VSC", "start": start, "end": end}]
        desc = f"{name} laps {start}-{end}"
    elif etype == "rain":
        start = done + _int(event, "laps_until", 5)
        if start < done + 1 or start > total:
            raise EventError(f"rain must start within the remaining race (laps {done + 1}-{total})")
        dur = _int(event, "duration_laps", 0)  # 0 = until the end
        end = None if dur == 0 else min(total, start + dur - 1)
        if raw.get("rain_start_lap") is None:
            new["rain_start_lap"], new["rain_end_lap"] = start, end
        else:  # another wet spell
            new["wet_periods"] = list(raw.get("wet_periods") or []) + [{"start": start, "end": end or total}]
        desc = (f"Simulated rain forecast: wet from lap {start}"
                + (f" to lap {end}" if end else " to the finish"))
    else:
        factor = event.get("factor", 1.5)
        if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not 1.0 < factor <= 3.0:
            raise EventError("degradation factor must be > 1.0 and ≤ 3.0")
        value = round(float(raw["deg_multiplier"]) * float(factor), 4)
        lo, hi = VALIDATION_BOUNDS["deg_multiplier"]
        if value > hi:
            raise EventError(f"resulting degradation multiplier {value} exceeds the limit {hi}")
        new["deg_multiplier"] = value
        desc = f"Tyre degradation x{factor:g} (multiplier {raw['deg_multiplier']:g} → {value:g})"

    record = {"type": etype, "params": {k: v for k, v in event.items() if k != "type"},
              "description": desc, "at_completed_laps": done}
    new.setdefault("events", [])
    new["events"] = list(new["events"]) + [record]
    new.setdefault("provenance", {})
    new["provenance"] = {**new["provenance"], f"event:{etype}": "simulated event (user-applied)"}
    return new, record

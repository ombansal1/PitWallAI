"""Input validation for race states. Invalid user inputs are reported, never repaired."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .config import COMPOUNDS, DEFAULT_CONFIG, VALIDATION_BOUNDS, ModelConfig
from .models import DriverState, RaceState


class InputValidationError(ValueError):
    def __init__(self, issues: List[Dict[str, str]]):
        self.issues = issues
        super().__init__("; ".join(f"{i['field']}: {i['message']}" for i in issues))


_TOP_REQUIRED = ["total_laps", "completed_laps", "pit_loss_s", "track_condition",
                 "deg_multiplier", "target", "competitors"]
_DRIVER_REQUIRED = ["code", "gap_to_leader_s", "baseline_pace_s", "compound",
                    "tyre_age_laps", "available_sets"]


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_num(issues, field, value, bounds, integer=False):
    if value is None:
        issues.append({"field": field, "message": "missing value"})
        return False
    if not _is_number(value):
        issues.append({"field": field, "message": f"must be a number (got {value!r})"})
        return False
    if not math.isfinite(float(value)):
        issues.append({"field": field, "message": "must be finite (NaN/inf not allowed)"})
        return False
    if integer and float(value) != int(value):
        issues.append({"field": field, "message": "must be a whole number of laps"})
        return False
    lo, hi = bounds
    if not lo <= value <= hi:
        issues.append({"field": field, "message": f"must be between {lo} and {hi} (got {value})"})
        return False
    return True


def _validate_driver(issues, prefix, d: Any, completed: Optional[int]):
    if not isinstance(d, dict):
        issues.append({"field": prefix, "message": "driver entry must be an object"})
        return
    for key in _DRIVER_REQUIRED:
        if key not in d:
            issues.append({"field": f"{prefix}.{key}", "message": "missing field"})
    code = d.get("code")
    if "code" in d and (not isinstance(code, str) or not code.strip()):
        issues.append({"field": f"{prefix}.code", "message": "must be a non-empty string"})
    if "gap_to_leader_s" in d:
        _check_num(issues, f"{prefix}.gap_to_leader_s", d["gap_to_leader_s"],
                   VALIDATION_BOUNDS["gap_to_leader_s"])
    if "baseline_pace_s" in d:
        _check_num(issues, f"{prefix}.baseline_pace_s", d["baseline_pace_s"],
                   VALIDATION_BOUNDS["baseline_pace_s"])
    if "tyre_age_laps" in d:
        _check_num(issues, f"{prefix}.tyre_age_laps", d["tyre_age_laps"],
                   VALIDATION_BOUNDS["tyre_age_laps"], integer=True)
    if "compound" in d and d["compound"] not in COMPOUNDS:
        issues.append({"field": f"{prefix}.compound",
                       "message": f"unknown compound {d['compound']!r}; use one of {list(COMPOUNDS)}"})
    sets = d.get("available_sets")
    if "available_sets" in d:
        if not isinstance(sets, dict):
            issues.append({"field": f"{prefix}.available_sets", "message": "must map compound -> sets"})
        else:
            for comp, n in sets.items():
                if comp not in COMPOUNDS:
                    issues.append({"field": f"{prefix}.available_sets",
                                   "message": f"unknown compound {comp!r}"})
                else:
                    _check_num(issues, f"{prefix}.available_sets.{comp}", n,
                               VALIDATION_BOUNDS["available_sets"], integer=True)
    used = d.get("compounds_used", [])
    if not isinstance(used, list) or any(c not in COMPOUNDS for c in used):
        issues.append({"field": f"{prefix}.compounds_used",
                       "message": f"must be a list of known compounds (got {used!r})"})
    stops = d.get("stops_made", 0)
    _check_num(issues, f"{prefix}.stops_made", stops, (0, 10), integer=True)


def validate_race_input(raw: Any, cfg: ModelConfig = DEFAULT_CONFIG) -> List[Dict[str, str]]:
    """Return a list of {field, message} issues. Empty list = valid."""
    issues: List[Dict[str, str]] = []
    if not isinstance(raw, dict):
        return [{"field": "race_state", "message": "race state must be an object"}]
    for key in _TOP_REQUIRED:
        if key not in raw or raw[key] is None:
            issues.append({"field": key, "message": "missing field"})

    ok_total = "total_laps" in raw and _check_num(
        issues, "total_laps", raw.get("total_laps"), VALIDATION_BOUNDS["total_laps"], integer=True)
    ok_done = "completed_laps" in raw and _check_num(
        issues, "completed_laps", raw.get("completed_laps"), VALIDATION_BOUNDS["completed_laps"],
        integer=True)
    total = int(raw["total_laps"]) if ok_total else None
    done = int(raw["completed_laps"]) if ok_done else None
    if total is not None and done is not None and done >= total:
        issues.append({"field": "completed_laps",
                       "message": f"must be less than total_laps ({total}); no laps remain to decide on"})
    if "pit_loss_s" in raw:
        _check_num(issues, "pit_loss_s", raw.get("pit_loss_s"), VALIDATION_BOUNDS["pit_loss_s"])
    if "deg_multiplier" in raw:
        _check_num(issues, "deg_multiplier", raw.get("deg_multiplier"),
                   VALIDATION_BOUNDS["deg_multiplier"])
    if "track_condition" in raw and raw.get("track_condition") not in ("dry", "wet"):
        issues.append({"field": "track_condition", "message": "must be 'dry' or 'wet'"})

    # Optional lap windows must lie inside the remaining race.
    def _lap_field(name):
        v = raw.get(name)
        if v is None:
            return None
        if _check_num(issues, name, v, (1, 100), integer=True):
            v = int(v)
            if done is not None and total is not None and not (done + 1 <= v <= total):
                issues.append({"field": name,
                               "message": f"must be within the remaining race (laps {done + 1}-{total})"})
                return None
            return v
        return None

    rs, re_, ss, se = (_lap_field("rain_start_lap"), _lap_field("rain_end_lap"),
                       _lap_field("sc_start_lap"), _lap_field("sc_end_lap"))
    if rs is not None and re_ is not None and re_ < rs:
        issues.append({"field": "rain_end_lap", "message": "must not be before rain_start_lap"})
    if (raw.get("sc_start_lap") is None) != (raw.get("sc_end_lap") is None):
        issues.append({"field": "sc_end_lap", "message": "Safety Car needs both start and end laps"})
    if ss is not None and se is not None and se < ss:
        issues.append({"field": "sc_end_lap", "message": "must not be before sc_start_lap"})

    def _periods(name: str, need_kind: bool):
        val = raw.get(name)
        if val is None:
            return
        if not isinstance(val, list):
            issues.append({"field": name, "message": "must be a list of periods"})
            return
        for i, p in enumerate(val):
            f = f"{name}[{i}]"
            if not isinstance(p, dict):
                issues.append({"field": f, "message": "must be an object with start and end laps"})
                continue
            if need_kind and p.get("kind") not in ("SC", "VSC"):
                issues.append({"field": f + ".kind", "message": "must be 'SC' or 'VSC'"})
            ok_s = _check_num(issues, f + ".start", p.get("start"), (1, 200), integer=True)
            ok_e = _check_num(issues, f + ".end", p.get("end"), (1, 200), integer=True)
            if ok_s and ok_e:
                if p["end"] < p["start"]:
                    issues.append({"field": f + ".end", "message": "must not be before the start lap"})
                elif total is not None and done is not None and (p["end"] < done + 1 or p["start"] > total):
                    issues.append({"field": f, "message": f"must overlap the remaining race (laps {done + 1}-{total})"})

    _periods("neutral_periods", True)
    _periods("wet_periods", False)

    tm = raw.get("tyre_model")
    if tm is not None:
        curves = tm.get("curves") if isinstance(tm, dict) else None
        if not isinstance(tm, dict) or tm.get("kind") != "ml" or not isinstance(curves, dict) or not curves:
            issues.append({"field": "tyre_model", "message": "must be an ML tyre model with curves per compound"})
        else:
            for comp, curve in curves.items():
                if comp not in cfg.compounds:
                    issues.append({"field": f"tyre_model.curves.{comp}", "message": "unknown compound"})
                elif (not isinstance(curve, list) or len(curve) < 2
                      or not all(_is_number(x) and math.isfinite(float(x)) and -5 <= x <= 30 for x in curve)):
                    issues.append({"field": f"tyre_model.curves.{comp}",
                                   "message": "must be a list of at least 2 finite numbers of seconds (-5 to 30)"})
            pc = tm.get("progress_curve")
            if pc is not None and (not isinstance(pc, list) or len(pc) < 2 or not all(
                    _is_number(x) and math.isfinite(float(x)) and -20 <= x <= 5 for x in pc)):
                issues.append({"field": "tyre_model.progress_curve",
                               "message": "must be a list of at least 2 finite numbers of seconds"})

    if raw.get("fuel_effect_s_per_lap") is not None:
        _check_num(issues, "fuel_effect_s_per_lap", raw.get("fuel_effect_s_per_lap"), (0.0, 0.2))

    closed = raw.get("pit_closed_laps", [])
    if not isinstance(closed, list) or any(not _is_number(x) or float(x) != int(x) for x in closed):
        issues.append({"field": "pit_closed_laps", "message": "must be a list of lap numbers"})

    if "target" in raw and raw.get("target") is not None:
        _validate_driver(issues, "target", raw["target"], done)
    comps = raw.get("competitors")
    if "competitors" in raw and comps is not None:
        if not isinstance(comps, list):
            issues.append({"field": "competitors", "message": "must be a list"})
        else:
            for i, c in enumerate(comps):
                _validate_driver(issues, f"competitors[{i}]", c, done)
            codes = [c.get("code") for c in comps if isinstance(c, dict)]
            tcode = raw.get("target", {}).get("code") if isinstance(raw.get("target"), dict) else None
            if len(set(codes)) != len(codes):
                issues.append({"field": "competitors", "message": "driver codes must be unique"})
            if tcode in codes:
                issues.append({"field": "competitors",
                               "message": f"target driver {tcode} is also listed as a competitor"})
    return issues


def rebased_pace(d: Dict[str, Any], state_like, cfg: ModelConfig = DEFAULT_CONFIG) -> Optional[float]:
    """Base pace re-measured with the active tyre model from the driver's recent real laps
    (`pace_samples`, fuel-normalised to the decision lap). None when there are no samples."""
    from .simulation import tyre_adj
    samples = [x for x in d.get("pace_samples") or []
               if isinstance(x, dict) and x.get("compound") in cfg.compounds and _is_number(x.get("time"))]
    if not samples:
        return None

    def lap_time(x):   # normalise the lap to the decision lap's fuel load with the active progress model
        if getattr(state_like, "progress_curve", None) and _is_number(x.get("raw_time")) and _is_number(x.get("lap")):
            return float(x["raw_time"]) - state_like.progress_effect(int(x["lap"]))
        return float(x["time"])
    vals = sorted(lap_time(x) - float(x.get("wet_adj", 0.0))
                  - tyre_adj(state_like, x["compound"], float(x.get("age", 0)), cfg, multiplier=1.0)
                  for x in samples)
    n = len(vals)
    return round(vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2, 3)


def _driver_from(d: Dict[str, Any], pace: Optional[float] = None) -> DriverState:
    return DriverState(
        code=str(d["code"]).strip().upper(),
        name=str(d.get("name", d["code"])),
        team=str(d.get("team", "")),
        gap_to_leader_s=float(d["gap_to_leader_s"]),
        baseline_pace_s=float(d["baseline_pace_s"] if pace is None else pace),
        compound=d["compound"],
        tyre_age_laps=int(d["tyre_age_laps"]),
        compounds_used=list(dict.fromkeys(list(d.get("compounds_used", [])) + [d["compound"]])),
        available_sets={k: int(v) for k, v in d["available_sets"].items()},
        stops_made=int(d.get("stops_made", 0)),
    )


def parse_race_state(raw: Dict[str, Any], cfg: ModelConfig = DEFAULT_CONFIG) -> RaceState:
    """Validate and build a RaceState. Raises InputValidationError; never alters values."""
    issues = validate_race_input(raw, cfg)
    if issues:
        raise InputValidationError(issues)
    opt = lambda k: int(raw[k]) if raw.get(k) is not None else None  # noqa: E731
    tm = raw.get("tyre_model") or {}
    curves = {k: [float(x) for x in v] for k, v in (tm.get("curves") or {}).items()}
    pace: Dict[str, Optional[float]] = {}
    if curves:
        # Real-race base pace was measured with the formula; re-measure it with the ML curves so the
        # two tyre models are compared like for like (synthetic base pace = new-soft pace, no change).
        probe = RaceState("probe", "probe", "", int(raw["total_laps"]), int(raw["completed_laps"]), 0.0, "dry",
                          None, None, None, None, 1.0, [], None, [], tyre_curves=curves,
                          progress_curve=[float(x) for x in tm.get("progress_curve") or []])
        for d in [raw["target"]] + list(raw["competitors"]):
            pace[str(d["code"]).strip().upper()] = rebased_pace(d, probe, cfg)
    pick = lambda d: pace.get(str(d["code"]).strip().upper())  # noqa: E731
    return RaceState(
        scenario_id=str(raw.get("scenario_id", "custom")),
        scenario_name=str(raw.get("scenario_name", "Custom")),
        track_name=str(raw.get("track_name", "Unknown circuit")),
        total_laps=int(raw["total_laps"]),
        completed_laps=int(raw["completed_laps"]),
        pit_loss_s=float(raw["pit_loss_s"]),
        track_condition=raw["track_condition"],
        rain_start_lap=opt("rain_start_lap"),
        rain_end_lap=opt("rain_end_lap"),
        sc_start_lap=opt("sc_start_lap"),
        sc_end_lap=opt("sc_end_lap"),
        deg_multiplier=float(raw["deg_multiplier"]),
        pit_closed_laps=sorted({int(x) for x in raw.get("pit_closed_laps", [])}),
        target=_driver_from(raw["target"], pick(raw["target"])),
        competitors=[_driver_from(c, pick(c)) for c in raw["competitors"]],
        data_source=str(raw.get("data_source", "synthetic")),
        data_source_label=str(raw.get("data_source_label", "Synthetic scenario")),
        provenance=dict(raw.get("provenance", {})),
        events=list(raw.get("events", [])),
        notes=list(raw.get("notes", [])),
        fuel_effect_s_per_lap=float(raw.get("fuel_effect_s_per_lap") or 0.0),
        neutral_periods=[{"kind": p["kind"], "start": int(p["start"]), "end": int(p["end"])}
                         for p in raw.get("neutral_periods") or []],
        wet_periods=[{"start": int(p["start"]), "end": int(p["end"])} for p in raw.get("wet_periods") or []],
        tyre_curves=curves,
        tyre_model_label=str(tm.get("label") or "ML tyre model") if curves else "Formula (pitwall/config.py)",
        progress_curve=[float(x) for x in tm.get("progress_curve") or []] if curves else [],
    )

"""Optional FastF1 historical-replay source.

Builds the common race-state dict from a past race using ONLY laps up to the selected
decision lap. Retrieved values, derived estimates and synthetic assumptions are labelled in
`provenance`. This is a historical replay, never a live feed; later weather is not used.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import COMPOUNDS, DEFAULT_CONFIG, ModelConfig

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fastf1_cache"


class DataSourceError(RuntimeError):
    pass


def fastf1_available() -> bool:
    try:
        import fastf1  # noqa: F401
        return True
    except Exception:
        return False


def _ff1():
    try:
        import fastf1
    except Exception as e:  # ImportError or broken install
        raise DataSourceError("FastF1 is not installed. Install with: pip install -r requirements-optional.txt") from e
    return fastf1


def enable_cache(cache_dir: Path = CACHE_DIR) -> None:
    ff1 = _ff1()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(cache_dir))


def get_race_schedule(year: int, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Completed race weekends for a season: RoundNumber, EventName, EventDate."""
    ff1 = _ff1()
    try:
        enable_cache(cache_dir)
        sched = ff1.get_event_schedule(int(year), include_testing=False)
        sched = sched[sched["RoundNumber"] > 0]
        sched = sched[pd.to_datetime(sched["EventDate"]) < pd.Timestamp(datetime.now())]
        return sched[["RoundNumber", "EventName", "EventDate"]].reset_index(drop=True)
    except DataSourceError:
        raise
    except Exception as e:
        raise DataSourceError(f"Could not load the {year} schedule ({type(e).__name__}: {e})") from e


def load_session(year: int, round_number: int, cache_dir: Path = CACHE_DIR):
    """Load only laps + weather for a race session (no telemetry, no messages)."""
    ff1 = _ff1()
    try:
        enable_cache(cache_dir)
        session = ff1.get_session(int(year), int(round_number), "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as e:
        raise DataSourceError(f"Could not load {year} round {round_number} race "
                              f"({type(e).__name__}: {e})") from e
    try:
        laps = session.laps          # FastF1 raises DataNotLoadedError if timing failed to load
    except Exception as e:
        raise DataSourceError(f"Lap timing for {year} round {round_number} could not be retrieved "
                              f"(network blocked/unavailable or no data): {type(e).__name__}") from e
    if laps is None or len(laps) == 0:
        raise DataSourceError(f"No lap timing available for {year} round {round_number}.")
    return session


def session_drivers(session) -> List[str]:
    return sorted(str(d) for d in session.laps["Driver"].dropna().unique())


def _secs(td) -> Optional[float]:
    if td is None or pd.isna(td):
        return None
    return float(pd.Timedelta(td).total_seconds())


def _model_tyre_adj(compound: str, age: float, cfg: ModelConfig) -> float:
    cp = cfg.compounds[compound]
    return cp.pace_offset_s + cp.deg_s_per_lap * age + cp.cliff_s_per_lap * max(0.0, age - cp.cliff_age_laps)


def driver_names(session) -> Dict[str, Dict[str, str]]:
    """code -> {name, team} from session results when available (FastF1 or replay)."""
    out: Dict[str, Dict[str, str]] = {}
    try:
        res = session.results
    except Exception:
        return out
    if res is None or not len(res) or "Abbreviation" not in res.columns:
        return out
    for _, r in res.iterrows():
        code = r.get("Abbreviation")
        if isinstance(code, str) and code:
            name = r.get("FullName")
            team = r.get("TeamName")
            out[code] = {"name": name if isinstance(name, str) and name else code,
                         "team": team if isinstance(team, str) else ""}
    return out


def estimate_pit_loss(upto: pd.DataFrame, decision_lap: int, cfg: ModelConfig):
    """Median green-flag pit-stop loss from stops completed by ANY car up to the decision lap.
    loss = in-lap + out-lap - 2 x the driver's median green lap nearby. Returns (value, n) or (None, 0)."""
    samples = []
    for code, d in upto.groupby("Driver"):
        d = d.sort_values("LapNumber").reset_index(drop=True)
        if "Stint" not in d.columns:
            continue
        for i in range(1, len(d)):
            if not (d.at[i, "Stint"] > d.at[i - 1, "Stint"]):
                continue
            inlap, outlap = d.iloc[i - 1], d.iloc[i]
            if outlap["LapNumber"] > decision_lap or inlap["LapNumber"] <= 1:
                continue
            if str(inlap.get("TrackStatus", "1")) != "1" or str(outlap.get("TrackStatus", "1")) != "1":
                continue
            if pd.isna(inlap["LapTime"]) or pd.isna(outlap["LapTime"]):
                continue
            near = d[(d["LapNumber"] >= inlap["LapNumber"] - 4) & (d["LapNumber"] <= outlap["LapNumber"] + 3)
                     & (d["LapNumber"] != inlap["LapNumber"]) & (d["LapNumber"] != outlap["LapNumber"])
                     & (d["LapNumber"] <= decision_lap) & d["LapTime"].notna()
                     & (d["TrackStatus"].astype(str) == "1")]
            if len(near) < 2:
                continue
            ref = float(np.median([_secs(t) for t in near["LapTime"]]))
            loss = _secs(inlap["LapTime"]) + _secs(outlap["LapTime"]) - 2 * ref
            lo, hi = cfg.pit_loss_sane_range_s
            if lo <= loss <= hi:
                samples.append(loss)
    if len(samples) < 2:
        return None, len(samples)
    return float(np.median(samples)), len(samples)


def _driver_entry(drv_laps: pd.DataFrame, row: pd.Series, gap: float, decision_lap: int,
                  cfg: ModelConfig, notes: List[str], fuel: float = 0.0,
                  names: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
    code = str(row["Driver"])
    raw_comp = row.get("Compound", "")
    compound = str(raw_comp).upper() if isinstance(raw_comp, str) else ""
    if compound not in COMPOUNDS and "Stint" in drv_laps.columns and pd.notna(row.get("Stint")):
        # Fill a missing compound from earlier laps of the same stint (same tyre set).
        same = drv_laps[(drv_laps["Stint"] == row.get("Stint")) & drv_laps["Compound"].notna()]
        known = [str(c).upper() for c in same["Compound"] if str(c).upper() in COMPOUNDS]
        if known:
            compound = known[-1]
            notes.append(f"{code}: tyre compound missing at lap {decision_lap}; taken from earlier laps of the same stint.")
    if compound not in COMPOUNDS:
        raise DataSourceError(f"{code}: tyre compound unknown at lap {decision_lap} ({compound or 'missing'})")
    age = row.get("TyreLife")
    if age is None or pd.isna(age):
        stint = row.get("Stint")
        age = int((drv_laps["Stint"] == stint).sum()) if stint is not None and not pd.isna(stint) else None
        if age is None:
            raise DataSourceError(f"{code}: tyre age missing at lap {decision_lap}")
        notes.append(f"{code}: tyre age derived from stint lap count")
    clean = drv_laps[(drv_laps["LapNumber"] > 1) & drv_laps["LapTime"].notna()
                     & drv_laps["PitInTime"].isna() & drv_laps["PitOutTime"].isna()]
    if "TrackStatus" in clean.columns:
        clean = clean[clean["TrackStatus"].astype(str) == "1"]
    recent = clean.tail(6)
    if len(recent) < 2:
        raise DataSourceError(f"{code}: too early in the race, fewer than 2 green-flag laps before lap "
                              f"{decision_lap} to measure pace. Pick a later lap")
    # Derived estimate: remove the model's tyre effect so the baseline is comparable.
    est, samples = [], []
    for _, r in recent.iterrows():
        c = str(r.get("Compound", compound)).upper()
        a = r.get("TyreLife")
        if c in COMPOUNDS and a is not None and not pd.isna(a):
            # slick laps assumed dry, INTERMEDIATE/WET laps assumed wet (approximation)
            wet_adj = 0.0 if cfg.compounds[c].is_dry else (
                cfg.track_condition_penalty_s["wet"] + cfg.tyre_weather_penalty_s["wet"][c])
            # normalise to the decision lap's fuel load (lighter car later = faster)
            fuel_adj = -fuel * (decision_lap - float(r["LapNumber"]))
            est.append(_secs(r["LapTime"]) - _model_tyre_adj(c, max(0.0, float(a) - 1), cfg) - wet_adj
                       - fuel_adj)
            samples.append({"lap": int(r["LapNumber"]), "compound": c, "age": max(0.0, float(a) - 1),
                            "time": round(_secs(r["LapTime"]) - fuel_adj, 3), "raw_time": round(_secs(r["LapTime"]), 3),
                            "wet_adj": round(wet_adj, 3)})
    baseline = float(np.median(est)) if est else float(np.median([_secs(t) for t in recent["LapTime"]]))
    used = [str(c).upper() for c in drv_laps["Compound"].dropna().unique() if str(c).upper() in COMPOUNDS]
    stint = row.get("Stint")
    sets = dict(cfg.default_available_sets)
    info = (names or {}).get(code, {})
    team = info.get("team") or (row.get("Team") if isinstance(row.get("Team"), str) else "")
    return {"code": code, "name": info.get("name", code), "team": team,
            "gap_to_leader_s": round(gap, 3), "baseline_pace_s": round(baseline, 3),
            "compound": compound, "tyre_age_laps": int(age),
            "compounds_used": used or [compound], "available_sets": sets, "pace_samples": samples,
            "stops_made": int(stint) - 1 if stint is not None and not pd.isna(stint) else 0}


def build_race_state(session, driver: str, decision_lap: int, cfg: ModelConfig = DEFAULT_CONFIG,
                     n_competitors: int = 4, label: str = "", source: str = "fastf1") -> Dict[str, Any]:
    """Common race-state dict from laps <= decision_lap only (works for FastF1 and replay sessions)."""
    try:
        laps = session.laps
        total = session.total_laps
    except Exception as e:
        raise DataSourceError(f"Session timing not loaded ({type(e).__name__}).") from e
    missing = {"Driver", "LapNumber", "LapTime", "Time", "Compound", "TyreLife", "Stint",
               "PitInTime", "PitOutTime"} - set(laps.columns)
    if missing:
        raise DataSourceError(f"Lap data is missing required fields: {sorted(missing)}")
    if total is None or pd.isna(total):
        raise DataSourceError("Scheduled race distance (total laps) unavailable for this session.")
    total, decision_lap = int(total), int(decision_lap)
    if not 1 <= decision_lap < total:
        raise DataSourceError(f"Decision lap must be between 1 and {total - 1}.")
    upto = laps[laps["LapNumber"] <= decision_lap]       # no future information beyond this point
    at = upto[(upto["LapNumber"] == decision_lap) & upto["Time"].notna()]
    if driver not in set(at["Driver"]):
        raise DataSourceError(f"{driver} has no timing for lap {decision_lap} (retired, lapped or missing data).")
    trow = at[at["Driver"] == driver].iloc[0]
    if "PitInTime" in at.columns and pd.notna(trow.get("PitInTime")):
        raise DataSourceError(f"{driver} pitted at the end of lap {decision_lap}, so the stop was already "
                              f"under way. Choose lap {decision_lap - 1} or {decision_lap + 1}.")
    fuel = cfg.real_fuel_effect_s_per_lap
    names = driver_names(session)
    ref = at["Time"].min()                               # leader's lap-end time = common reference
    at = at.assign(gap=(at["Time"] - ref).dt.total_seconds()).sort_values("gap")
    order = list(at["Driver"])
    i = order.index(driver)
    neighbours = sorted((c for c in order if c != driver),
                        key=lambda c: (abs(order.index(c) - i), order.index(c)))[:n_competitors]
    notes: List[str] = []
    entries = {}
    for code in [driver] + neighbours:
        row = at[at["Driver"] == code].iloc[0]
        try:
            entries[code] = _driver_entry(upto[upto["Driver"] == code].sort_values("LapNumber"),
                                          row, float(row["gap"]), decision_lap, cfg, notes, fuel, names)
        except DataSourceError as e:
            if code == driver:
                raise
            notes.append(f"Competitor skipped — {e}")
    target_row = at[at["Driver"] == driver].iloc[0]
    status = str(target_row.get("TrackStatus", "1") or "1")

    # Track condition: what the field is running at the decision lap (plus the rainfall flag
    # sampled at or before that moment, never later samples).
    comps = at["Compound"].dropna().astype(str).str.upper()
    wet_share = float(comps.isin(["INTERMEDIATE", "WET"]).mean()) if len(comps) else 0.0
    raining = False
    wd = getattr(session, "weather_data", None)
    if wd is not None and len(wd) and "Rainfall" in wd.columns:
        past = wd[wd["Time"] <= target_row["Time"]]
        raining = bool(past.iloc[-1]["Rainfall"]) if len(past) else False
    condition = "wet" if wet_share > 0.5 else "dry"
    weather_prov = (f"derived: {wet_share:.0%} of cars on intermediate/wet tyres at the decision lap"
                    + ("; rain reported" if raining else "; no rain reported")
                    + ("; wet track ASSUMED to last to the finish (no forecast used)" if condition == "wet" else ""))
    if raining and condition == "dry":
        notes.append("Light rain was reported at the decision point, but most cars were still on slicks.")
    pit_loss, n_stops = estimate_pit_loss(upto, decision_lap, cfg)
    if pit_loss is None:
        pit_loss = cfg.default_pit_loss_s
        pit_prov = (f"synthetic assumption ({cfg.default_pit_loss_s:.0f} s): fewer than 2 clean green-flag "
                    "stops before the decision lap to measure from")
    else:
        pit_prov = f"derived from {n_stops} real green-flag pit stops before the decision lap (median)"
    sc_start = sc_end = None
    if "4" in status:
        sc_start, sc_end = decision_lap + 1, min(total, decision_lap + cfg.fastf1_assumed_sc_laps)
        notes.append(f"Safety Car active at lap {decision_lap}; it is assumed to last until lap {sc_end} "
                     "(end time not known at the decision point).")
    if any(s in status for s in ("6", "7")):
        notes.append("Virtual Safety Car active at the decision point (not modelled).")
    event_name = str(getattr(session, "event", {}).get("EventName", "Race")) \
        if hasattr(getattr(session, "event", None), "get") else "Race"
    year = getattr(getattr(session, "date", None), "year", "")
    who = entries[driver]["name"]
    src_label = "FastF1" if source == "fastf1" else "Real race replay"
    return {
        "scenario_id": f"{source}_{year}_{event_name.replace(' ', '_')}_{driver}_L{decision_lap}",
        "scenario_name": f"{event_name} {year}: {who} after lap {decision_lap}",
        "track_name": event_name, "total_laps": total, "completed_laps": decision_lap,
        "pit_loss_s": round(pit_loss, 2), "track_condition": condition,
        "fuel_effect_s_per_lap": fuel,
        "rain_start_lap": None, "rain_end_lap": None, "sc_start_lap": sc_start, "sc_end_lap": sc_end,
        "deg_multiplier": 1.0, "pit_closed_laps": [],
        "target": entries[driver], "competitors": [entries[c] for c in neighbours if c in entries],
        "data_source": source,
        "data_source_label": label or f"{src_label}: {event_name} {year}, after lap {decision_lap} "
                                      "(historical data, not a live feed)",
        "replay": {"source": source, "year": year, "event": event_name, "driver": driver,
                   "decision_lap": decision_lap},
        "provenance": {
            "driver names": "retrieved" if names else "not available (codes shown)",
            "total_laps": "retrieved (race distance)",
            "completed_laps": "user-selected decision point",
            "gap_to_leader_s": "derived from retrieved lap-end times at the decision lap",
            "compound / tyre_age_laps / compounds_used / stops_made": "retrieved (laps up to decision point)",
            "baseline_pace_s": "derived estimate (median of recent clean laps minus model tyre effect)",
            "track_condition": weather_prov,
            "safety_car": "retrieved track status; duration is a synthetic assumption" if sc_start
            else "retrieved track status (no SC at decision point)",
            "pit_loss_s": pit_prov,
            "fuel_effect_s_per_lap": f"synthetic assumption ({fuel:.2f} s faster per lap as fuel burns)",
            "available_sets": "synthetic assumption (config default)",
            "deg_multiplier": "synthetic assumption (1.0)",
            "rain forecast": "none — only user-injected simulated forecasts are used",
        },
        "events": [], "notes": notes,
    }


def load_race_state_with_fallback(loader, fallback_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Run `loader()` (returns a race-state dict). On any DataSourceError, return the
    synthetic fallback clearly relabelled, plus the error message."""
    try:
        return {"raw": loader(), "used_fallback": False, "error": None}
    except DataSourceError as e:
        fb = dict(fallback_raw)
        fb["data_source"] = "synthetic_fallback"
        fb["data_source_label"] = f"Synthetic fallback — FastF1 unavailable ({e})"
        return {"raw": fb, "used_fallback": True, "error": str(e)}

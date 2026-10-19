"""Real-race replays without FastF1.

Source data: FastF1-derived race timing published by TracingInsights on GitHub (MIT licence,
https://github.com/TracingInsights). Every race in data/replays/index.json ships inside the
project as a compact JSON file, so replays work offline. scripts/build_replays.py rebuilds them.

`ReplaySession` mimics the small part of a FastF1 `Session` that PitWall uses (`laps`,
`total_laps`, `weather_data`, `event`, `date`, `results`), so the same race-state builder and
the same "what really happened" code work for both sources.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .fastf1_source import DataSourceError

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "replays"
RACES_DIR = ROOT / "races"
INDEX_PATH = ROOT / "index.json"
FORMAT = "pitwall-replay-1"
SOURCE_TEXT = ("Real timing data (FastF1-derived) from the TracingInsights archive on GitHub, MIT licence. "
               "Historical replay, not a live feed.")

# 2018 used named compounds; per race they are mapped to soft/medium/hard roles by softness.
SOFTNESS_2018 = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD"]
DRY = ("SOFT", "MEDIUM", "HARD")
WET = ("INTERMEDIATE", "WET")


# --------------------------------------------------------------------------- conversion
def _clean(v):
    return None if v is None or v == "None" or (isinstance(v, float) and v != v) else v


def _num(v, nd=3):
    v = _clean(v)
    if v is None:
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def compound_map(raw_compounds: List[Any]) -> Dict[str, str]:
    """Map the dry compound names actually used in a race onto SOFT/MEDIUM/HARD roles."""
    seen = {str(c).upper() for c in raw_compounds if _clean(c)} - {"UNKNOWN", "TEST_UNKNOWN", "NONE"}
    dry = [c for c in SOFTNESS_2018 if c in seen]
    if not (set(dry) - set(DRY)):  # already soft/medium/hard naming
        return {c: c for c in seen}
    roles = DRY if len(dry) >= 3 else (("SOFT", "MEDIUM") if len(dry) == 2 else ("SOFT",))
    # softest -> SOFT, hardest -> HARD; extra middle names collapse onto MEDIUM
    mapping = {c: c for c in seen if c in WET}
    for i, name in enumerate(dry):
        if i == 0:
            mapping[name] = roles[0]
        elif i == len(dry) - 1:
            mapping[name] = roles[-1]
        else:
            mapping[name] = "MEDIUM"
    return mapping


def convert_raw(year: int, event: str, laptimes: Dict[str, Any], drivers: Dict[str, Any]) -> Dict[str, Any]:
    """Raw TracingInsights files -> compact PitWall replay snapshot (pure data, no code)."""
    need = ["time", "lap", "compound", "stint", "life", "pos", "status", "sesT", "drv", "pin", "pout"]
    missing = [k for k in need if k not in laptimes]
    if missing:
        raise DataSourceError(f"{year} {event}: lap file is missing fields {missing}")
    n = len(laptimes["lap"])
    cmap = compound_map(laptimes["compound"])
    rain = laptimes.get("wR", [None] * n)
    wtime = laptimes.get("wT", [None] * n)
    rows = {k: [] for k in ("drv", "lap", "time", "sesT", "compound", "life", "stint", "pos", "status",
                            "pin", "pout", "rain", "wT")}
    for i in range(n):
        lap = _num(laptimes["lap"][i], 0)
        drv = _clean(laptimes["drv"][i])
        if lap is None or not drv:
            continue
        comp = _clean(laptimes["compound"][i])
        comp = cmap.get(str(comp).upper()) if comp else None
        rows["drv"].append(str(drv))
        rows["lap"].append(int(lap))
        rows["time"].append(_num(laptimes["time"][i]))
        rows["sesT"].append(_num(laptimes["sesT"][i]))
        rows["compound"].append(comp)
        rows["life"].append(_num(laptimes["life"][i], 0))
        rows["stint"].append(_num(laptimes["stint"][i], 0))
        rows["pos"].append(_num(laptimes["pos"][i], 0))
        st = _clean(laptimes["status"][i])
        rows["status"].append(str(st) if st is not None else None)
        rows["pin"].append(_num(laptimes["pin"][i]))
        rows["pout"].append(_num(laptimes["pout"][i]))
        r = _clean(rain[i])
        rows["rain"].append(bool(r) if r is not None else None)
        rows["wT"].append(_num(wtime[i]))
    if len(rows["lap"]) < 200:
        raise DataSourceError(f"{year} {event}: only {len(rows['lap'])} laps of timing data")
    drv_list = []
    for d in drivers.get("drivers", []):
        code = _clean(d.get("driver"))
        if code:
            name = " ".join(x for x in (_clean(d.get("fn")), _clean(d.get("ln"))) if x) or code
            drv_list.append({"code": code, "name": name, "team": _clean(d.get("team")) or "",
                             "number": str(_clean(d.get("dn")) or "")})
    known = {d["code"] for d in drv_list}
    for code in sorted(set(rows["drv"]) - known):
        drv_list.append({"code": code, "name": code, "team": "", "number": ""})
    date = None
    lsd = laptimes.get("lSD") or []
    for v in lsd:
        if _clean(v):
            date = str(v)[:10]
            break
    return {"format": FORMAT,
            "meta": {"year": int(year), "event": event, "date": date,
                     "laps_winner": int(max(rows["lap"])),
                     "compound_map": {k: v for k, v in cmap.items() if k != v},
                     "source": SOURCE_TEXT},
            "drivers": drv_list, "laps": rows}


# --------------------------------------------------------------------------- session wrapper
def _td(seconds):
    return pd.NaT if seconds is None else pd.Timedelta(seconds=float(seconds))


class ReplaySession:
    """FastF1-shaped view over a replay snapshot."""

    def __init__(self, snap: Dict[str, Any]):
        if snap.get("format") != FORMAT:
            raise DataSourceError("Unsupported replay file format.")
        self.snapshot = snap
        m = snap["meta"]
        L = snap["laps"]
        team = {d["code"]: d["team"] for d in snap["drivers"]}
        self.laps = pd.DataFrame({
            "Driver": L["drv"], "Team": [team.get(c, "") for c in L["drv"]],
            "LapNumber": pd.Series(L["lap"], dtype="float64"),
            "LapTime": [_td(x) for x in L["time"]], "Time": [_td(x) for x in L["sesT"]],
            "Compound": L["compound"], "TyreLife": pd.Series(L["life"], dtype="float64"),
            "Stint": pd.Series(L["stint"], dtype="float64"), "Position": pd.Series(L["pos"], dtype="float64"),
            "TrackStatus": [s if s is not None else "1" for s in L["status"]],
            "PitInTime": [_td(x) for x in L["pin"]], "PitOutTime": [_td(x) for x in L["pout"]],
        })
        w = pd.DataFrame({"Time": [_td(x) for x in L["wT"]], "Rainfall": L["rain"]})
        w = w.dropna(subset=["Time"]).drop_duplicates(subset=["Time"]).sort_values("Time")
        w["Rainfall"] = w["Rainfall"].fillna(False).astype(bool)
        self.weather_data = w.reset_index(drop=True)
        self.total_laps = int(m["laps_winner"])
        self.event = {"EventName": m["event"]}
        self.date = pd.Timestamp(m["date"]) if m.get("date") else pd.Timestamp(year=m["year"], month=1, day=1)
        self.results = pd.DataFrame({"Abbreviation": [d["code"] for d in snap["drivers"]],
                                     "FullName": [d["name"] for d in snap["drivers"]],
                                     "TeamName": [d["team"] for d in snap["drivers"]]})
        self.source_label = (f"Real race replay: {m['event']} {m['year']} (historical timing data, not live)")
        self.compound_note = (f"2018 tyre names mapped to soft/medium/hard roles: {m['compound_map']}"
                              if m.get("compound_map") else "")


# --------------------------------------------------------------------------- index / loading
def race_id(year: int, event: str) -> str:
    slug = event.lower().replace("ã", "a").replace("é", "e").replace("ó", "o")
    return f"{int(year)}_" + "".join(ch if ch.isalnum() else "_" for ch in slug).strip("_")


def load_index(path: Path = INDEX_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise DataSourceError("The real-race index (data/replays/index.json) is missing.")
    return json.loads(path.read_text(encoding="utf-8"))


def seasons(index: Optional[Dict[str, Any]] = None) -> List[int]:
    index = index or load_index()
    return sorted({r["year"] for r in index["races"]}, reverse=True)


def races_for(year: int, index: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    index = index or load_index()
    return sorted((r for r in index["races"] if r["year"] == int(year)),
                  key=lambda r: (r.get("date") or "", r["event"]))


def load_replay(rid: str, races_dir: Path = RACES_DIR) -> ReplaySession:
    """Load a bundled race by id (e.g. '2023_singapore_grand_prix'). Raises DataSourceError."""
    path = races_dir / f"{rid}.json"
    if not path.exists():
        raise DataSourceError(f"Race file {path.name} isn't in data/replays/races.")
    try:
        return ReplaySession(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError) as e:
        raise DataSourceError(f"Race file {path.name} is damaged ({type(e).__name__}).") from e


def driver_options(session) -> List[Dict[str, Any]]:
    """Drivers with lap data, labelled with real names, in finishing-order of the timing data."""
    laps = session.laps
    names = {}
    res = getattr(session, "results", None)
    if res is not None and len(res):
        for _, r in res.iterrows():
            names[str(r.get("Abbreviation"))] = (str(r.get("FullName") or r.get("Abbreviation")),
                                                 str(r.get("TeamName") or ""))
    done = laps[laps["Time"].notna() & (laps["LapTime"].notna() | laps["Position"].notna())]
    agg = done.groupby("Driver").agg(n=("LapNumber", "max"), t=("Time", "max")).reset_index()
    agg = agg.sort_values(["n", "t"], ascending=[False, True])
    out = []
    for _, r in agg.iterrows():
        nm, tm = names.get(r["Driver"], (r["Driver"], ""))
        out.append({"code": r["Driver"], "name": nm, "team": tm, "laps": int(r["n"])})
    return out

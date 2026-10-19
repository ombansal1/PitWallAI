"""Common internal data structures shared by data sources, agents and the simulator."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DriverState:
    code: str
    name: str
    team: str
    gap_to_leader_s: float          # common time reference: leader at decision point = 0 s
    baseline_pace_s: float          # representative green-flag lap on a fresh SOFT, dry track
    compound: str
    tyre_age_laps: int              # laps completed on the current set
    compounds_used: List[str]
    available_sets: Dict[str, int]  # unused/usable sets remaining per compound
    stops_made: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RaceState:
    scenario_id: str
    scenario_name: str
    track_name: str
    total_laps: int
    completed_laps: int
    pit_loss_s: float
    track_condition: str                 # 'dry' | 'wet' at the decision point
    rain_start_lap: Optional[int]        # simulated forecast: first wet lap
    rain_end_lap: Optional[int]          # last wet lap (None = until the end)
    sc_start_lap: Optional[int]          # first Safety Car lap
    sc_end_lap: Optional[int]            # last Safety Car lap
    deg_multiplier: float
    pit_closed_laps: List[int]
    target: DriverState
    competitors: List[DriverState]
    data_source: str = "synthetic"
    data_source_label: str = "Synthetic scenario"
    provenance: Dict[str, str] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    fuel_effect_s_per_lap: float = 0.0   # lap time gained per lap as fuel burns (0 = not modelled)
    # Extra periods beyond the single SC / rain window above (chat scenarios, real-race events):
    neutral_periods: List[Dict[str, Any]] = field(default_factory=list)   # {"kind": "SC"|"VSC", "start", "end"}
    wet_periods: List[Dict[str, Any]] = field(default_factory=list)       # {"start", "end"}
    # ML tyre model (optional): seconds lost vs a new SOFT for each tyre age, per dry compound.
    # Empty = the hand-written formula in config.py.
    tyre_curves: Dict[str, List[float]] = field(default_factory=dict)
    tyre_model_label: str = "Formula (pitwall/config.py)"
    # ML mode: learned race-progress effect (fuel + track), seconds at progress 0, 0.05 … 1. Replaces the
    # fixed fuel_effect_s_per_lap when present.
    progress_curve: List[float] = field(default_factory=list)

    def progress_effect(self, lap: int) -> float:
        """Seconds gained on `lap` compared with the decision lap as the race goes on (negative = faster)."""
        if self.progress_curve:
            return _interp(self.progress_curve, lap / self.total_laps) - _interp(
                self.progress_curve, self.completed_laps / self.total_laps)
        return -self.fuel_effect_s_per_lap * (lap - self.completed_laps)

    @property
    def remaining_laps(self) -> int:
        return self.total_laps - self.completed_laps

    @property
    def first_lap(self) -> int:
        return self.completed_laps + 1

    def laps(self) -> range:
        return range(self.completed_laps + 1, self.total_laps + 1)

    def condition_on_lap(self, lap: int) -> str:
        if any(p["start"] <= lap <= p["end"] for p in self.wet_periods):
            return "wet"
        if self.rain_start_lap is not None:
            end = self.rain_end_lap if self.rain_end_lap is not None else self.total_laps
            if self.rain_start_lap <= lap <= end:
                return "wet"
            if self.track_condition == "wet" and lap < self.rain_start_lap:
                return "wet"
            return "dry"
        if self.track_condition == "wet":
            if self.rain_end_lap is not None and lap > self.rain_end_lap:
                return "dry"
            return "wet"
        return "dry"

    def all_neutral(self) -> List[Dict[str, Any]]:
        """Every Safety Car / VSC period (the main SC window plus any extra periods), by start lap."""
        out = []
        if self.sc_start_lap is not None and self.sc_end_lap is not None:
            out.append({"kind": "SC", "start": self.sc_start_lap, "end": self.sc_end_lap})
        out += [dict(p) for p in self.neutral_periods]
        return sorted(out, key=lambda p: (p["start"], p["kind"]))

    def neutral_kind(self, lap: int) -> Optional[str]:
        """'SC', 'VSC' or None. A full Safety Car wins if periods overlap."""
        kinds = {p["kind"] for p in self.all_neutral() if p["start"] <= lap <= p["end"]}
        return "SC" if "SC" in kinds else ("VSC" if kinds else None)

    def sc_on_lap(self, lap: int) -> bool:
        """True when the race is neutralised (Safety Car or VSC) on this lap."""
        return self.neutral_kind(lap) is not None

    def neutral_ahead(self) -> List[Dict[str, Any]]:
        return [p for p in self.all_neutral() if p["end"] >= self.first_lap]

    def rain_arrival(self) -> Optional[Dict[str, int]]:
        """The first wet spell that starts after the next lap (rain still to come), if any."""
        return next((r for r in self.wet_runs() if r["start"] > self.first_lap), None)

    def dries_after(self) -> Optional[int]:
        """If the track is wet on the next lap and dries before the finish: the last wet lap."""
        runs = self.wet_runs()
        if runs and runs[0]["start"] == self.first_lap and runs[0]["end"] < self.total_laps:
            return runs[0]["end"]
        return None

    def wet_runs(self) -> List[Dict[str, int]]:
        """Consecutive wet laps over the remaining race, e.g. [{'start': 30, 'end': 41}]."""
        runs: List[Dict[str, int]] = []
        for lap in self.laps():
            if self.condition_on_lap(lap) == "wet":
                if runs and runs[-1]["end"] == lap - 1:
                    runs[-1]["end"] = lap
                else:
                    runs.append({"start": lap, "end": lap})
        return runs

    def all_drivers(self) -> List[DriverState]:
        return [self.target] + list(self.competitors)

    def driver(self, code: str) -> DriverState:
        for d in self.all_drivers():
            if d.code == code:
                return d
        raise KeyError(code)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self) -> "RaceState":
        return copy.deepcopy(self)


@dataclass
class PitStop:
    lap: int        # pit at the END of this lap
    compound: str   # compound fitted for lap + 1 onwards


@dataclass
class Strategy:
    id: str
    label: str
    family: str                 # pit_now | delay | stay_out | event_aligned | two_stop | earliest_open
    stops: List[PitStop]
    rationale: str = ""
    origin: str = "plan_v1"     # which plan version introduced it

    @property
    def stop_key(self) -> Tuple[Tuple[int, str], ...]:
        return tuple((s.lap, s.compound) for s in self.stops)

    @property
    def first_pit_lap(self) -> Optional[int]:
        return self.stops[0].lap if self.stops else None

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label, "family": self.family,
                "stops": [asdict(s) for s in self.stops], "rationale": self.rationale,
                "origin": self.origin}


@dataclass
class SimulationResult:
    strategy_id: str
    driver_code: str
    laps: List[Dict[str, Any]]          # per-lap breakdown
    lap_time_total_s: float
    pit_loss_total_s: float
    pit_loss_count: int
    remaining_time_s: float             # sum(lap times) + sum(pit losses)
    finish_time_ref_s: float            # gap_to_leader + remaining_time (common reference)
    stints: List[Dict[str, Any]]
    traffic_laps: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "lap_time_total_s": round(self.lap_time_total_s, 3),
            "pit_loss_total_s": round(self.pit_loss_total_s, 3),
            "pit_loss_count": self.pit_loss_count,
            "remaining_time_s": round(self.remaining_time_s, 3),
            "finish_time_ref_s": round(self.finish_time_ref_s, 3),
            "traffic_laps": self.traffic_laps,
            "stints": self.stints,
        }


def _interp(curve: List[float], x: float) -> float:
    """Linear interpolation on an evenly spaced 0..1 grid."""
    x = min(1.0, max(0.0, x)) * (len(curve) - 1)
    i = min(int(x), len(curve) - 2)
    return curve[i] + (curve[i + 1] - curve[i]) * (x - i)

"""Model parameters and assumptions for the PitWall AI simulator.

Units: all times are seconds (s); all lap counts/ages are integer laps.

Completed-laps convention
-------------------------
* ``completed_laps`` = laps fully completed by the target car at the decision point.
* Remaining laps = ``total_laps - completed_laps``; the simulator models laps
  ``completed_laps + 1 .. total_laps``.
* A stop "on lap P" means the car pits at the END of lap P: lap P is the in-lap on
  the old tyres, the pit-stop loss is added once at lap P and the new set is used
  from lap P + 1 with tyre age 0. "Pit now" (box this lap) = P = completed_laps + 1.
* Valid stop laps: completed_laps + 1 .. total_laps - 1 (at least one lap on new tyres).
* ``tyre_age_laps`` = laps already completed on the current set. The age used for a
  lap is the age at the START of that lap.

Lap-time model
--------------
lap_time = baseline_pace + tyre_adj + traffic_adj + weather_adj + fuel_adj + safety_car_adj
  fuel_adj   = -fuel_effect_s_per_lap * (lap - completed_laps)   (0 for synthetic scenarios;
               real-race replays use an assumed 0.06 s/lap because cars get lighter)
  tyre_adj   = compound_offset + deg_rate * deg_multiplier * age
               + cliff_rate * deg_multiplier * max(0, age - cliff_age)
  weather_adj = track_condition_penalty[condition] + tyre_weather_penalty[condition][compound]
  safety_car_adj = max(0, baseline * SC_LAP_TIME_FACTOR - green_lap_time) on SC laps only
remaining_time = sum(lap_times) + sum(pit_losses)   (each stop's loss counted once)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class CompoundParams:
    name: str
    is_dry: bool
    pace_offset_s: float      # pace vs a fresh SOFT on a dry track (wet effects are in weather tables)
    deg_s_per_lap: float      # linear wear per lap of tyre age
    cliff_age_laps: int       # age after which additional wear applies
    cliff_s_per_lap: float    # additional wear per lap beyond the cliff
    max_stint_laps: int       # safe tyre life at deg_multiplier == 1.0


COMPOUNDS: Dict[str, CompoundParams] = {
    "SOFT": CompoundParams("SOFT", True, 0.00, 0.085, 16, 0.25, 26),
    "MEDIUM": CompoundParams("MEDIUM", True, 0.45, 0.055, 26, 0.20, 38),
    "HARD": CompoundParams("HARD", True, 0.85, 0.035, 36, 0.15, 50),
    "INTERMEDIATE": CompoundParams("INTERMEDIATE", False, 0.00, 0.060, 24, 0.20, 35),
    "WET": CompoundParams("WET", False, 0.00, 0.050, 28, 0.20, 40),
}
DRY_COMPOUNDS = tuple(c for c, p in COMPOUNDS.items() if p.is_dry)
WET_COMPOUNDS = tuple(c for c, p in COMPOUNDS.items() if not p.is_dry)

# Additional lap time for the whole field when the track is wet.
TRACK_CONDITION_PENALTY_S = {"dry": 0.0, "wet": 4.0}

# Tyre suitability: extra lap time for each compound by track condition.
TYRE_WEATHER_PENALTY_S = {
    "dry": {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0, "INTERMEDIATE": 4.5, "WET": 7.0},
    "wet": {"SOFT": 10.0, "MEDIUM": 10.5, "HARD": 11.0, "INTERMEDIATE": 0.0, "WET": 1.2},
}


@dataclass(frozen=True)
class ModelConfig:
    compounds: Dict[str, CompoundParams] = field(default_factory=lambda: dict(COMPOUNDS))
    track_condition_penalty_s: Dict[str, float] = field(
        default_factory=lambda: dict(TRACK_CONDITION_PENALTY_S))
    tyre_weather_penalty_s: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in TYRE_WEATHER_PENALTY_S.items()})

    # Safety Car: everyone laps at baseline * factor; a stop under SC costs pit_loss * factor.
    # Simplification: tyre age still increases by one lap per SC lap (no reduced wear under SC).
    sc_lap_time_factor: float = 1.40
    sc_pit_loss_factor: float = 0.50
    # Virtual Safety Car: cars hold a delta (~30-35% slower); a stop saves a bit less than under a full SC.
    vsc_lap_time_factor: float = 1.30
    vsc_pit_loss_factor: float = 0.60

    # Traffic: rejoining 0 < gap <= threshold behind another simulated car costs
    # `traffic_penalty_s` on each of the next `traffic_laps` green-flag laps.
    traffic_enabled: bool = True
    traffic_gap_threshold_s: float = 1.5
    traffic_penalty_s: float = 0.6
    traffic_laps: int = 2

    # Strategy constraints (checked by the Strategy Critic).
    max_additional_stops: int = 2
    require_two_dry_compounds: bool = True   # waived once INTERMEDIATE/WET is used
    max_slick_laps_on_wet: int = 2           # more than this = unsafe / infeasible
    short_delay_laps: int = 3                # "pit after a short delay" = box 3 laps later

    # Orchestration.
    max_revisions: int = 2
    # AI mode: Pit Wall Operations may pick any Critic-approved option within this many seconds of the
    # fastest one (shortlist of at most ai_shortlist_max), and must say why. Outside the window = not allowed.
    ai_choice_window_s: float = 2.0
    ai_shortlist_max: int = 5

    # Numerical integrity bounds (fraction of baseline pace).
    min_lap_factor: float = 0.80
    max_lap_factor: float = 2.50

    # Explicit risk rule (no invented probabilities):
    #   HIGH   if margin to next-best feasible option < risk_margin_high_s
    #          or the final stint ends above tyre_life_risk_fraction of its safe life
    #   MEDIUM if margin < risk_margin_low_s, it is the only feasible option,
    #          or the result depends on a forecast / Safety Car timing
    #   LOW    otherwise
    risk_margin_high_s: float = 1.5
    risk_margin_low_s: float = 5.0
    tyre_life_risk_fraction: float = 0.90

    # Defaults used when a data source cannot provide a value (labelled as synthetic).
    default_pit_loss_s: float = 22.0
    default_available_sets: Dict[str, int] = field(default_factory=lambda: {
        "SOFT": 1, "MEDIUM": 1, "HARD": 1, "INTERMEDIATE": 2, "WET": 1})
    fastf1_assumed_sc_laps: int = 3
    # Real-race replays only: cars get lighter as fuel burns. Typical value ~0.06 s per lap
    # (synthetic assumption, labelled in provenance). Synthetic scenarios use 0.
    real_fuel_effect_s_per_lap: float = 0.06
    pit_loss_sane_range_s: tuple = (12.0, 40.0)


DEFAULT_CONFIG = ModelConfig()

RISK_RULE_TEXT = (
    "HIGH if margin to next-best feasible option < {h:.1f} s or final stint ends above "
    "{f:.0%} of safe tyre life; MEDIUM if margin < {l:.1f} s, it is the only feasible "
    "option, or it depends on a forecast/event timing; otherwise LOW."
)

# Input validation bounds.
VALIDATION_BOUNDS = {
    "total_laps": (5, 100),
    "completed_laps": (0, 99),
    "pit_loss_s": (5.0, 60.0),
    "baseline_pace_s": (40.0, 200.0),
    "gap_to_leader_s": (0.0, 3600.0),   # lapped cars and red flags can create large gaps
    "tyre_age_laps": (0, 80),
    "deg_multiplier": (0.5, 3.0),
    "available_sets": (0, 6),
}

QUESTIONS = {
    "full_strategy": "What is the best strategy for the rest of the race?",
    "pit_now_vs_stay_out": "Should we pit now or stay out?",
    "compound_choice": "If we box now, which compound should we fit?",
    "event_response": "How should we respond to the current race event?",
}

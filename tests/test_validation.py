import copy
import math

import pytest

from pitwall.orchestrator import run_analysis
from pitwall.records import DecisionStore
from pitwall.validation import InputValidationError, parse_race_state, validate_race_input


def fields(issues):
    return {i["field"] for i in issues}


def test_valid_input_has_no_issues(simple_raw):
    assert validate_race_input(simple_raw) == []


@pytest.mark.parametrize("key", ["total_laps", "completed_laps", "pit_loss_s", "target", "competitors"])
def test_missing_fields(simple_raw, key):
    del simple_raw[key]
    assert key in fields(validate_race_input(simple_raw))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values(simple_raw, value):
    simple_raw["pit_loss_s"] = value
    simple_raw["target"]["baseline_pace_s"] = value
    f = fields(validate_race_input(simple_raw))
    assert {"pit_loss_s", "target.baseline_pace_s"} <= f


@pytest.mark.parametrize("key,value", [("total_laps", 2), ("total_laps", 500), ("pit_loss_s", -1.0),
                                       ("deg_multiplier", 9.0), ("completed_laps", -1),
                                       ("total_laps", 10.5), ("pit_loss_s", "fast"), ("total_laps", True)])
def test_out_of_range_or_wrong_type(simple_raw, key, value):
    simple_raw[key] = value
    assert key in fields(validate_race_input(simple_raw))


def test_completed_laps_must_leave_a_lap(simple_raw):
    simple_raw["completed_laps"] = 10
    assert "completed_laps" in fields(validate_race_input(simple_raw))


def test_driver_level_errors(simple_raw):
    simple_raw["target"]["compound"] = "SUPERSOFT"
    simple_raw["target"]["tyre_age_laps"] = -3
    del simple_raw["target"]["available_sets"]
    simple_raw["competitors"][0]["gap_to_leader_s"] = None
    simple_raw["competitors"][1]["code"] = "AAA"
    f = fields(validate_race_input(simple_raw))
    assert {"target.compound", "target.tyre_age_laps", "target.available_sets",
            "competitors[0].gap_to_leader_s", "competitors"} <= f


def test_event_windows_must_be_inside_remaining_race(simple_raw):
    simple_raw["sc_start_lap"], simple_raw["sc_end_lap"] = 3, 4      # already completed laps
    simple_raw["rain_start_lap"] = 12                                 # after the finish
    f = fields(validate_race_input(simple_raw))
    assert {"sc_start_lap", "sc_end_lap", "rain_start_lap"} <= f
    simple_raw2 = dict(simple_raw, sc_start_lap=7, sc_end_lap=None, rain_start_lap=None)
    assert "sc_end_lap" in fields(validate_race_input(simple_raw2))


def test_parse_raises_and_never_mutates(simple_raw):
    bad = copy.deepcopy(simple_raw)
    bad["completed_laps"] = 99
    snapshot = copy.deepcopy(bad)
    with pytest.raises(InputValidationError):
        parse_race_state(bad)
    assert bad == snapshot


def test_invalid_input_stops_workflow_without_record(simple_raw, isolated_decision_log):
    simple_raw["pit_loss_s"] = math.nan
    snapshot = copy.deepcopy(simple_raw)
    r = run_analysis(simple_raw)
    assert r["status"] == "invalid_input" and r["recommendation"] is None
    assert any(i["field"] == "pit_loss_s" for i in r["input_issues"])
    assert not r["candidates"] and r["plan"] is None
    assert simple_raw["target"] == snapshot["target"]  # inputs untouched
    assert DecisionStore().load_all() == []


def test_unknown_question_is_input_error(simple_raw):
    r = run_analysis(simple_raw, "what_is_the_meaning_of_life")
    assert r["status"] == "invalid_input" and "question" in fields(r["input_issues"])

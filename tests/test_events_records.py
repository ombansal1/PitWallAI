import copy
import json

import pytest

from pitwall.comparison import compare_results
from pitwall.events import EventError, apply_event
from pitwall.orchestrator import run_analysis
from pitwall.records import DecisionStore, build_decision_record


# ----------------------------------------------------------------- events
def test_safety_car_event_updates_copy_only(scenario):
    raw = scenario("dry_normal")
    snap = copy.deepcopy(raw)
    new, rec = apply_event(raw, {"type": "safety_car", "laps_until": 0, "duration_laps": 4})
    assert raw == snap
    assert (new["sc_start_lap"], new["sc_end_lap"]) == (21, 24)
    assert new["completed_laps"] == raw["completed_laps"]  # same decision point
    assert new["events"][-1]["description"] == rec["description"]


def test_rain_and_degradation_events(scenario):
    raw = scenario("dry_normal")
    new, _ = apply_event(raw, {"type": "rain", "laps_until": 5, "duration_laps": 0})
    assert new["rain_start_lap"] == 25 and new["rain_end_lap"] is None
    new2, _ = apply_event(raw, {"type": "degradation", "factor": 1.5})
    assert new2["deg_multiplier"] == pytest.approx(1.5)


@pytest.mark.parametrize("ev", [{"type": "rain", "laps_until": 99}, {"type": "safety_car", "laps_until": 80},
                                {"type": "degradation", "factor": 0.5}, {"type": "meteor"},
                                {"type": "safety_car", "duration_laps": 0}])
def test_invalid_events_rejected(scenario, ev):
    with pytest.raises(EventError):
        apply_event(scenario("dry_normal"), ev)


def test_event_triggers_recalculation_and_comparison(scenario):
    raw = scenario("dry_normal")
    before = run_analysis(raw)
    new, rec = apply_event(raw, {"type": "rain", "laps_until": 6, "duration_laps": 0})
    after = run_analysis(new, trigger_event=rec)
    assert after["trigger_event"]["type"] == "rain"
    assert after["recommendation"]["stops"][-1]["compound"] in ("INTERMEDIATE", "WET")
    cmp = compare_results(before, after)
    assert cmp["changed"] and cmp["same_decision_point"]
    assert cmp["previous_choice_now_feasible"] is False  # slicks into a wet race now break a constraint
    assert "infeasible" in cmp["reason"]


def test_safety_car_event_changes_numbers(scenario):
    raw = scenario("dry_normal")
    before = run_analysis(raw)
    new, rec = apply_event(raw, {"type": "safety_car", "laps_until": 0, "duration_laps": 4})
    after = run_analysis(new, trigger_event=rec)
    assert after["recommendation"]["numbers"]["pit_loss_total_s"] == pytest.approx(21.5 * 0.5)
    cmp = compare_results(before, after)
    assert cmp["reason"]  # a numerical reason is always produced
    assert isinstance(cmp["previous_choice_now_s"], float)


def test_unchanged_recommendation_is_allowed(scenario):
    raw = scenario("dry_normal")
    a, b = run_analysis(raw), run_analysis(raw)
    cmp = compare_results(a, b)
    assert cmp["changed"] is False and cmp["reason"].startswith("Unchanged")


# ----------------------------------------------------------------- records
def test_record_contents_and_idempotent_append(scenario, isolated_decision_log):
    r = run_analysis(scenario("safety_car"))
    rec = build_decision_record(r)
    for key in ("analysis_id", "timestamp_utc", "input_snapshot", "data_source", "triggering_event",
                "candidates", "critic_decisions", "revisions", "status", "selected_strategy",
                "final_recommendation"):
        assert key in rec
    store = DecisionStore()
    assert store.append(rec) is True
    assert store.append(rec) is False  # rerun of the same analysis -> no duplicate
    assert len(store.load_all()) == 1
    assert json.loads(store.to_json_bytes())[0]["analysis_id"] == r["analysis_id"]
    assert r["analysis_id"] in store.to_csv_bytes().decode()


def test_review_required_is_recorded(simple_raw):
    simple_raw["target"]["compounds_used"] = ["SOFT"]
    simple_raw["target"]["available_sets"] = {}
    r = run_analysis(simple_raw)
    rec = build_decision_record(r)
    assert rec["status"] == "review_required" and rec["selected_strategy"] is None
    assert "No feasible" in rec["final_recommendation"]["reason"]


def test_corrupted_log_line_is_skipped(isolated_decision_log):
    isolated_decision_log.write_text('{"analysis_id": "A"}\nnot json\n{"analysis_id": "B"}\n')
    assert [r["analysis_id"] for r in DecisionStore().load_all()] == ["A", "B"]

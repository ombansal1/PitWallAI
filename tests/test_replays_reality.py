"""Real-race replays (bundled data) and the hindsight 'what really happened' check."""
import copy

import pandas as pd
import pytest

from pitwall.data_sources import replays as rp
from pitwall.data_sources.fastf1_source import DataSourceError, build_race_state
from pitwall.orchestrator import run_analysis
from pitwall.reality import _verdict, actual_outcome, classification, compare_with_reality
from pitwall.records import DecisionStore, build_decision_record, build_reality_record
from pitwall.validation import validate_race_input

SG = "2023_singapore_grand_prix"


@pytest.fixture(scope="module")
def singapore():
    return rp.load_replay(SG)


def test_index_covers_2019_to_2026_and_excludes_2018():
    idx = rp.load_index()
    years = {r["year"] for r in idx["races"]}
    assert min(years) == 2019 and max(years) == 2026 and 2018 not in years
    assert len(idx["races"]) >= 160
    for r in idx["races"]:
        assert (rp.RACES_DIR / f"{r['id']}.json").exists()


def test_replay_session_has_real_names_and_fastf1_shape(singapore):
    for col in ("Driver", "LapNumber", "LapTime", "Time", "Compound", "TyreLife", "Stint", "TrackStatus",
                "PitInTime", "PitOutTime", "Position"):
        assert col in singapore.laps.columns
    drivers = rp.driver_options(singapore)
    assert drivers[0]["code"] == "SAI" and drivers[0]["name"] == "Carlos Sainz" and drivers[0]["team"] == "Ferrari"
    assert singapore.total_laps == 62


def test_build_state_from_real_race_is_valid_and_labelled(singapore):
    raw = build_race_state(singapore, "SAI", 17, source="replay")
    assert validate_race_input(raw) == []
    assert raw["data_source"] == "replay" and raw["target"]["name"] == "Carlos Sainz"
    assert raw["completed_laps"] == 17 and raw["target"]["compound"] == "MEDIUM"
    assert raw["fuel_effect_s_per_lap"] > 0 and "assumption" in raw["provenance"]["fuel_effect_s_per_lap"]
    assert raw["replay"] == {"source": "replay", "year": 2023, "event": "Singapore Grand Prix",
                             "driver": "SAI", "decision_lap": 17}


def test_no_hindsight_future_laps_do_not_change_the_state(singapore):
    full = build_race_state(singapore, "SAI", 17, source="replay")
    cut = copy.copy(singapore)
    cut.laps = singapore.laps[singapore.laps["LapNumber"] <= 17].copy()
    cut.weather_data = singapore.weather_data[singapore.weather_data["Time"] <= singapore.laps[
        (singapore.laps["Driver"] == "SAI") & (singapore.laps["LapNumber"] == 17)]["Time"].iloc[0]]
    truncated = build_race_state(cut, "SAI", 17, source="replay")
    assert full == truncated


def test_pit_loss_measured_from_real_stops_before_decision():
    s = rp.load_replay("2024_italian_grand_prix")
    raw = build_race_state(s, "LEC", 20, source="replay")
    assert "derived from" in raw["provenance"]["pit_loss_s"]
    assert 15 < raw["pit_loss_s"] < 40


def test_driver_mid_pit_stop_is_reported(singapore):
    with pytest.raises(DataSourceError, match="pitted at the end of lap 20"):
        build_race_state(singapore, "SAI", 20, source="replay")


def test_damaged_or_missing_race_file(tmp_path):
    with pytest.raises(DataSourceError):
        rp.load_replay("2099_nowhere_grand_prix")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DataSourceError, match="damaged"):
        rp.load_replay("bad", races_dir=tmp_path)


def test_convert_raw_handles_none_strings_and_2018_names():
    laps = {"time": [90.0] * 200 + ["None"], "lap": list(range(1, 202)), "compound": ["ULTRASOFT"] * 100 +
            ["SOFT"] * 50 + ["SUPERSOFT"] * 50 + ["None"], "stint": [1] * 201, "life": [1] * 201, "pos": [1] * 201,
            "status": ["1"] * 201, "sesT": [float(i) for i in range(201)], "drv": ["AAA"] * 201,
            "pin": ["None"] * 201, "pout": ["None"] * 201}
    snap = rp.convert_raw(2018, "Test GP", laps, {"drivers": [{"driver": "AAA", "fn": "Ann", "ln": "Able",
                                                               "team": "T", "dn": "1"}]})
    assert set(c for c in snap["laps"]["compound"] if c) == {"SOFT", "MEDIUM", "HARD"}
    assert snap["laps"]["time"][-1] is None and snap["drivers"][0]["name"] == "Ann Able"
    with pytest.raises(DataSourceError, match="missing fields"):
        rp.convert_raw(2019, "X", {"time": [], "lap": []}, {"drivers": []})


# ----------------------------------------------------------------- what really happened
def test_actual_outcome_matches_known_history(singapore):
    out = actual_outcome(singapore, "SAI", 17, ["RUS", "NOR", "LEC", "HAM"])
    assert out["stops"] == [{"lap": 20, "compound": "HARD", "under": "Safety Car"}]
    assert out["finish_position"] == 1 and out["status"] == "Finished" and out["winner"] == "SAI"
    assert "Safety Car" in out["events_after"] and out["real_time_to_flag_s"] > 0
    monza = rp.load_replay("2024_italian_grand_prix")
    lec = actual_outcome(monza, "LEC", 10)
    assert [s["lap"] for s in lec["stops"]] == [15] and lec["stops"][0]["compound"] == "HARD"
    assert lec["finish_position"] == 1


def test_classification_marks_retirements(singapore):
    cls = classification(singapore.laps)
    assert cls[0]["code"] == "SAI" and cls[0]["status"] == "Finished"
    assert any(c["status"].startswith("Retired") for c in cls)


def test_compare_with_reality_end_to_end(singapore):
    raw = build_race_state(singapore, "SAI", 17, source="replay")
    res = run_analysis(raw)
    snapshot = copy.deepcopy(res)
    cmp = compare_with_reality(res, actual_outcome(singapore, "SAI", 17, [c["code"] for c in raw["competitors"]]))
    assert res == snapshot                      # the analysis is never altered by hindsight
    assert cmp["verdict"]["code"] in ("match", "timing", "tyres", "different")
    t = cmp["times"]
    assert t["model_estimate_for_team_call_s"] > 0 and t["real_time_for_team_call_s"] > 0
    assert cmp["model_error_s"] == pytest.approx(t["real_time_for_team_call_s"] - t["model_estimate_for_team_call_s"],
                                                 abs=1e-2)
    assert any("Safety Car" in e for e in cmp["explanations"])
    assert sum(cmp["error_by_condition_s"].values()) == pytest.approx(cmp["model_error_s"], abs=0.5)


@pytest.mark.parametrize("model,real,code", [
    ([], [], "match"),
    ([{"lap": 20, "compound": "HARD"}], [{"lap": 21, "compound": "HARD"}], "match"),
    ([{"lap": 20, "compound": "HARD"}], [{"lap": 26, "compound": "HARD"}], "timing"),
    ([{"lap": 20, "compound": "HARD"}], [{"lap": 20, "compound": "MEDIUM"}], "tyres"),
    ([{"lap": 20, "compound": "HARD"}], [], "different"),
])
def test_verdict_rules(model, real, code):
    assert _verdict(model, real)["code"] == code


def test_reality_record_is_linked_not_merged(singapore, isolated_decision_log):
    raw = build_race_state(singapore, "SAI", 17, source="replay")
    res = run_analysis(raw)
    cmp = compare_with_reality(res, actual_outcome(singapore, "SAI", 17))
    store = DecisionStore()
    store.append(build_decision_record(res))
    rec = build_reality_record(cmp, "2026-10-17T00:00:00+00:00")
    assert store.append(rec) and not store.append(rec)
    df = store.summary_frame()
    assert len(df) == 1 and df.iloc[0]["real_race_check"] == cmp["verdict"]["text"]


def test_every_bundled_race_builds_a_valid_state():
    """Smoke: one mid-race moment for the winner of every bundled race (data robustness)."""
    bad = []
    for r in rp.load_index()["races"]:
        s = rp.load_replay(r["id"])
        drv = rp.driver_options(s)[0]["code"]
        for lap in (s.total_laps // 2, s.total_laps // 2 + 1, s.total_laps // 2 - 1):
            try:
                raw = build_race_state(s, drv, lap, source="replay")
                break
            except DataSourceError:
                raw = None
        if raw is None or validate_race_input(raw):
            bad.append(r["id"])
    assert bad == []


def test_lap_times_are_real_values(singapore):
    sai = singapore.laps[(singapore.laps["Driver"] == "SAI") & (singapore.laps["LapNumber"] == 46)]
    assert sai["LapTime"].iloc[0] == pd.Timedelta(seconds=97.913)


def test_last_lap_crash_counts_as_retirement(singapore):
    """Russell crashed on the final lap at Singapore 2023: not a finisher in the timing data."""
    rus = next(c for c in classification(singapore.laps) if c["code"] == "RUS")
    assert rus["status"] == "Retired after lap 61" and rus["laps"] == 61

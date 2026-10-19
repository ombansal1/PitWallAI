"""FastF1 adapter tests using an offline fake session (no network, FastF1 not required)."""
import types

import pandas as pd
import pytest

from pitwall.data_sources import fastf1_source as ff
from pitwall.data_sources.synthetic import load_scenario
from pitwall.orchestrator import run_analysis
from pitwall.validation import validate_race_input

LAP = 90.0


def fake_session(total=30, upto_rain_lap=None, sc_lap=None, future_garbage=True):
    rows = []
    drivers = [("AAA", 0.0, "MEDIUM"), ("BBB", 1.5, "HARD"), ("TGT", 4.0, "MEDIUM"),
               ("CCC", 6.0, "SOFT"), ("DDD", 9.0, "MEDIUM")]
    for code, offset, comp in drivers:
        for lap in range(1, total + 1):
            t = pd.Timedelta(seconds=lap * LAP + offset + 120)
            future = lap > 12 and future_garbage
            rows.append({
                "Driver": code, "Team": "Team " + code, "LapNumber": float(lap),
                "LapTime": pd.Timedelta(seconds=(500.0 if future else LAP + 0.05 * lap)),
                "Time": t, "Compound": ("WET" if future else comp), "TyreLife": float(lap),
                "Stint": 1.0, "PitInTime": pd.NaT, "PitOutTime": pd.NaT,
                "TrackStatus": "4" if (sc_lap and lap == sc_lap) else "1", "Position": 1.0})
    laps = pd.DataFrame(rows)
    weather = pd.DataFrame({"Time": [pd.Timedelta(seconds=s) for s in (100, 1000, 2000, 2500)],
                            "Rainfall": [False, False, True, True]})  # rain only later in the race
    return types.SimpleNamespace(laps=laps, total_laps=total, weather_data=weather,
                                 event={"EventName": "Fake Grand Prix"},
                                 date=pd.Timestamp("2023-05-01"))


def test_builds_valid_common_race_state():
    raw = ff.build_race_state(fake_session(), "TGT", 10)
    assert validate_race_input(raw) == []
    assert raw["data_source"] == "fastf1" and "not a live feed" in raw["data_source_label"]
    assert raw["completed_laps"] == 10 and raw["total_laps"] == 30
    assert raw["target"]["gap_to_leader_s"] == pytest.approx(4.0)  # common leader reference
    assert {c["code"] for c in raw["competitors"]} == {"AAA", "BBB", "CCC", "DDD"}
    assert raw["target"]["tyre_age_laps"] == 10
    assert "synthetic assumption" in raw["provenance"]["pit_loss_s"]
    assert "derived" in raw["provenance"]["baseline_pace_s"]
    assert run_analysis(raw)["status"] in ("recommended", "review_required")


def test_uses_only_laps_up_to_decision_point():
    a = ff.build_race_state(fake_session(future_garbage=True), "TGT", 10)
    b = ff.build_race_state(fake_session(future_garbage=False), "TGT", 10)
    assert a["target"] == b["target"] and a["competitors"] == b["competitors"]


def test_later_weather_is_not_used_as_forecast():
    raw = ff.build_race_state(fake_session(), "TGT", 10)  # rainfall only recorded much later
    assert raw["track_condition"] == "dry" and raw["rain_start_lap"] is None


def test_safety_car_status_gives_labelled_assumption():
    raw = ff.build_race_state(fake_session(sc_lap=10), "TGT", 10)
    assert raw["sc_start_lap"] == 11 and "assumption" in raw["provenance"]["safety_car"]


@pytest.mark.parametrize("driver,lap", [("ZZZ", 10), ("TGT", 30), ("TGT", 0)])
def test_unavailable_driver_or_lap(driver, lap):
    with pytest.raises(ff.DataSourceError):
        ff.build_race_state(fake_session(), driver, lap)


def test_missing_fields_and_total_laps():
    s = fake_session()
    s.laps = s.laps.drop(columns=["TyreLife", "Compound"])
    with pytest.raises(ff.DataSourceError, match="missing required fields"):
        ff.build_race_state(s, "TGT", 10)
    s2 = fake_session()
    s2.total_laps = None
    with pytest.raises(ff.DataSourceError):
        ff.build_race_state(s2, "TGT", 10)


def test_too_few_clean_laps():
    s = fake_session()
    s.laps.loc[s.laps["Driver"] == "TGT", "LapTime"] = pd.NaT
    with pytest.raises(ff.DataSourceError, match="too early"):
        ff.build_race_state(s, "TGT", 10)


def test_load_failure_is_reported_and_fallback_is_labelled(monkeypatch):
    def broken():
        raise ff.DataSourceError("network unreachable")
    monkeypatch.setattr(ff, "_ff1", broken)
    with pytest.raises(ff.DataSourceError):
        ff.load_session(2023, 1)
    out = ff.load_race_state_with_fallback(lambda: ff.load_session(2023, 1), load_scenario("dry_normal"))
    assert out["used_fallback"] is True and "network unreachable" in out["error"]
    assert out["raw"]["data_source"] == "synthetic_fallback"
    assert "Synthetic fallback" in out["raw"]["data_source_label"]
    assert run_analysis(out["raw"])["status"] == "recommended"


def test_unloaded_session_raises_data_source_error(monkeypatch):
    class NotLoaded:
        def load(self, **kw):
            pass

        @property
        def laps(self):
            raise RuntimeError("DataNotLoadedError")

    fake_mod = types.SimpleNamespace(Cache=types.SimpleNamespace(enable_cache=lambda p: None),
                                     get_session=lambda y, r, s: NotLoaded())
    monkeypatch.setattr(ff, "_ff1", lambda: fake_mod)
    with pytest.raises(ff.DataSourceError, match="could not be retrieved"):
        ff.load_session(2023, 1)

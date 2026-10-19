"""ML tyre & lap-time model: dataset, training/evaluation, curves and use by the simulator."""
import json

import pytest

from pitwall.data_sources import replays as rp
from pitwall.data_sources.fastf1_source import build_race_state
from pitwall.ml import tyre_model as tm
from pitwall.orchestrator import run_analysis
from pitwall.simulation import curve_tyre_adj
from pitwall.validation import parse_race_state, validate_race_input

SMALL = ["2023_british_grand_prix", "2023_italian_grand_prix", "2024_british_grand_prix",
         "2024_italian_grand_prix", "2025_british_grand_prix", "2025_italian_grand_prix"]


def test_dataset_is_clean_and_driver_relative():
    df = tm.build_dataset(SMALL[:2])
    assert set(tm.FEATURES) <= set(df.columns) and len(df) > 1000
    assert df[tm.FEATURES + ["y"]].notna().all().all()
    assert df["compound"].isin(tm.DRY).all() and (df["tyre_age"] >= 0).all()
    assert df["gap_ahead"].between(0, tm.CLEAN_AIR_GAP_S).all()
    med = df.groupby(["race", "drv"])["y"].median().abs()
    assert med.max() < 1e-6          # each lap is compared with that driver's own typical lap


def test_train_evaluate_and_save(tmp_path):
    card = tm.train_and_save(SMALL, out_dir=tmp_path)
    assert card["train"]["seasons"] == [2023, 2024] and card["test"]["seasons"] == [2025]
    assert card["model_size_search"]["validation_season"] == 2024
    assert card["model_size_search"]["chosen_trees"] in tm.TREE_CHOICES
    for k in ("ml", "formula", "no_model"):
        assert set(card["metrics"][k]) == {"mae_s", "rmse_s", "r2"}
    assert {i["feature"] for i in card["importance"]} == set(tm.FEATURE_TEXT.values())
    curves = json.loads((tmp_path / "tyre_curves.json").read_text())["circuits"]
    assert tm.ALL in curves and "British Grand Prix" in curves
    for table in curves.values():
        for c in tm.DRY:
            v = table[c]
            assert len(v) == tm.MAX_AGE + 1
            assert all(b >= a - 1e-9 for a, b in zip(v, v[1:]))      # older tyres never faster
    assert curves["British Grand Prix"]["SOFT"][0] == 0.0             # reference: a new soft
    m = tm.load_model(tmp_path / "tyre_model.joblib")
    assert m is not None and m.predict("SOFT", 20, 0.5, "British Grand Prix") > m.predict("SOFT", 0, 0.5,
                                                                                         "British Grand Prix")


def test_shipped_model_beats_the_formula_on_unseen_seasons():
    card = tm.load_card()
    assert card and card["test"]["seasons"] == list(tm.TEST_YEARS)
    m = card["metrics"]
    assert m["ml"]["mae_s"] < m["formula"]["mae_s"] < m["no_model"]["mae_s"]
    assert all(v["ml_mae_s"] < v["formula_mae_s"] for v in card["per_compound"].values())


def test_curve_interpolation_and_extrapolation():
    curve = [0.0, 0.1, 0.3, 0.6]
    assert curve_tyre_adj(curve, 0) == 0.0 and curve_tyre_adj(curve, 2) == pytest.approx(0.3)
    assert curve_tyre_adj(curve, 1.5) == pytest.approx(0.2)
    assert curve_tyre_adj(curve, 5) == pytest.approx(1.2)                 # last slope 0.3 per lap
    assert curve_tyre_adj([0.5, 0.7], 1, multiplier=2.0) == pytest.approx(0.9)   # offset kept, wear doubled


def test_ml_tyre_model_in_a_real_race(scenario):
    s = rp.load_replay("2024_british_grand_prix")
    raw = build_race_state(s, "HAM", 18, source="replay")
    ml = dict(raw, tyre_model=tm.tyre_model_payload(raw["track_name"]))
    assert ml["tyre_model"]["circuit"] == "British Grand Prix"
    assert validate_race_input(ml) == []
    a, b = parse_race_state(raw), parse_race_state(ml)
    assert a.target.baseline_pace_s == raw["target"]["baseline_pace_s"]
    assert b.target.baseline_pace_s != a.target.baseline_pace_s and b.tyre_curves
    r = run_analysis(ml)
    assert r["status"] == "recommended"
    assert any(e["action"] == "tyre_model" for e in r["log"])
    syn = dict(scenario("dry_normal"), tyre_model=tm.tyre_model_payload(None))
    assert syn["tyre_model"]["circuit"] == tm.ALL
    assert parse_race_state(syn).target.baseline_pace_s == syn["target"]["baseline_pace_s"]   # no samples
    assert run_analysis(syn)["status"] == "recommended"


def test_bad_tyre_model_is_reported(scenario):
    raw = scenario("dry_normal")
    raw["tyre_model"] = {"kind": "ml", "curves": {"SOFT": [0.0, float("nan")], "SUPER": [0, 1]}}
    fields = {i["field"] for i in validate_race_input(raw)}
    assert {"tyre_model.curves.SOFT", "tyre_model.curves.SUPER"} <= fields
    raw["tyre_model"] = "magic"
    assert any(i["field"] == "tyre_model" for i in validate_race_input(raw))


def test_progress_curve_and_agents_vs_reality_evaluation():
    from pitwall.evaluation import evaluate, load, race_moments
    s = rp.load_replay("2024_british_grand_prix")
    raw = build_race_state(s, "HAM", 18, source="replay")
    pay = tm.tyre_model_payload(raw["track_name"])
    pc = pay["progress_curve"]
    assert len(pc) == tm.PROGRESS_POINTS and pc[0] == 0.0 and all(b <= a for a, b in zip(pc, pc[1:]))
    st = parse_race_state(dict(raw, tyre_model=pay))
    assert st.progress_effect(st.completed_laps) == 0.0 and st.progress_effect(st.total_laps) < 0
    ms = race_moments(s)
    assert ms and all(m["lap"] == m["real_first_stop"] - 3 for m in ms)
    out = evaluate(["2024_british_grand_prix"])
    assert {r["tyre_model"] for r in out["rows"]} == {"formula", "ml"}
    assert out["summary"]["formula"]["runs"] == out["summary"]["ml"]["runs"] == len(ms)
    shipped = load()
    assert shipped and shipped["summary"]["formula"]["runs"] > 300
    assert shipped["summary"]["ml"]["median_time_error_clean_s"] < shipped["summary"]["formula"]["median_time_error_clean_s"]

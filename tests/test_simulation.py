import pytest

from pitwall.models import PitStop, Strategy
from pitwall.simulation import (check_strategy_constraints, project_positions, rank_key,
                                simulate_strategy, tyre_schedule)
from pitwall.validation import parse_race_state


def S(*stops, sid=None):
    stops = [PitStop(l, c) for l, c in stops]
    return Strategy(sid or ("X" + "".join(f"{s.lap}{s.compound}" for s in stops)), "t", "test", stops)


def codes(v):
    return {x["code"] for x in v}


# ----------------------------------------------------------------- hand-calculated example
def test_manual_stay_out_time(simple_raw, simple_cfg):
    st = parse_race_state(simple_raw)
    r = simulate_strategy(st, st.target, S(), simple_cfg)
    # laps 6-10 on SOFT ages 2..6: 90 + 0.1*age -> 90.2+90.3+90.4+90.5+90.6
    assert [round(x["lap_time_s"], 6) for x in r.laps] == [90.2, 90.3, 90.4, 90.5, 90.6]
    assert r.remaining_time_s == pytest.approx(452.0)
    assert r.pit_loss_total_s == 0 and r.pit_loss_count == 0
    assert r.finish_time_ref_s == pytest.approx(5.0 + 452.0)  # common reference = gap + time


def test_manual_one_stop_time(simple_raw, simple_cfg):
    st = parse_race_state(simple_raw)
    r = simulate_strategy(st, st.target, S((7, "MEDIUM")), simple_cfg)
    # L6,L7 SOFT ages 2,3 = 90.2+90.3; L8-10 MEDIUM ages 0,1,2 = 90.35+90.40+90.45; + 20 s once
    assert r.lap_time_total_s == pytest.approx(451.7)
    assert r.pit_loss_total_s == pytest.approx(20.0)
    assert r.remaining_time_s == pytest.approx(471.7)


# ----------------------------------------------------------------- laps & pit timing
def test_remaining_laps_and_convention(simple_raw, simple_cfg):
    st = parse_race_state(simple_raw)
    assert st.remaining_laps == 5 and st.first_lap == 6
    r = simulate_strategy(st, st.target, S((6, "HARD")), simple_cfg)
    assert [x["lap"] for x in r.laps] == [6, 7, 8, 9, 10]
    assert r.laps[0]["pit_loss_s"] == 20.0 and r.laps[0]["compound"] == "SOFT"  # in-lap on old tyres
    assert r.laps[1]["compound"] == "HARD" and r.laps[1]["tyre_age"] == 0


@pytest.mark.parametrize("lap", [4, 5, 10, 11])
def test_stop_outside_remaining_race_rejected(simple_raw, simple_cfg, lap):
    st = parse_race_state(simple_raw)
    assert "PIT_OUTSIDE_RACE" in codes(check_strategy_constraints(st, st.target, S((lap, "HARD")), simple_cfg))


@pytest.mark.parametrize("lap", [6, 9])
def test_stop_inside_race_accepted(simple_raw, simple_cfg, lap):
    st = parse_race_state(simple_raw)
    assert check_strategy_constraints(st, st.target, S((lap, "HARD")), simple_cfg) == []


# ----------------------------------------------------------------- tyres
def test_tyre_age_increments_and_resets(simple_raw, simple_cfg):
    st = parse_race_state(simple_raw)
    sched = tyre_schedule(st, st.target, [PitStop(7, "MEDIUM")])
    assert [(r["compound"], r["tyre_age"]) for r in sched] == [
        ("SOFT", 2), ("SOFT", 3), ("MEDIUM", 0), ("MEDIUM", 1), ("MEDIUM", 2)]


def test_degradation_multiplier_scales_wear(simple_raw, simple_cfg):
    base = parse_race_state(simple_raw)
    simple_raw["deg_multiplier"] = 2.0
    high = parse_race_state(simple_raw)
    a = simulate_strategy(base, base.target, S(), simple_cfg)
    b = simulate_strategy(high, high.target, S(), simple_cfg)
    assert b.remaining_time_s - a.remaining_time_s == pytest.approx(2.0)  # wear 2.0 s doubled -> 4.0 s


def test_compound_specific_pace(simple_raw, simple_cfg):
    st = parse_race_state(simple_raw)
    soft = simulate_strategy(st, st.target, S((6, "SOFT")), simple_cfg).laps[1]["lap_time_s"]
    hard = simulate_strategy(st, st.target, S((6, "HARD")), simple_cfg).laps[1]["lap_time_s"]
    assert hard - soft == pytest.approx(0.70)


def test_stint_too_long_rejected(simple_raw, simple_cfg):
    simple_raw["total_laps"], simple_raw["completed_laps"] = 70, 10
    st = parse_race_state(simple_raw)
    assert "STINT_TOO_LONG" in codes(check_strategy_constraints(st, st.target, S(), simple_cfg))


# ----------------------------------------------------------------- pit loss exactly once
def test_pit_loss_counted_once_per_stop(simple_raw, simple_cfg):
    st = parse_race_state(simple_raw)
    one = simulate_strategy(st, st.target, S((7, "MEDIUM")), simple_cfg)
    two = simulate_strategy(st, st.target, S((6, "MEDIUM"), (8, "HARD")), simple_cfg)
    assert one.pit_loss_count == 1 and one.pit_loss_total_s == pytest.approx(20.0)
    assert two.pit_loss_count == 2 and two.pit_loss_total_s == pytest.approx(40.0)
    assert one.remaining_time_s == pytest.approx(sum(x["lap_time_s"] for x in one.laps) + 20.0)


# ----------------------------------------------------------------- weather & safety car
def test_rain_window_and_tyre_suitability(simple_raw, simple_cfg):
    simple_raw["rain_start_lap"], simple_raw["rain_end_lap"] = 8, 9
    st = parse_race_state(simple_raw)
    assert [st.condition_on_lap(l) for l in range(6, 11)] == ["dry", "dry", "wet", "wet", "dry"]
    r = simulate_strategy(st, st.target, S(), simple_cfg)
    pen = simple_cfg.track_condition_penalty_s["wet"] + simple_cfg.tyre_weather_penalty_s["wet"]["SOFT"]
    assert r.laps[2]["weather_adj_s"] == pytest.approx(pen) and r.laps[4]["weather_adj_s"] == 0
    inter = simulate_strategy(st, st.target, S((7, "INTERMEDIATE")), simple_cfg)
    assert inter.laps[2]["weather_adj_s"] == pytest.approx(simple_cfg.track_condition_penalty_s["wet"])
    assert inter.laps[4]["weather_adj_s"] == pytest.approx(simple_cfg.tyre_weather_penalty_s["dry"]["INTERMEDIATE"])


def test_slicks_on_wet_rejected(simple_raw, simple_cfg):
    simple_raw["rain_start_lap"] = 6
    st = parse_race_state(simple_raw)
    assert "SLICKS_ON_WET" in codes(check_strategy_constraints(st, st.target, S(), simple_cfg))
    assert check_strategy_constraints(st, st.target, S((6, "INTERMEDIATE")), simple_cfg) == []


def test_safety_car_limited_to_window_and_halves_pit_loss(simple_raw, simple_cfg):
    simple_raw["sc_start_lap"], simple_raw["sc_end_lap"] = 7, 8
    st = parse_race_state(simple_raw)
    r = simulate_strategy(st, st.target, S((7, "MEDIUM")), simple_cfg)
    sc_time = 90.0 * simple_cfg.sc_lap_time_factor
    assert [x["safety_car"] for x in r.laps] == [False, True, True, False, False]
    assert r.laps[1]["lap_time_s"] == pytest.approx(sc_time) and r.laps[2]["lap_time_s"] == pytest.approx(sc_time)
    assert r.laps[0]["lap_time_s"] == pytest.approx(90.2) and r.laps[3]["sc_adj_s"] == 0
    assert r.pit_loss_total_s == pytest.approx(20.0 * simple_cfg.sc_pit_loss_factor)
    green = simulate_strategy(st, st.target, S((6, "MEDIUM")), simple_cfg)
    assert green.pit_loss_total_s == pytest.approx(20.0)


# ----------------------------------------------------------------- constraints
def test_constraint_violations(simple_raw, simple_cfg):
    simple_raw["pit_closed_laps"] = [7]
    simple_raw["target"]["available_sets"] = {"SOFT": 0, "MEDIUM": 1, "HARD": 1}
    st = parse_race_state(simple_raw)
    chk = lambda s: codes(check_strategy_constraints(st, st.target, s, simple_cfg))  # noqa: E731
    assert "PIT_LANE_CLOSED" in chk(S((7, "HARD")))
    assert "COMPOUND_UNAVAILABLE" in chk(S((6, "SOFT")))
    assert "COMPOUND_UNAVAILABLE" in chk(S((6, "MEDIUM"), (8, "MEDIUM")))
    assert "TOO_MANY_STOPS" in chk(S((6, "MEDIUM"), (8, "HARD"), (9, "HARD")))


def test_two_compound_rule(simple_raw, simple_cfg):
    simple_raw["target"]["compounds_used"] = ["SOFT"]
    st = parse_race_state(simple_raw)
    assert "TWO_COMPOUND_RULE" in codes(check_strategy_constraints(st, st.target, S(), simple_cfg))
    assert "TWO_COMPOUND_RULE" not in codes(check_strategy_constraints(st, st.target, S((6, "HARD")), simple_cfg))


# ----------------------------------------------------------------- positions & tie-break
def test_positions_use_common_reference():
    pos = project_positions(100.0, "TGT", {"AAA": 99.0, "BBB": 101.5}, {"AAA": 0, "TGT": 3, "BBB": 8})
    assert pos["order"] == ["AAA", "TGT", "BBB"] and pos["position_in_group"] == 2
    assert pos["gap_to_car_ahead_s"] == pytest.approx(1.0) and pos["gap_to_car_behind_s"] == pytest.approx(1.5)
    tie = project_positions(100.0, "TGT", {"AAA": 100.0}, {"AAA": 1.0, "TGT": 3.0})
    assert tie["order"] == ["AAA", "TGT"]  # tie broken by current gap (car already ahead stays ahead)


def test_traffic_after_rejoin(simple_raw, simple_cfg):
    import dataclasses
    cfg = dataclasses.replace(simple_cfg, traffic_enabled=True)
    st = parse_race_state(simple_raw)
    no_tr = simulate_strategy(st, st.target, S((6, "HARD")), cfg)
    ahead = no_tr.laps[0]["cumulative_ref_s"] - 1.0  # a rival 1.0 s ahead when we rejoin
    tr = simulate_strategy(st, st.target, S((6, "HARD")), cfg, rival_traces={"AAA": {6: ahead}})
    assert tr.traffic_laps == cfg.traffic_laps
    assert tr.remaining_time_s - no_tr.remaining_time_s == pytest.approx(cfg.traffic_penalty_s * cfg.traffic_laps)


def test_rank_key_tie_break():
    a, b = S(sid="B"), S((6, "HARD"), sid="A")
    assert min([b, a], key=lambda s: rank_key(100.0, s)) is a  # equal time -> fewer stops
    c, d = S((6, "HARD"), sid="C"), S((8, "HARD"), sid="D")
    assert min([c, d], key=lambda s: rank_key(100.0, s)) is d  # then later first stop

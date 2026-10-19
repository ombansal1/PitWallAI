"""Multiple Safety Car / VSC / rain periods, the chat scenario reader and hindsight (real events) mode."""
import copy

import pytest

from pitwall.agents.assistant import ChatContext, PitWallAssistant
from pitwall.data_sources import replays as rp
from pitwall.data_sources.fastf1_source import build_race_state
from pitwall.events import apply_event
from pitwall.models import PitStop, Strategy
from pitwall.orchestrator import run_analysis
from pitwall.reality import apply_real_events, real_events_after
from pitwall.scenario import apply, parse
from pitwall.simulation import simulate_strategy
from pitwall.validation import parse_race_state, validate_race_input


def S(*stops):
    return Strategy("X", "t", "test", [PitStop(l, c) for l, c in stops])


# ----------------------------------------------------------------- model: VSC + several periods
def test_vsc_is_cheaper_than_green_but_dearer_than_sc(simple_raw, simple_cfg):
    simple_raw["neutral_periods"] = [{"kind": "VSC", "start": 6, "end": 6}, {"kind": "SC", "start": 8, "end": 8}]
    st = parse_race_state(simple_raw)
    assert [st.neutral_kind(l) for l in range(6, 11)] == ["VSC", None, "SC", None, None]
    vsc = simulate_strategy(st, st.target, S((6, "HARD")), simple_cfg)
    sc = simulate_strategy(st, st.target, S((8, "HARD")), simple_cfg)
    green = simulate_strategy(st, st.target, S((7, "HARD")), simple_cfg)
    assert vsc.pit_loss_total_s == pytest.approx(20 * simple_cfg.vsc_pit_loss_factor)
    assert sc.pit_loss_total_s == pytest.approx(20 * simple_cfg.sc_pit_loss_factor)
    assert green.pit_loss_total_s == pytest.approx(20)
    assert vsc.laps[0]["lap_time_s"] == pytest.approx(90 * simple_cfg.vsc_lap_time_factor)


def test_several_wet_spells(simple_raw):
    simple_raw["wet_periods"] = [{"start": 6, "end": 6}, {"start": 9, "end": 10}]
    st = parse_race_state(simple_raw)
    assert [st.condition_on_lap(l) for l in range(6, 11)] == ["wet", "dry", "dry", "wet", "wet"]
    assert st.wet_runs() == [{"start": 6, "end": 6}, {"start": 9, "end": 10}]


@pytest.mark.parametrize("periods,field", [
    ([{"kind": "XX", "start": 6, "end": 7}], "neutral_periods[0].kind"),
    ([{"kind": "SC", "start": 8, "end": 7}], "neutral_periods[0].end"),
    ([{"kind": "SC", "start": 2, "end": 3}], "neutral_periods[0]"),
    ("lap 6", "neutral_periods"),
])
def test_period_validation(simple_raw, periods, field):
    simple_raw["neutral_periods"] = periods
    assert field in {i["field"] for i in validate_race_input(simple_raw)}


def test_vsc_event_and_second_safety_car(scenario):
    raw = scenario("dry_normal")
    raw, _ = apply_event(raw, {"type": "vsc", "laps_until": 2, "duration_laps": 2})
    raw, _ = apply_event(raw, {"type": "safety_car", "laps_until": 0, "duration_laps": 3})
    raw, _ = apply_event(raw, {"type": "safety_car", "laps_until": 10, "duration_laps": 3})
    assert (raw["sc_start_lap"], raw["sc_end_lap"]) == (21, 23)
    assert raw["neutral_periods"] == [{"kind": "VSC", "start": 23, "end": 24}, {"kind": "SC", "start": 31, "end": 33}]
    res = run_analysis(raw)
    assert res["status"] == "recommended"
    assert any("Virtual Safety Car on laps 23–24" in x for x in res["research"]["track_status"]["summary"])


def test_planner_tests_a_stop_in_every_neutral_period(scenario):
    raw = scenario("dry_normal")
    raw["neutral_periods"] = [{"kind": "VSC", "start": 25, "end": 26}, {"kind": "SC", "start": 40, "end": 42}]
    res = run_analysis(raw)
    first_laps = {d["stops"][0]["lap"] for d in res["candidate_defs"].values() if d["stops"]}
    assert 25 in first_laps and 40 in first_laps


# ----------------------------------------------------------------- scenario reader
@pytest.mark.parametrize("text,parts", [
    ("After lap 20 of 56, on 18-lap-old mediums, VSC on laps 24-26 and rain from lap 35 to 45",
     ["Deciding after lap 20 of 56", "On Medium tyres, 18 laps old", "VSC on laps 24–26", "Rain from lap 35 to lap 45"]),
    ("15 laps to go, raining until lap 50, vsc in 3 laps",
     ["Deciding after lap 41 of 56", "Rain from lap 42 to lap 50", "VSC on laps 44–45"]),
    ("Scenario: lap 40 of 70, safety car laps 42 to 45, then a VSC on lap 55, rain from lap 60",
     ["Deciding after lap 40 of 70", "Safety Car on laps 42–45", "VSC on laps 55–56", "Rain from lap 60 to the finish"]),
    ("imagine a 58-lap race, after lap 12, on softs, high tyre wear",
     ["Deciding after lap 12 of 58", "On Soft tyres", "Tyre wear ×1.5"]),
])
def test_scenario_parsing(scenario, text, parts):
    raw = scenario("dry_normal")
    sc = parse(text, raw)
    assert sc.parts == parts
    assert validate_race_input(apply(raw, sc)) == []


def test_scenario_apply_never_mutates_and_replaces_events(scenario):
    raw = scenario("safety_car")
    before = copy.deepcopy(raw)
    new = apply(raw, parse("lap 30 of 56, rain from lap 40", raw))
    assert raw == before
    assert new["sc_start_lap"] is None and new["wet_periods"] == [{"start": 40, "end": 56}]
    assert new["completed_laps"] == 30


def test_chat_scenario_runs_agents_with_charts(scenario):
    raw = scenario("dry_normal")
    a = PitWallAssistant().answer("After lap 20 of 56 on 18-lap-old mediums, VSC on laps 24-26, rain from lap 35 to 45",
                                  ChatContext(raw=raw, question_key="full_strategy", result=run_analysis(raw)))
    assert a.intent == "what_if" and "**What they decided:" in a.text
    assert "Intermediate" in a.text and "VSC" in a.text
    assert a.charts["plans"][0]["name"] == "Agents' plan" and a.charts["neutral"] and a.charts["wet"]
    assert a.scenario_raw["neutral_periods"] == [{"kind": "VSC", "start": 24, "end": 26}]


def test_scenario_charts_are_valid(scenario):
    from ui import charts
    raw = scenario("dry_normal")
    a = PitWallAssistant().answer("safety car on laps 25-27 and rain from lap 40 to 50",
                                  ChatContext(raw=raw, question_key="full_strategy", result=run_analysis(raw)))
    charts.scenario_timeline(a.charts).to_dict(validate=True)
    charts.options_chart(a.charts["options"]).to_dict(validate=True)


# ----------------------------------------------------------------- hindsight: real events given to agents
def test_real_events_extracted_from_real_race():
    s = rp.load_replay("2021_abu_dhabi_grand_prix")
    ev = real_events_after(s, 30)
    assert ev["neutral"] == [{"kind": "VSC", "start": 36, "end": 38}, {"kind": "SC", "start": 53, "end": 57}]
    assert ev["wet"] == []


def test_hindsight_mode_is_labelled_and_changes_the_call():
    s = rp.load_replay("2024_british_grand_prix")
    raw = build_race_state(s, "HAM", 20, source="replay")
    new = apply_real_events(raw, real_events_after(s, 20))
    assert validate_race_input(new) == []
    assert new["replay"]["real_events"] is True and "HINDSIGHT" in new["notes"][-1]
    assert "HINDSIGHT" in new["provenance"]["safety_car / VSC / rain"]
    assert raw.get("rain_start_lap") is None            # default mode never sees the rain
    res = run_analysis(new)
    assert "Intermediate" in res["recommendation"]["headline"]


def test_default_mode_still_has_no_hindsight():
    s = rp.load_replay("2021_abu_dhabi_grand_prix")
    raw = build_race_state(s, "VER", 30, source="replay")
    assert raw["neutral_periods"] == [] if "neutral_periods" in raw else True
    assert raw["sc_start_lap"] is None and not (raw.get("replay") or {}).get("real_events")

"""Agent workflow tests, including the labelled Critic rejection-and-revision demonstrations."""
import copy
import dataclasses

import pytest

from pitwall.config import DEFAULT_CONFIG
from pitwall.models import PitStop, Strategy
from pitwall.orchestrator import run_analysis
from pitwall.simulation import check_strategy_constraints, rank_key
from pitwall.validation import parse_race_state

SCENARIOS = ["dry_normal", "safety_car", "rain_approaching", "high_degradation"]


def strip_volatile(r):
    r = copy.deepcopy(r)
    for k in ("analysis_id", "created_at"):
        r.pop(k)
    return r


@pytest.mark.parametrize("sid", SCENARIOS)
def test_scenarios_produce_validated_recommendation(scenario, sid):
    r = run_analysis(scenario(sid))
    assert r["status"] == "recommended"
    assert r["retry_count"] <= DEFAULT_CONFIG.max_revisions
    assert r["selected_id"] in r["critic_reports"][-1]["accepted"]


@pytest.mark.parametrize("sid", SCENARIOS)
def test_selected_is_lowest_time_feasible(scenario, sid):
    """Independently recompute every candidate: selection must be the feasible minimum."""
    raw = scenario(sid)
    r = run_analysis(raw)
    st = parse_race_state(raw)
    best = None
    for d in r["candidate_defs"].values():
        s = Strategy(d["id"], d["label"], d["family"], [PitStop(x["lap"], x["compound"]) for x in d["stops"]])
        if check_strategy_constraints(st, st.target, s):
            continue
        t = r["simulations"][s.id]["remaining_time_s"]
        if best is None or rank_key(t, s) < best[0]:
            best = (rank_key(t, s), s.id)
    assert r["selected_id"] == best[1]
    # and the time matches a fresh simulation (no traffic in the check -> allow traffic delta only)
    sim = r["simulations"][r["selected_id"]]
    assert sim["remaining_time_s"] == pytest.approx(sim["lap_time_total_s"] + sim["pit_loss_total_s"], abs=1e-3)


def test_deterministic_for_identical_inputs(scenario):
    a = run_analysis(scenario("rain_approaching"))
    b = run_analysis(scenario("rain_approaching"))
    assert strip_volatile(a) == strip_volatile(b)


def test_inputs_are_not_mutated(scenario):
    raw = scenario("high_degradation")
    snap = copy.deepcopy(raw)
    run_analysis(raw)
    assert raw == snap


def test_planner_controls_research_and_tools(scenario):
    dry = run_analysis(scenario("dry_normal"))
    assert "weather_analysis" not in dry["research"]
    rain = run_analysis(scenario("rain_approaching"))
    assert "weather_analysis" in rain["research"]
    comp = run_analysis(scenario("dry_normal"), "compound_choice")
    assert comp["rivals"] == {} and comp["positions"] == {}
    assert {c["family"] for c in comp["candidates"]} == {"pit_now"}
    assert "project_rivals" not in comp["plan"]["simulation_tools"]
    binary = run_analysis(scenario("dry_normal"), "pit_now_vs_stay_out")
    assert {c["family"] for c in binary["candidates"]} == {"pit_now", "stay_out"}


# ======================================================================================
# CRITIC REJECTION-AND-REVISION DEMONSTRATIONS
# ======================================================================================
def test_demo_critic_rejects_all_then_planner_revises(scenario):
    """High degradation: every one-stop plan exceeds safe tyre life -> Critic rejects all ->
    routes NO_FEASIBLE_CANDIDATE to the Planner -> plan v2 adds two-stop candidates -> accepted."""
    r = run_analysis(scenario("high_degradation"))
    first = r["critic_reports"][0]
    assert first["accepted"] == []
    assert first["revision_requests"][0]["target"] == "planner"
    assert first["revision_requests"][0]["code"] == "NO_FEASIBLE_CANDIDATE"
    assert r["retry_count"] == 1 and len(r["plan_history"]) == 2
    assert r["plan_history"][1]["candidate_spec"]["two_stop"] is True
    assert r["status"] == "recommended" and r["recommendation"]["stops"].__len__() == 2
    assert any(e["action"] == "route_to_planner" for e in r["log"])


def test_demo_critic_adds_missing_wet_transition(scenario):
    """Binary question in the rain: pit-now slicks and stay-out both fail -> Planner adds wet stops."""
    r = run_analysis(scenario("rain_approaching"), "pit_now_vs_stay_out")
    codes = {q["code"] for q in r["critic_reports"][0]["revision_requests"]}
    assert "MISSING_WET_TRANSITION" in codes
    assert r["status"] == "recommended" and r["recommendation"]["stops"][0]["compound"] == "INTERMEDIATE"


def test_demo_critic_rejects_numerical_fault_and_executor_reruns(scenario):
    """Fault injection (test hook): first simulation attempt double-counts a pit loss.
    The Critic detects it, routes a re-run to the Executor, and the clean result passes."""
    clean = run_analysis(scenario("dry_normal"))

    def corrupt_first_attempt(sid, res, attempt):
        if attempt == 1 and res.laps:
            res.pit_loss_total_s += 21.5  # pit loss counted twice in the total
            res.remaining_time_s += 21.5
        return res

    r = run_analysis(scenario("dry_normal"), fault_injector=corrupt_first_attempt)
    first = r["critic_reports"][0]
    assert any(v["status"] == "integrity_failed_rerun" for v in first["verdicts"].values())
    assert first["revision_requests"][0]["target"] == "executor"
    assert r["retry_count"] == 1
    assert r["selected_id"] == clean["selected_id"]
    assert r["recommendation"]["numbers"]["remaining_time_s"] == clean["recommendation"]["numbers"]["remaining_time_s"]


def test_persistent_fault_is_never_accepted(scenario):
    def always_corrupt(sid, res, attempt):
        res.remaining_time_s = float("nan")
        return res

    r = run_analysis(scenario("dry_normal"), fault_injector=always_corrupt)
    assert r["status"] == "review_required"
    assert "NUMERICAL_INTEGRITY" in r["recommendation"]["reason"]
    assert r["retry_count"] <= DEFAULT_CONFIG.max_revisions


def test_retry_limit_is_enforced(scenario):
    cfg = dataclasses.replace(DEFAULT_CONFIG, max_revisions=0)
    r = run_analysis(scenario("high_degradation"), cfg=cfg)
    assert r["retry_count"] == 0
    assert r["status"] == "review_required"
    assert any(e["action"] == "retry_limit_reached" for e in r["log"])
    assert "STINT_TOO_LONG" in r["recommendation"]["reason"]


def test_no_feasible_strategy_returns_review_required(simple_raw):
    simple_raw["target"]["compounds_used"] = ["SOFT"]  # stay-out breaks the two-compound rule
    simple_raw["target"]["available_sets"] = {"SOFT": 0, "MEDIUM": 0, "HARD": 0, "INTERMEDIATE": 0, "WET": 0}
    r = run_analysis(simple_raw)
    assert r["status"] == "review_required" and r["selected_id"] is None
    assert r["recommendation"]["headline"] == "Review required"
    assert "TWO_COMPOUND_RULE" in r["recommendation"]["reason"]
    assert r["retry_count"] <= DEFAULT_CONFIG.max_revisions


def test_log_reflects_real_activity_in_mock_mode(scenario):
    r = run_analysis(scenario("dry_normal"))
    assert r["model_mode"] == "mock"
    assert not any(e["kind"] == "llm" for e in r["log"])  # no fabricated LLM calls
    sims = [e for e in r["log"] if e["action"] == "simulate_strategy"]
    assert len(sims) == len(r["candidates"])
    agents = {e["agent"] for e in r["log"]}
    assert {"Planner", "Race Research", "Strategy Executor", "Strategy Critic", "Pit Wall Operations"} <= agents


def test_ollama_unreachable_falls_back_honestly(scenario):
    from pitwall.llm import OllamaReasoner
    r = run_analysis(scenario("dry_normal"), reasoner=OllamaReasoner(host="http://127.0.0.1:9"))
    assert r["status"] == "recommended" and r["recommendation"]["llm_used"] is False
    unavailable = [e for e in r["log"] if e["action"] == "ai_unavailable"]
    assert {e["agent"] for e in unavailable} == {"Planner", "Pit Wall Operations"}
    assert r["recommendation"]["decided_by"].startswith("Rule") and r["ai"]["used"] is False

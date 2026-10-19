"""Pit Wall Assistant (chat) — routing, grounded answers and what-ifs that run the agents."""
import copy

import pytest

from pitwall.agents.assistant import ChatContext, PitWallAssistant
from pitwall.data_sources import replays as rp
from pitwall.data_sources.fastf1_source import build_race_state
from pitwall.llm import OllamaReasoner
from pitwall.orchestrator import run_analysis
from pitwall.reality import actual_outcome, compare_with_reality
from pitwall.records import DecisionStore


@pytest.fixture(scope="module")
def dry():
    from pitwall.data_sources.synthetic import load_scenario
    raw = load_scenario("dry_normal")
    return raw, run_analysis(raw)


def ask(q, raw, res, **kw):
    return PitWallAssistant().answer(q, ChatContext(raw=raw, question_key="full_strategy", result=res, **kw))


@pytest.mark.parametrize("q,intent", [
    ("hi", "help"), ("What is an undercut?", "glossary"), ("why this call?", "explain_call"),
    ("why not pit now?", "why_not"), ("what about soft tyres?", "why_not"), ("what did the critic reject?", "critic"),
    ("how risky is this?", "risk"), ("who is ahead of us?", "gaps"), ("is rain coming?", "weather"),
    ("how much faster is it?", "numbers"), ("how do the agents work?", "agents"), ("tell me a joke", "fallback"),
])
def test_routing(dry, q, intent):
    raw, res = dry
    assert ask(q, raw, res).intent == intent


def test_answers_come_from_the_analysis(dry):
    raw, res = dry
    call = res["recommendation"]["headline"]
    assert call in ask("why this call?", raw, res).text
    pit_now = next(c for c in res["candidates"] if c["family"] == "pit_now" and c["status"] == "ACCEPTED")
    a = ask("why not pit now?", raw, res).text
    assert pit_now["label"] in a and f"{pit_now['delta_to_best_s']:.2f} s slower" in a
    stay = next(c for c in res["candidates"] if c["family"] == "stay_out")
    assert stay["issue_list"][0] in ask("should we stay out?", raw, res).text


def test_deterministic(dry):
    raw, res = dry
    assert ask("why not pit now?", raw, res).text == ask("why not pit now?", raw, res).text


def test_what_if_runs_agents_without_saving_or_mutating(dry, isolated_decision_log):
    raw, res = dry
    before = copy.deepcopy(raw)
    a = ask("What if it rains in 4 laps?", raw, res)
    assert a.intent == "what_if" and any("five agents" in t for t in a.tools)
    assert "Intermediate" in a.text and "the call changes" in a.text and a.charts and a.scenario_raw
    assert raw == before and DecisionStore().load_all() == []


@pytest.mark.parametrize("q,expect", [
    ("what if a safety car comes out next lap?", "Safety Car on laps 21–24"),
    ("what if there's a safety car on lap 30 for 3 laps?", "Safety Car on laps 30–32"),
    ("what if a VSC comes out next lap?", "VSC on laps 21–22"),
    ("what if it rains from lap 35?", "Rain from lap 35 to the finish"),
    ("what if it rains in 5 laps for 6 laps?", "Rain from lap 25 to lap 30"),
    ("what if tyre wear is 1.8x?", "Tyre wear ×1.8"),
    ("what if tyres wear 50% faster?", "Tyre wear ×1.5"),
    ("what if a pit stop costs 30 seconds?", "Pit stops cost 30 s"),
    ("what if the pit lane is closed on laps 21 and 22?", "Pit lane closed on laps 21, 22"),
])
def test_what_if_parsing(dry, q, expect):
    raw, res = dry
    assert expect in ask(q, raw, res).text


def test_what_if_invalid_is_explained(dry):
    raw, res = dry
    a = ask("what if it rains on lap 90?", raw, res)
    assert "can't run that scenario" in a.text


def test_without_analysis(dry):
    raw, _ = dry
    assert ask("why this call?", raw, None).intent == "no_result"
    a = ask("what if a safety car comes out next lap?", raw, None)
    assert a.intent == "what_if" and any("baseline" in t for t in a.tools)


def test_reality_questions(dry):
    raw, res = dry
    assert "made up" in ask("what really happened?", raw, res).text
    s = rp.load_replay("2024_italian_grand_prix")
    raw2 = build_race_state(s, "LEC", 10, source="replay")
    res2 = run_analysis(raw2)
    calls = []

    def reveal():
        calls.append(1)
        return compare_with_reality(res2, actual_outcome(s, "LEC", 10))
    a = ask("What actually happened in the real race?", raw2, res2, is_real_race=True, reveal=reveal)
    assert calls == [1] and "Team: Box lap 15 for Hard" in a.text and a.tools


def test_stale_note(dry):
    raw, res = dry
    assert "inputs changed" in ask("why this call?", raw, res, stale=True).text


def test_ollama_unavailable_keeps_rule_answer(dry):
    raw, res = dry
    plain = ask("why this call?", raw, res)
    a = PitWallAssistant(reasoner=OllamaReasoner(host="http://127.0.0.1:9")).answer(
        "why this call?", ChatContext(raw=raw, question_key="full_strategy", result=res))
    assert a.text == plain.text and "AI model unavailable" in a.source and "Ollama" in a.source

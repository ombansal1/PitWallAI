"""AI-model mode: the agents talk to an OpenAI-compatible chat API (here a local stand-in server).
Checks that the AI really drives the plan and the call, that the Critic still guards every plan,
and that every failure falls back to the rules with an honest log entry."""
import json
from pathlib import Path

import pytest

from pitwall.agents.assistant import ChatContext, PitWallAssistant
from pitwall.llm import LLMReasoner, MockReasoner, extract_json, find_api_key, make_reasoner
from pitwall.orchestrator import run_analysis

PLAN = {"rationale": ["Mediums are 20 laps old, so test boxing soon", "Hards can reach the flag"],
        "analyses": ["track_status", "tyre_analysis", "gap_analysis"],
        "include_stay_out": False, "include_box_now": True, "delays": [4, 8], "compounds": ["HARD", "SOFTS"],
        "event_stops": [], "wet_transition": False, "return_to_slicks": False, "two_stop": False,
        "custom_plans": [{"stops": [{"lap": 30, "tyre": "HARD"}], "why": "long hard stint"},
                         {"stops": [{"lap": 22, "tyre": "SOFT"}], "why": "aggressive softs"},      # too long: Critic
                         {"stops": [{"lap": 60, "tyre": "HARD"}], "why": "late stop"},             # outside race
                         {"stops": [{"lap": 25, "tyre": "SUPERSOFT"}], "why": "unknown tyre"}]}   # dropped


def options_in(payload):
    user = payload["messages"][-1]["content"]
    return json.loads(user.split("OPTIONS: ", 1)[1].split("\n", 1)[0])


def script(plan=PLAN, choose=1, revise=None, chat="Simple answer from the AI."):
    def handler(payload):
        system = payload["messages"][0]["content"]
        user = payload["messages"][-1]["content"]
        if "Planner agent" in system:
            if "CRITIC FEEDBACK" in user:
                return reply(revise or {"rationale": ["n/a"]})
            return reply(plan)
        if system.startswith("You are Pit Wall Operations"):
            opts = options_in(payload)
            pick = opts[min(choose, len(opts) - 1)]["id"] if isinstance(choose, int) else choose
            return reply({"choice": pick, "why": "less tyre wear at the end", "explanation":
                          ["The hards reach the flag comfortably.", "The gap is small."],
                          "radio": "Box this lap, box. Hards, long stint to the end."})
        if "Convert the user's" in system:
            return reply({"is_scenario": True, "line": "safety car on laps 24-27"})
        return reply(chat)
    return handler


def reply(content):
    from conftest import FakeAI
    return FakeAI.reply(content)


def brain(fake_ai, **kw):
    return LLMReasoner("groq", api_key="test-key", base_url=fake_ai.url, **kw)


def test_ai_writes_plan_critic_guards_it_and_ai_makes_the_call(fake_ai, scenario):
    fake_ai.handler = script()
    r = run_analysis(scenario("dry_normal"), reasoner=brain(fake_ai))
    assert r["status"] == "recommended"
    plan = r["plan_history"][0]
    assert "gpt-oss-120b" in plan["author"] and plan["rationale"][0].startswith("AI: Mediums")
    assert any("unknown tyre" in x for x in plan["rationale"])            # validator explained what it dropped
    spec = plan["candidate_spec"]
    assert spec["include_stay_out"] is True and spec["delays"] == [4, 8]   # reference option kept
    assert spec["compounds"] == ["HARD", "SOFT"]
    fam = {c["id"]: c for c in r["candidates"]}
    assert fam["PIT_L30_HAR"]["family"] == "ai_plan"
    assert fam["PIT_L22_SOF"]["status"] == "REJECTED" and "laps old" in fam["PIT_L22_SOF"]["issues"]
    assert fam["PIT_L60_HAR"]["status"] == "REJECTED" and "outside the race" in fam["PIT_L60_HAR"]["issues"]
    assert not any("SUP" in c for c in fam)
    rec = r["recommendation"]
    assert rec["decided_by"].startswith("AI model") and rec["ai"]["time_cost_s"] > 0   # AI picked the 2nd fastest
    assert rec["selected_id"] != rec["ai"]["fastest_id"] and rec["selected_id"] in rec["ai"]["shortlist"]
    assert rec["risk"]["category"] in ("MEDIUM", "HIGH") and "faster exists" in rec["risk"]["reasons"][0]
    if rec["stops"][0]["lap"] != 21:      # the stand-in's "Box this lap" radio contradicts a later stop
        assert any(e["action"] == "ai_radio_rejected" for e in r["log"]) and "template" not in rec["radio"]
    assert rec["numbers"]["remaining_time_s"] == next(c["remaining_time_s"] for c in r["candidates"]
                                                      if c["id"] == rec["selected_id"])   # numbers from code
    assert r["ai"]["used"] and [c["purpose"] for c in r["ai"]["calls"]] == [
        "Planner: write the plan", "Operations: choose the call"]
    assert fake_ai.requests[0]["auth"] == "Bearer test-key"
    p = fake_ai.requests[0]["payload"]
    assert p["model"] == "openai/gpt-oss-120b" and p["response_format"] == {"type": "json_object"}
    assert p["reasoning_effort"] == "low"


def test_ai_choice_outside_shortlist_is_refused(fake_ai, scenario):
    fake_ai.handler = script(choose="STAY_OUT_MADE_UP")
    r = run_analysis(scenario("dry_normal"), reasoner=brain(fake_ai))
    rec = r["recommendation"]
    assert rec["selected_id"] == rec["ai"]["fastest_id"]
    assert any(e["action"] == "ai_choice_rejected" for e in r["log"])


def test_unusable_ai_reply_falls_back_to_rules(fake_ai, scenario):
    fake_ai.handler = lambda p: reply("Sure! I think you should pit. (no JSON here)")
    r = run_analysis(scenario("dry_normal"), reasoner=brain(fake_ai))
    rules = run_analysis(scenario("dry_normal"))
    assert r["status"] == "recommended" and r["plan_history"][0]["author"] == "rules"
    assert r["recommendation"]["headline"] == rules["recommendation"]["headline"]
    assert [e["agent"] for e in r["log"] if e["action"] == "ai_failed"] == ["Planner", "Pit Wall Operations"]
    assert r["ai"]["used"] is False


def test_ai_revision_after_critic_feedback(fake_ai, scenario):
    raw = scenario("dry_normal")
    raw["neutral_periods"] = [{"kind": "VSC", "start": 26, "end": 27}]
    revise = {"rationale": ["Use the VSC"], "event_stops": [{"lap": 26, "why": "cheap stop under the VSC"}],
              "custom_plans": [{"stops": [{"lap": 26, "tyre": "HARD"}], "why": "VSC stop"}]}
    fake_ai.handler = script(revise=revise)
    r = run_analysis(raw, reasoner=brain(fake_ai))
    assert r["critic_reports"][0]["revision_requests"][0]["code"] == "MISSING_SC_OPTION"
    assert len(r["plan_history"]) == 2 and "gpt-oss" in r["plan_history"][1]["author"]
    assert any(x.startswith("v2: AI added a stop on lap 26") for x in r["plan_history"][1]["rationale"])
    assert [c["purpose"] for c in r["ai"]["calls"]] == ["Planner: write the plan", "Planner: revise plan v1",
                                                        "Operations: choose the call"]
    assert any(c["first_pit_lap"] == 26 for c in r["candidates"])


def test_rate_limit_is_retried_then_succeeds(fake_ai, scenario):
    calls = {"n": 0}
    inner = script()

    def handler(p):
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, {"error": {"message": "rate limit"}}, {"retry-after": "0"}
        return inner(p)
    fake_ai.handler = handler
    r = run_analysis(scenario("dry_normal"), reasoner=brain(fake_ai))
    assert r["ai"]["calls"][0]["ok"] and "gpt-oss" in r["plan_history"][0]["author"]


def test_rejected_key_and_missing_key(fake_ai, scenario):
    fake_ai.handler = lambda p: (401, {"error": {"message": "invalid key"}}, {})
    r = run_analysis(scenario("dry_normal"), reasoner=brain(fake_ai))
    assert r["status"] == "recommended"
    assert any("API key was rejected" in e["detail"] for e in r["log"] if e["action"] == "ai_failed")
    nokey = LLMReasoner("groq", api_key=None, base_url=fake_ai.url)
    assert nokey.available() is False and "no API key" in nokey.last_error
    assert nokey.test_connection()["ok"] is False


def test_test_connection_and_ollama(fake_ai):
    fake_ai.handler = lambda p: reply({"ok": True, "team_radio": "box box box box box"})
    out = brain(fake_ai).test_connection()
    assert out["ok"] and "Connected to openai/gpt-oss-120b" in out["message"]
    root = fake_ai.url[:-3]
    oll = make_reasoner("ollama", model="llama3.2", base_url=fake_ai.url)
    assert oll.available() and oll.test_connection()["ok"]
    assert "response_format" not in fake_ai.requests[-1]["payload"]      # Ollama: JSON asked in the prompt
    missing = make_reasoner("ollama", model="mistral", base_url=root + "/v1")
    assert missing.available() is False and "ollama pull mistral" in missing.last_error
    assert isinstance(make_reasoner("rules"), MockReasoner)


def test_chat_uses_ai_and_reads_odd_scenarios(fake_ai, scenario):
    fake_ai.handler = script(chat="The agents box now for hards because the mediums are worn.")
    raw = scenario("dry_normal")
    res = run_analysis(raw)
    bot = PitWallAssistant(reasoner=brain(fake_ai))
    ctx = ChatContext(raw=raw, question_key="full_strategy", result=res)
    a = bot.answer("why this call?", ctx)
    assert a.text.startswith("The agents box now") and "AI model" in a.source and a.intent == "explain_call"
    b = bot.answer("pace car on track from 24 through 27, then what?", ctx)
    assert b.intent == "what_if" and b.charts and b.scenario_raw
    sc = [p for p in b.scenario_raw["neutral_periods"] if p["kind"] == "SC"] or [
        {"start": b.scenario_raw["sc_start_lap"], "end": b.scenario_raw["sc_end_lap"]}]
    assert (sc[0]["start"], sc[0]["end"]) == (24, 27)
    assert any("AI model rewrote your scenario" in t for t in b.tools)


def test_extract_json_and_key_lookup(tmp_path, monkeypatch):
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure, here: {"a": "x}y", "b": {"c": 2}} thanks') == {"a": "x}y", "b": {"c": 2}}
    assert extract_json("no json") is None and extract_json('[1, 2]') is None
    env = tmp_path / ".env"
    env.write_text('# comment\nGROQ_API_KEY="from-dotenv"\n')
    assert find_api_key("groq", dotenv_path=env) == ("from-dotenv", ".env file")
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert find_api_key("groq", dotenv_path=env) == ("from-env", "environment variable")
    assert find_api_key("groq", secrets={"GROQ_API_KEY": "from-secrets"}, dotenv_path=env)[1] == "secrets.toml"
    assert find_api_key("ollama") == (None, "not needed")
    assert find_api_key("gemini", dotenv_path=Path(tmp_path / "missing")) == (None, "not set")


@pytest.mark.parametrize("q", ["compound_choice", "pit_now_vs_stay_out"])
def test_narrow_questions_keep_rule_options(fake_ai, scenario, q):
    fake_ai.handler = script()
    r = run_analysis(scenario("dry_normal"), q, reasoner=brain(fake_ai))
    assert r["plan_history"][0]["author"] == "rules"
    assert any(e["action"] == "rule_plan_kept" for e in r["log"])


def test_radio_that_contradicts_the_plan_is_replaced(fake_ai, scenario):
    fake_ai.handler = script(choose=1)            # stand-in radio says "Box this lap" for every choice
    r = run_analysis(scenario("dry_normal"), reasoner=brain(fake_ai))
    rec = r["recommendation"]
    boxes_now = rec["stops"] and rec["stops"][0]["lap"] == 21
    if boxes_now:
        assert rec["radio"]["text"].startswith("Box this lap")
    else:
        assert "template" not in rec["radio"] and any(e["action"] == "ai_radio_rejected" for e in r["log"])


def test_radio_mismatch_rules():
    from pitwall.agents.operations import radio_mismatch

    class WS:
        race_state = type("S", (), {"first_lap": 21})()
    rec = {"stops": [{"lap": 29, "compound": "HARD"}]}
    assert radio_mismatch("Plan is box lap 29 for hards", rec, WS) is None
    assert "lap 30" in radio_mismatch("Box lap 30 for hards", rec, WS)
    assert "box now" in radio_mismatch("Box, box. Hards.", rec, WS)
    assert "different tyre" in radio_mismatch("Extend, then softs", rec, WS)
    assert "no more stops" in radio_mismatch("Stay out to the end", rec, WS)

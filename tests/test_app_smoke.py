"""Streamlit smoke tests using streamlit.testing (headless, no browser), page by page."""
from pathlib import Path

import pytest

from pitwall.records import DecisionStore

APP = str(Path(__file__).resolve().parent.parent / "app.py")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
PAGES = ["race_control", "agents", "chat", "strategy_lab", "model_lab", "setup", "records", "about"]


def button(at, label):
    return next(b for b in at.button if b.label == label)


_SWITCH = {}


def _switch_supported() -> bool:
    """Older Streamlit test runners (e.g. 1.60) ignore switch_page for st.navigation pages; the app itself
    works there (checked in a browser), so page-switching tests are skipped on those versions."""
    if "ok" not in _SWITCH:
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.switch_page("views/about.py").run()
        _SWITCH["ok"] = any("About PitWall AI" in m.value for m in at.markdown)
    return _SWITCH["ok"]


def go(at, page):
    if page != "race_control" and not _switch_supported():
        pytest.skip("this Streamlit version's AppTest can't switch top-navigation pages")
    return at.switch_page(f"views/{page}.py").run()


def start(timeout=120):
    return AppTest.from_file(APP, default_timeout=timeout).run()


def analyze(at):
    go(at, "race_control")
    button(at, "Analyze Strategy").click().run()
    assert not at.exception
    return at


def test_every_page_renders_before_and_after_an_analysis(isolated_decision_log):
    at = start()
    for p in PAGES:
        go(at, p)
        assert not at.exception, p
    analyze(at)
    assert any("completed and recorded" in s.value for s in at.success)
    for p in PAGES:
        go(at, p)
        assert not at.exception, p


def test_app_end_to_end(isolated_decision_log):
    at = analyze(start())
    assert len(DecisionStore().load_all()) == 1
    at.run()  # plain rerun: no duplicate record, no re-analysis
    assert len(DecisionStore().load_all()) == 1
    go(at, "strategy_lab")
    button(at, "Apply rain forecast").click().run()
    assert not at.exception
    assert any("Event applied" in s.value for s in at.success)
    assert len(DecisionStore().load_all()) == 2  # same inputs as last analysis -> no extra baseline
    at.run()
    assert len(DecisionStore().load_all()) == 2  # event not re-applied on rerun
    assert at.session_state["w_rain_start"] == 26
    assert len(at.session_state["events_log"]) == 1
    assert at.session_state["comparison"]["changed"] is True


def test_inputs_survive_page_switches(isolated_decision_log):
    at = start()
    go(at, "setup")
    at.number_input(key="w_pit_loss").set_value(25.0).run()
    for p in ("agents", "chat", "model_lab", "records", "race_control"):
        go(at, p)
    assert at.session_state["w_pit_loss"] == 25.0
    analyze(at)
    at.run()
    assert at.session_state["last_result"]["input_snapshot"]["pit_loss_s"] == 25.0
    assert at.session_state["w_pit_loss"] == 25.0 and at.session_state["w_total_laps"] == 56
    assert not any("out of date" in w.value for w in at.warning)


def test_app_marks_stale_and_reports_invalid_input(isolated_decision_log):
    at = analyze(start())
    go(at, "setup")
    at.number_input(key="w_pit_loss").set_value(30.0).run()
    assert any("out of date" in w.value for w in at.warning)
    at.number_input(key="w_completed_laps").set_value(60).run()
    analyze(at)
    assert any("Invalid inputs" in e.value for e in at.error)
    assert len(DecisionStore().load_all()) == 1  # invalid run is not recorded


def test_app_scenario_switch(isolated_decision_log):
    at = start()
    at.selectbox(key="w_scenario").set_value("high_degradation").run()
    assert not at.exception
    assert at.session_state["w_deg"] == pytest.approx(1.7)
    analyze(at)
    assert "Two-stop" in at.session_state["last_result"]["recommendation"]["headline"]


def test_app_fastf1_failure_offers_synthetic_fallback(isolated_decision_log, monkeypatch):
    from pitwall.data_sources import fastf1_source as ff

    def offline(year, cache_dir=None):
        raise ff.DataSourceError("Could not load the schedule (offline)")
    monkeypatch.setattr(ff, "fastf1_available", lambda: True)
    monkeypatch.setattr(ff, "get_race_schedule", offline)
    at = start()
    at.radio(key="w_source").set_value("FastF1 download (optional)").run()
    assert not at.exception
    assert any("offline" in e.value for e in at.error)
    button(at, "Use synthetic fallback").click().run()
    assert not at.exception
    assert at.session_state["raw_base"]["data_source"] == "synthetic_fallback"
    analyze(at)
    assert at.session_state["last_result"]["status"] == "recommended"


def test_fastf1_option_hidden_when_not_installed(isolated_decision_log, monkeypatch):
    from pitwall.data_sources import fastf1_source as ff
    monkeypatch.setattr(ff, "fastf1_available", lambda: False)
    at = start()
    assert at.radio(key="w_source").options == ["Example scenarios", "Real races (2019–2026)"]


def test_switching_the_ai_model_off_marks_results_stale(isolated_decision_log, fake_ai, monkeypatch):
    from test_ai_agents import script
    monkeypatch.setenv("PITWALL_AI_BASE_URL", fake_ai.url)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    fake_ai.handler = script()
    at = analyze(start())
    assert not any("out of date" in w.value for w in at.warning)
    go(at, "model_lab")
    at.toggle(key="ai_use").set_value(False).run()
    assert not at.exception
    assert at.session_state["ai_provider"] == "rules"
    assert any("out of date" in w.value for w in at.warning)


def test_model_lab_has_no_key_or_provider_settings(isolated_decision_log, monkeypatch):
    at = start()
    go(at, "model_lab")
    assert not at.exception
    assert not at.text_input                                   # nowhere to type a key
    assert all(sb.key != "ai_provider" for sb in at.selectbox)
    assert not any(b.label == "Test connection" for b in at.button)
    assert all(t.key != "ai_use" for t in at.toggle)          # no AI configured: no switch either
    assert any("No AI model is set up" in i.value for i in at.info)


def test_configured_key_shows_model_and_switch(isolated_decision_log, fake_ai, monkeypatch):
    monkeypatch.setenv("PITWALL_AI_BASE_URL", fake_ai.url)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    at = start()
    go(at, "model_lab")
    assert not at.exception
    assert at.toggle(key="ai_use").value is True
    assert any("openai/gpt-oss-120b" in m.value for m in at.markdown)
    assert not at.text_input


def test_analysis_with_ai_model_and_ml_tyres(isolated_decision_log, fake_ai, monkeypatch):
    from test_ai_agents import script
    monkeypatch.setenv("PITWALL_AI_BASE_URL", fake_ai.url)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    fake_ai.handler = script()
    at = start()
    assert at.session_state["ai_provider"] == "groq"          # a configured key switches the AI on
    at.radio(key="w_tyre_model").set_value("ML model").run()
    analyze(at)
    res = at.session_state["last_result"]
    assert res["ai"]["used"] and res["recommendation"]["decided_by"].startswith("AI model")
    assert res["input_snapshot"]["tyre_model"]["kind"] == "ml"
    go(at, "agents")
    assert not at.exception
    assert any("written by the AI" in m.value for m in at.markdown)
    go(at, "race_control")
    assert any("The AI model explains" in m.value for m in at.markdown)


def test_app_review_required_renders(isolated_decision_log):
    at = start()
    go(at, "setup")
    at.multiselect(key="w_t_used").set_value(["MEDIUM"]).run()
    for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        at.number_input(key=f"w_t_sets_{c}").set_value(0).run()
    analyze(at)
    assert at.session_state["last_result"]["status"] == "review_required"
    assert len(DecisionStore().load_all()) == 1


def _load_real(at, year, race, driver, lap, hindsight=False):
    at.radio(key="w_source").set_value("Real races (2019–2026)").run()
    at.selectbox(key="rr_year").set_value(year).run()
    at.selectbox(key="rr_race").set_value(race).run()
    at.selectbox(key="rr_driver").set_value(driver).run()
    at.slider(key="rr_lap").set_value(lap).run()
    if hindsight:
        at.toggle(key="rr_real_events").set_value(True).run()
    button(at, "Load this moment").click().run()
    assert not at.exception


def test_app_real_race_reveal_and_chat(isolated_decision_log):
    at = start()
    _load_real(at, 2024, "2024_italian_grand_prix", "LEC", 10)
    assert at.session_state["raw_base"]["target"]["name"] == "Charles Leclerc"
    assert not any(b.label == "Reveal what really happened" for b in at.button)  # nothing to reveal yet
    analyze(at)
    go(at, "race_control")
    button(at, "Reveal what really happened").click().run()
    assert not at.exception
    cmp = at.session_state["reality"]
    assert cmp["team_call"] == "Box lap 15 for Hard" and cmp["finish_position"] == 1
    at.run()  # rerun must not add another reality record
    kinds = [r.get("record_type", "decision") for r in DecisionStore().load_all()]
    assert kinds == ["decision", "reality_check"]
    go(at, "chat")
    at.chat_input(key="chat_box").set_value("What if a safety car comes out next lap?").run()
    assert not at.exception
    assert at.session_state["chat"][-1]["intent"] == "what_if"
    assert len(DecisionStore().load_all()) == 2  # what-ifs are never saved
    go(at, "records")
    assert not at.exception


def test_app_chat_on_synthetic(isolated_decision_log):
    at = start()
    go(at, "chat")
    button(at, "Why this call?").click().run()
    assert at.session_state["chat"][-1]["intent"] == "no_result"
    analyze(at)
    go(at, "chat")
    button(at, "Why not pit now?").click().run()
    assert not at.exception
    assert "slower" in at.session_state["chat"][-1]["text"]
    button(at, "Clear chat").click().run()
    assert at.session_state["chat"] == []


def test_app_hindsight_toggle_gives_agents_real_events(isolated_decision_log):
    at = start()
    _load_real(at, 2024, "2024_british_grand_prix", "HAM", 20, hindsight=True)
    base = at.session_state["raw_base"]
    assert base["replay"]["real_events"] is True and base["rain_start_lap"] == 27
    analyze(at)
    assert "Intermediate" in at.session_state["last_result"]["recommendation"]["headline"]
    go(at, "race_control")
    button(at, "Reveal what really happened").click().run()
    assert at.session_state["reality"]["team_call"] == "Box lap 27 for Intermediate, lap 38 for Soft"


def test_app_chat_scenario_can_be_loaded(isolated_decision_log):
    at = start()
    go(at, "chat")
    at.chat_input(key="chat_box").set_value(
        "After lap 25 of 56 on 20-lap-old mediums, VSC on laps 27-28 and rain from lap 40 to 48").run()
    assert not at.exception
    msg = at.session_state["chat"][-1]
    assert msg["intent"] == "what_if" and msg["charts"] and msg["scenario_raw"]
    assert DecisionStore().load_all() == []                # scenarios are not saved
    button(at, "Load this scenario into the race inputs").click().run()
    assert not at.exception
    assert at.session_state["w_completed_laps"] == 25 and at.session_state["w_t_age"] == 20
    assert at.session_state["periods_df"]["Type"].tolist() == ["VSC", "Rain"]
    analyze(at)
    assert at.session_state["last_result"]["status"] == "recommended"


def test_app_vsc_preset_runs_live_in_the_lab(isolated_decision_log):
    at = start()
    assert at.session_state["w_live"] is True
    go(at, "strategy_lab")
    button(at, "🟡 VSC next lap").click().run()
    assert not at.exception
    res = at.session_state["last_result"]
    assert res["trigger_event"]["type"] == "vsc"
    assert res["input_snapshot"]["neutral_periods"] == [{"kind": "VSC", "start": 21, "end": 22}]
    assert at.session_state["pending_run"] is None
    assert len(DecisionStore().load_all()) == 2            # baseline + event, each once


def test_ml_lab_predictor(isolated_decision_log):
    at = start()
    go(at, "model_lab")
    assert not at.exception
    from pitwall.ml.tyre_model import load_model
    if load_model() is None:      # saved with another scikit-learn version: the page offers a retrain
        assert any(b.label.startswith("Train the ML model now") for b in at.button)
        return
    at.slider(key="ml_try_age").set_value(30).run()
    assert not at.exception
    assert any("Predicted lap vs typical" in m.value for m in at.markdown)

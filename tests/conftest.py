import copy
import dataclasses
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pitwall.config import COMPOUNDS, DEFAULT_CONFIG, CompoundParams  # noqa: E402
from pitwall.data_sources.synthetic import load_scenario  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_decision_log(tmp_path, monkeypatch):
    """Never write test records into the project's data folder."""
    monkeypatch.setenv("PITWALL_DECISION_LOG", str(tmp_path / "decision_log.jsonl"))
    monkeypatch.setenv("PITWALL_LIVE_DELAY", "0")   # live agent view runs at full speed in tests
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "PITWALL_AI_PROVIDER"):
        monkeypatch.delenv(k, raising=False)            # tests never use a real AI model
    monkeypatch.setenv("PITWALL_DOTENV", str(tmp_path / "no.env"))
    yield tmp_path / "decision_log.jsonl"


@pytest.fixture
def simple_cfg():
    """Round-number parameters so race times can be calculated by hand."""
    comps = dict(COMPOUNDS)
    comps["SOFT"] = CompoundParams("SOFT", True, 0.00, 0.10, 100, 0.0, 50)
    comps["MEDIUM"] = CompoundParams("MEDIUM", True, 0.35, 0.05, 100, 0.0, 50)
    comps["HARD"] = CompoundParams("HARD", True, 0.70, 0.02, 100, 0.0, 60)
    return dataclasses.replace(DEFAULT_CONFIG, compounds=comps, traffic_enabled=False)


def _driver(code, gap=0.0, pace=90.0, compound="SOFT", age=2, used=None, sets=None):
    return {"code": code, "name": code, "team": "T", "gap_to_leader_s": gap, "baseline_pace_s": pace,
            "compound": compound, "tyre_age_laps": age, "compounds_used": used or [compound],
            "available_sets": sets or {"SOFT": 1, "MEDIUM": 1, "HARD": 1, "INTERMEDIATE": 1, "WET": 1},
            "stops_made": 0}


@pytest.fixture
def simple_raw():
    """10-lap race, 5 laps completed, target on 2-lap-old SOFT, 90.0 s baseline, 20 s pit loss."""
    return {
        "scenario_id": "unit", "scenario_name": "Unit test", "track_name": "Test",
        "total_laps": 10, "completed_laps": 5, "pit_loss_s": 20.0, "track_condition": "dry",
        "rain_start_lap": None, "rain_end_lap": None, "sc_start_lap": None, "sc_end_lap": None,
        "deg_multiplier": 1.0, "pit_closed_laps": [],
        "target": _driver("TGT", gap=5.0, used=["MEDIUM", "SOFT"]),
        "competitors": [_driver("AAA", gap=0.0, pace=90.2, compound="MEDIUM", used=["MEDIUM", "HARD"]),
                        _driver("BBB", gap=12.0, pace=90.1, compound="HARD", used=["HARD", "SOFT"])],
        "data_source": "synthetic", "data_source_label": "unit", "provenance": {}, "events": [], "notes": [],
    }


@pytest.fixture
def make_driver():
    return _driver


@pytest.fixture
def scenario():
    return lambda sid: copy.deepcopy(load_scenario(sid))


class FakeAI:
    """A local stand-in for an OpenAI-compatible chat API (Groq/Gemini/Ollama all use this format).
    `handler(payload) -> (status, body_dict_or_text, headers)` decides each reply; every request is kept."""

    def __init__(self):
        self.requests = []
        self.handler = None
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"models": [{"name": "llama3.2:latest"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"payload": payload, "auth": self.headers.get("Authorization")})
                status, body, headers = fake.handler(payload)
                if isinstance(body, dict):
                    body = json.dumps(body)
                data = body.encode()
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def reply(content, tokens=321):
        if not isinstance(content, str):
            content = json.dumps(content)
        return 200, {"choices": [{"message": {"role": "assistant", "content": content}}],
                     "usage": {"total_tokens": tokens}}, {}

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_ai():
    f = FakeAI()
    yield f
    f.close()

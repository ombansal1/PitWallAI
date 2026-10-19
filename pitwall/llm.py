"""AI model ("agent brain") backends.

* MockReasoner: no AI model. Agents use their deterministic rules and templates (works offline,
  used by the tests, and the automatic fallback whenever the AI model can't be reached).
* LLMReasoner: a real large language model behind an OpenAI-compatible chat API. Three free options:
    - Groq (cloud, free plan): OpenAI's open-weight gpt-oss-120b. Needs a free API key.
    - Google Gemini (cloud, free tier): Gemini Flash. Needs a free API key.
    - Ollama (runs on your own PC, free): e.g. llama3.2. No key, but the model must be downloaded.

What the AI model does in PitWall (see pitwall/agents/*):
  Planner - chooses which strategies to test and revises the plan when the Critic objects.
  Pit Wall Operations - picks the call from the Critic-approved shortlist and explains it.
  Chat - understands free-text questions and scenarios and answers in plain English.
What it never does: lap-time maths (Python simulation) and rule checks (the Critic). Every AI answer is
parsed and validated; anything unusable falls back to the rules, and that is written in the activity log.
Only the standard library is used for HTTP, so no extra package is needed.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    default_model: str
    key_env: Optional[str]          # environment variable / secret holding the API key (None = no key)
    key_url: str                    # where to get a key (or the model)
    models: Tuple[str, ...]
    note: str


PROVIDERS: Dict[str, Provider] = {
    "groq": Provider(
        "groq", "Groq (free cloud plan)", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b",
        "GROQ_API_KEY", "https://console.groq.com/keys",
        ("openai/gpt-oss-120b", "openai/gpt-oss-20b"),
        "Runs OpenAI's open-weight gpt-oss-120b on Groq's servers. Free plan: about 30 requests a minute "
        "and 1,000 a day. Sign up at console.groq.com, then create an API key."),
    "gemini": Provider(
        "gemini", "Google Gemini (free tier)", "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-flash-latest", "GEMINI_API_KEY", "https://aistudio.google.com/apikey",
        ("gemini-flash-latest", "gemini-flash-lite-latest"),
        "Google's Gemini Flash through its OpenAI-compatible endpoint. Free tier with daily limits; "
        "Google may use free-tier prompts to improve its products. Key from Google AI Studio."),
    "ollama": Provider(
        "ollama", "Ollama (free, on your PC)", "http://localhost:11434/v1", "llama3.2", None,
        "https://ollama.com/download", ("llama3.2", "qwen2.5", "gpt-oss:20b"),
        "Runs an open model on your own computer, fully offline. Install Ollama, run `ollama pull llama3.2`, "
        "keep Ollama open. Slower and less capable than the cloud models on a normal laptop."),
}
DEFAULT_PROVIDER = "groq"


_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open(req, timeout: float):
    """Local servers (Ollama, test stubs) are reached directly, never through a web proxy."""
    url = req if isinstance(req, str) else req.full_url
    host = urllib.parse.urlparse(url).hostname or ""
    if host in ("localhost", "127.0.0.1", "::1"):
        return _NO_PROXY.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


# ---------------------------------------------------------------------------- keys
def read_dotenv(path: Path) -> Dict[str, str]:
    """Minimal .env reader (KEY=VALUE lines, # comments, optional quotes). Never overrides the environment."""
    out: Dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def find_api_key(provider: str, secrets: Optional[Dict[str, Any]] = None,
                 dotenv_path: Optional[Path] = None) -> Tuple[Optional[str], str]:
    """Look for a provider's key: Streamlit secrets → environment variable → .env file.
    Returns (key or None, where it was found)."""
    p = PROVIDERS[provider]
    if not p.key_env:
        return None, "not needed"
    for name in (p.key_env, p.key_env.lower()):
        try:
            if secrets and secrets.get(name):
                return str(secrets[name]).strip(), "secrets.toml"
        except Exception:  # st.secrets raises when no secrets file exists
            pass
    if os.environ.get(p.key_env):
        return os.environ[p.key_env].strip(), "environment variable"
    if dotenv_path is not None:
        v = read_dotenv(dotenv_path).get(p.key_env)
        if v:
            return v, ".env file"
    return None, "not set"


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first JSON object in a model reply (tolerates code fences and chatter around it)."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except ValueError:
        pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(t[start:i + 1])
                        return v if isinstance(v, dict) else None
                    except ValueError:
                        break
        start = t.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------- reasoners
class MockReasoner:
    """No AI model: the agents' deterministic rules decide."""
    mode = "mock"
    provider = "rules"
    model = "rules"
    label = "Rules only (no AI model)"
    short = "Rules"
    last_error: Optional[str] = None

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def available(self) -> bool:
        return True

    def begin_run(self) -> None:
        self.calls = []

    def ask_json(self, system: str, user: str, purpose: str = "", max_tokens: int = 900) -> Optional[Dict]:
        return None

    def write(self, system: str, user: str, purpose: str = "", max_tokens: int = 500) -> Optional[str]:
        return None

    def rewrite(self, prompt: str) -> Optional[str]:
        return None


class LLMReasoner:
    """A large language model behind an OpenAI-compatible /chat/completions endpoint."""
    mode = "ai"

    def __init__(self, provider: str = DEFAULT_PROVIDER, model: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 45.0, max_wait_s: float = 12.0):
        self.provider = provider if provider in PROVIDERS else DEFAULT_PROVIDER
        p = PROVIDERS[self.provider]
        self.model = (model or p.default_model).strip()
        self.api_key = (api_key or "").strip() or None
        self.base_url = (base_url or p.base_url).rstrip("/")
        self.timeout, self.max_wait_s = timeout, max_wait_s
        self.label = f"AI model: {self.model} via {p.label.split(' (')[0]}"
        self.short = f"{self.model} ({p.label.split(' (')[0]})"
        self.last_error: Optional[str] = None
        self.calls: List[Dict[str, Any]] = []
        self._ok_cache: Optional[bool] = None

    # ----------------------------------------------------------------- status
    @property
    def needs_key(self) -> bool:
        return PROVIDERS[self.provider].key_env is not None

    def available(self) -> bool:
        """Cheap check before a run: is a key set (cloud) / is the server up (Ollama)?"""
        if self.needs_key:
            if not self.api_key:
                self.last_error = f"no API key for {PROVIDERS[self.provider].label}"
                return False
            return True
        if self._ok_cache is not None:
            return self._ok_cache
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        try:
            with _open(f"{root}/api/tags", timeout=2.0) as r:
                tags = json.loads(r.read().decode())
            names = {m.get("name", "").split(":")[0] for m in tags.get("models", [])}
            if self.model.split(":")[0] not in names:
                self.last_error = f"model '{self.model}' is not downloaded in Ollama (run: ollama pull {self.model})"
                self._ok_cache = False
            else:
                self._ok_cache = True
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.last_error = f"Ollama is not running at {root} ({e})"
            self._ok_cache = False
        return self._ok_cache

    def begin_run(self) -> None:
        self.calls = []

    # ----------------------------------------------------------------- HTTP
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", "User-Agent": "PitWall-AI/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        with _open(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False, max_tokens: int = 900,
             purpose: str = "") -> Optional[str]:
        """One chat completion. Returns the text, or None (reason in last_error). Retries once on a
        rate limit (waiting what the server asks, up to max_wait_s) or a temporary server error."""
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.2,
                                   "max_tokens": max_tokens}
        if json_mode and self.provider != "ollama":
            payload["response_format"] = {"type": "json_object"}
        if self.provider == "groq" and "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"     # keeps answers fast and inside the free token budget
        t0, err, data = time.time(), None, None
        for attempt in (1, 2):
            try:
                data = self._post(payload)
                err = None
                break
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode()[:300]
                except Exception:
                    pass
                if e.code in (401, 403):
                    err = "the API key was rejected (check it was copied correctly)"
                    break
                if e.code == 404:
                    err = f"model '{self.model}' not found at {PROVIDERS[self.provider].label}"
                    break
                if e.code == 400 and "response_format" in payload and attempt == 1:
                    payload.pop("response_format")      # some models don't support JSON mode
                    continue
                if e.code == 429 or e.code >= 500:
                    wait = 2.0
                    try:
                        wait = float(e.headers.get("retry-after", wait))
                    except (TypeError, ValueError):
                        pass
                    err = ("rate limit reached (free plan); try again in a minute" if e.code == 429
                           else f"server error {e.code}")
                    if attempt == 1 and wait <= self.max_wait_s:
                        time.sleep(max(0.5, wait))
                        continue
                    break
                err = f"HTTP {e.code}: {body or e.reason}"
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                err = f"could not reach {PROVIDERS[self.provider].label} ({getattr(e, 'reason', e)})"
                break
            except ValueError as e:
                err = f"unreadable reply ({e})"
                break
        latency = round(time.time() - t0, 2)
        text, tokens = None, None
        if data is not None:
            try:
                text = (data["choices"][0]["message"].get("content") or "").strip()
                tokens = (data.get("usage") or {}).get("total_tokens")
            except (KeyError, IndexError, TypeError, AttributeError):
                err = "reply had an unexpected format"
            if text == "" and err is None:
                err = "empty reply"
        self.last_error = err
        self.calls.append({"purpose": purpose or "chat", "ok": err is None, "latency_s": latency,
                           "tokens": tokens, "error": err, "model": self.model})
        return text if err is None else None

    # ----------------------------------------------------------------- helpers used by the agents
    def ask_json(self, system: str, user: str, purpose: str = "", max_tokens: int = 900) -> Optional[Dict]:
        text = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         json_mode=True, max_tokens=max_tokens, purpose=purpose)
        if text is None:
            return None
        data = extract_json(text)
        if data is None:
            self.last_error = "the reply was not valid JSON"
            if self.calls:
                self.calls[-1].update(ok=False, error=self.last_error)
        return data

    def write(self, system: str, user: str, purpose: str = "", max_tokens: int = 500) -> Optional[str]:
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         max_tokens=max_tokens, purpose=purpose)

    def rewrite(self, prompt: str) -> Optional[str]:
        return self.write("You are a concise Formula 1 race engineer.", prompt, purpose="rewrite")

    def test_connection(self) -> Dict[str, Any]:
        """Send one tiny request and report what happened (used by the 'Test connection' button)."""
        if not self.available():
            return {"ok": False, "message": self.last_error}
        data = self.ask_json("Reply with JSON only.", 'Return {"ok": true, "team_radio": "<five words>"}',
                             purpose="connection test", max_tokens=200)
        call = self.calls[-1] if self.calls else {}
        if data is None:
            return {"ok": False, "message": self.last_error or "no reply"}
        return {"ok": True, "message": f"Connected to {self.model} in {call.get('latency_s', 0):.1f} s. "
                                       f"Reply: {json.dumps(data)[:120]}"}


class OllamaReasoner(LLMReasoner):
    """Backwards-compatible shortcut for a local Ollama model."""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434", timeout: float = 60.0):
        super().__init__("ollama", model=model, base_url=host.rstrip("/") + "/v1", timeout=timeout)


def make_reasoner(provider: str, model: Optional[str] = None, api_key: Optional[str] = None,
                  base_url: Optional[str] = None):
    """Rules when provider == 'rules', otherwise an LLMReasoner."""
    if provider in (None, "", "rules", "mock"):
        return MockReasoner()
    return LLMReasoner(provider, model=model, api_key=api_key, base_url=base_url)

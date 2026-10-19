# PitWall AI — Agentic AI for Formula 1 race strategy (proof of concept)

Five collaborating agents decide the pit-stop call for a car in a race: **Planner → Race Research → Strategy
Executor → Strategy Critic → Pit Wall Operations**. A free AI model (Groq's gpt-oss-120b, or Gemini / Ollama)
drives the Planner and the final decision; a Python simulator does every lap-time calculation; the Critic's rules
check every plan and can send work back. A machine-learning tyre model trained on ~139,000 real laps can replace
the hand-written tyre formula. Works fully offline with rules if no AI model is set up.

## Run it (Windows Command Prompt, Python 3.11–3.13)

From the project folder (the one containing `app.py`):

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-optional.txt
python -m pytest
streamlit run app.py
```

Open http://localhost:8501. Stop with `Ctrl+C`. (`requirements-optional.txt` adds FastF1; leave it out if you only
want the built-in races.)

## Switch on the free AI model (recommended)

1. Go to **console.groq.com**, sign in, open **API Keys → Create API Key**, copy the key (starts with `gsk_`).
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste the key:
   `GROQ_API_KEY = "gsk_..."`
3. Restart the app. The header shows **AI: openai/gpt-oss-120b**. Use **Model Lab → Test connection** to check.

You can also paste the key on the Model Lab page (kept for that browser session only). Alternatives on the same
page: Google Gemini (key from aistudio.google.com) or Ollama (free, runs on your PC, `ollama pull llama3.2`).
**Never put your key in the ZIP you hand in.** If the AI model is missing, slow or rate-limited, every agent
falls back to its rules and the activity log says so.

## The pages

| Page | What it's for |
|---|---|
| **Race Control** | Pick a race moment (example scenario or one of 162 real races), choose the tyre model, press **Analyze Strategy**; see the call, the AI's explanation, the team radio, and "what really happened" for real races. |
| **Agents** | Watch the five agents work live; pipeline diagram with each agent's brain (AI model / rules / Python); expandable *given · did · decided · passed on* panels; every AI call; full activity log. |
| **Ask the Pit Wall** | Chat. Ask why, compare options, or describe a scenario ("after lap 20 of 56 on 18-lap-old mediums, VSC on laps 24–26, rain from lap 35 to 45"); it runs all five agents and answers in plain English with a race-timeline chart. |
| **Strategy Lab** | One-click race events (Safety Car, VSC, rain, tyre wear) with before/after comparison; every option tested, with charts. |
| **Model Lab** | AI model settings and guard rails; the ML tyre model (accuracy on unseen races, learned wear curves, try-the-model); agents vs real teams evaluation. |
| **Race Setup** | Every input, editable, with data provenance. |
| **Records** | Saved decision records, KPIs and JSON/CSV downloads. |
| **About** | Architecture, lap-time model, Critic rules, real races vs FastF1, limits. |

## How the AI is used (and kept safe)

| Agent | Brain | Guard rail |
|---|---|---|
| Planner | AI model | Writes the plan as JSON (which plans to test, incl. its own). Validated: unknown tyres dropped; illegal plans kept so the Critic rejects them; "stay out" and "box now" always kept. Revises the plan from the Critic's feedback. |
| Race Research | Python | Pure calculation. |
| Strategy Executor | Python simulator | The AI never produces a lap time. |
| Strategy Critic | Rules | Deliberately not AI: the safety gate (max 2 revisions). |
| Pit Wall Operations | AI model | Chooses from Critic-approved options within 2 s of the fastest, explains why, writes the radio. Off-list choices are refused; a radio message that contradicts the plan is replaced. |
| Chat | AI model | Rewrites the agents' exact results simply; reads unusual scenario wording. |

One OpenAI-compatible client (`pitwall/llm.py`, standard library only) talks to Groq, Gemini or Ollama.

## The ML tyre model

`pitwall/ml/tyre_model.py`: scikit-learn gradient-boosted trees (monotonic in tyre age, race progress and gap to
the car ahead) predict each clean dry lap relative to the driver's typical lap from compound, tyre age, race
progress, circuit, season, gap to the car ahead and stint number. The number of trees is chosen on 2024 with a
model trained on 2019–2023; it is then trained on 2019–2024 and tested on **2025–2026 races it never saw**:
average error **0.559 s per lap vs 0.704 s for the formula (−21%)**, better on every compound. With
**Tyre model: ML model**, the Executor uses the learned wear and race-progress curves for that circuit (base pace is
re-measured with them). Retrain: `python scripts/train_tyre_model.py` (~15 s).

**Agents vs real teams** (`python scripts/evaluate_agents.py`, ~1 min): 406 real moments (P1, P5, P10 finishers,
3 laps before their first real stop). The agents made the team's call 43% of the time with the formula (39% with
ML tyres); with ML tyres the time-to-flag error on clean races drops from 17.4 s to 10.1 s (median).

## Real races vs FastF1

Both are real F1 timing data. **Real races** (162 races, 2019 → Singapore 2026) are built into the project and
load instantly offline (TracingInsights archive, MIT licence, itself made with FastF1). **FastF1** downloads a
race from F1's servers while the app runs (needs internet and `pip install -r requirements-optional.txt`); its
option only appears when it is installed. Use it on **Race Control → Choose the race moment → FastF1 download**:
pick a season and race, press **Download race timing** (the first download takes about a minute, then it is
cached in `data/fastf1_cache`), choose the driver and lap, then **Use this moment**. The Docker image (AWS)
includes FastF1 and keeps its cache in a Docker volume.

## Deploy on AWS

See **DEPLOY_AWS.md** (EC2 + Docker, step by step). `Dockerfile` and `docker-compose.yml` are included.

## Project layout

```
app.py                     entry point: top navigation + header
views/*.py                 one file per page (calls ui/pages.py)
ui/                        pages, components, agents view, Model Lab, charts, tables, styles, session state
pitwall/agents/            Planner, Research, Executor, Critic, Operations, chat assistant, AI brief/validation
pitwall/llm.py             AI model client (Groq / Gemini / Ollama) + rules fallback
pitwall/ml/tyre_model.py   ML tyre & lap-time model (dataset, training, evaluation, curves)
pitwall/evaluation.py      agents vs real teams
pitwall/orchestrator.py    shared workspace, routing, bounded revision loop
pitwall/simulation.py      lap-time model, constraints, positions
pitwall/config.py          all model parameters and assumptions
pitwall/reality.py         "what really happened" + hindsight events
pitwall/data_sources/      synthetic scenarios, built-in real races, optional FastF1
data/replays/              162 real races (~14 MB)
data/ml/, data/eval/       trained model card, curves, model; evaluation results
scripts/                   train_tyre_model.py, evaluate_agents.py, demo_critic_revision.py, build_replays.py
tests/                     pytest suite (incl. headless page tests and a stand-in AI server)
deploy/aws/, Dockerfile    deployment
```

## Notes and assumptions

* **Lap convention:** a stop "on lap P" happens at the end of lap P; new tyres from lap P+1 at age 0.
* **Lap time** = base pace + tyre effect (formula or ML curve) + weather + traffic + fuel/track effect + Safety
  Car/VSC slow-down. Pit loss ×0.5 under a Safety Car, ×0.6 under a VSC (assumptions in `config.py`).
* **No hindsight:** race states use only laps up to the decision point, except the opt-in, clearly labelled
  hindsight switch (real Safety Cars/VSCs/rain; red flags treated as a Safety Car).
* **Selection without AI:** lowest time among validated options; ties → fewer stops → later stop → id.
* Decision records: `data/decisions/decision_log.jsonl` (set `PITWALL_DECISION_LOG` to move it).
* The Deploy button is hidden (`.streamlit/config.toml`, `toolbarMode = "viewer"`).
* Advanced: `PITWALL_AI_BASE_URL` points the client at any other OpenAI-compatible endpoint.

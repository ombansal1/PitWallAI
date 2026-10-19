"""Model Lab page: the AI model (agent brain), the ML tyre model and the agents-vs-real-teams evaluation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pitwall import evaluation
from pitwall.llm import PROVIDERS
from pitwall.ml import tyre_model as tm

from . import charts
from .help_text import h
from .state import AI_CHOICES, AI_LABEL, ai_key, ai_status, ai_test_cb, on_ai_provider_change
from .styles import esc, kpis, section, tip


def md(x: str) -> None:
    st.markdown(x, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _card():
    return tm.load_card()


@st.cache_data(show_spinner=False)
def _curves():
    return tm.load_curves()


@st.cache_data(show_spinner=False)
def _eval():
    return evaluation.load()


@st.cache_resource(show_spinner=False)
def _model():
    return tm.load_model()


# ====================================================================== AI model
AI_ROLES = [
    ("🧭 Planner", "AI model", "Reads a short race brief and decides which pit-stop plans to test, including its "
     "own custom plans. Revises the plan when the Critic objects.", "Unknown tyres or unreadable laps are dropped; "
     "illegal plans are kept so the Critic visibly rejects them; 'stay out' and 'box now' are always kept."),
    ("🔎 Race Research", "Python tools", "Measures tyres, gaps, weather and track status.", "Pure calculation."),
    ("⚙️ Strategy Executor", "Python simulator", "Simulates every plan lap by lap (formula or ML tyre model).",
     "The AI never produces a lap time."),
    ("🛡️ Strategy Critic", "Rules", "Checks racing rules and every number; sends work back up to twice.",
     "Deliberately not AI: it is the safety gate."),
    ("📋 Pit Wall Operations", "AI model", "Chooses the call from the Critic-approved options within 2 s of the "
     "fastest, explains it and writes the team radio.", "A choice outside the shortlist is refused and the "
     "fastest option is used; all numbers come from the simulator."),
    ("💬 Chat", "AI model", "Understands free-text questions and scenarios and answers in plain English.",
     "It is given the agents' exact results and told not to change any number."),
]


def ai_section() -> None:
    ss = st.session_state
    s = ai_status()
    test = ss.get("ai_test") or {}
    state = ("Ready" if s["ready"] and s["on"] else "Key missing" if s["on"] else "Off (rules)")
    tested = s.get("tested")
    md(kpis([("Agent brain", esc(state), "", "ai" if s["on"] and s["ready"] else "", h("ai_brain")),
             ("Provider", esc(AI_LABEL.get(ss.get("ai_provider", "rules"), "").split(" (")[0]
                              if s["on"] else "None"), "", "", ""),
             ("Model", esc(ss.get("ai_model") if s["on"] else "–"), "", "", ""),
             ("Connection test", "✓ passed" if tested else ("✕ failed" if tested is False else "not run"),
              "", "good" if tested else "", "")], cols=4))
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        md(section("Choose the agents' brain", h("ai_brain"), first=True))
        st.selectbox("AI model provider", AI_CHOICES, key="ai_provider", format_func=lambda k: AI_LABEL[k],
                     on_change=on_ai_provider_change, help="Groq is recommended: free, fast and well known.")
        p = ss.get("ai_provider", "rules")
        if p in PROVIDERS:
            prov = PROVIDERS[p]
            st.text_input("Model name", key="ai_model", help="Suggested: " + ", ".join(prov.models))
            st.caption(prov.note)
            if prov.key_env:
                key, where = ai_key()
                st.text_input(f"API key ({prov.key_env})", key="ai_key_input", type="password",
                              placeholder="paste your key here, or keep it in .streamlit/secrets.toml",
                              help="Kept only in this browser session. For a permanent setup use secrets.toml "
                                   "(see the steps on the right).")
                st.caption(f"Key: {'found in ' + where if key else 'not set yet'}.")
            else:
                st.text_input("Ollama address", key="ai_ollama_url", help="Where Ollama is running.")
            st.button("Test connection", key="ai_test_btn", on_click=ai_test_cb, type="primary",
                      help="Sends one tiny request to the AI model.")
            if test and test.get("provider") == p:
                (st.success if test.get("ok") else st.error)(test.get("message", ""))
        else:
            st.info("The agents use their built-in rules. Pick Groq above to give them a real AI model.")
    with right:
        md(section("Get a free Groq key (2 minutes)", "Groq offers a free plan; no card is needed for it.", first=True))
        md('<ol class="pw-steps">'
           '<li>Open <b>console.groq.com</b> and sign in (Google, GitHub or email).</li>'
           '<li>Go to <b>API Keys</b> → <b>Create API Key</b>, give it a name, and copy it (it starts with '
           '<code>gsk_</code>).</li>'
           '<li>Quick try: paste it in the box on the left and press <b>Test connection</b>.</li>'
           '<li>Permanent: create the file <code>.streamlit/secrets.toml</code> in the project folder with one line: '
           '<code>GROQ_API_KEY = "gsk_..."</code>, then restart the app. It will start with the AI switched on.</li>'
           '<li>Never share the key or put it in the ZIP you hand in. On AWS, pass it as an environment variable.</li>'
           '</ol>')
    md(section("What the AI model does, and what keeps it safe", h("ai_brain")))
    cols = st.columns(3)
    for i, (agent, brain, job, guard) in enumerate(AI_ROLES):
        chip = "ai" if brain == "AI model" else ""
        cols[i % 3].markdown(
            f'<div class="pw-card"><h4>{esc(agent)} <span class="pw-chip {chip}">{esc(brain)}</span></h4>'
            f'<p>{esc(job)}</p><p><b>Guard rail:</b> {esc(guard)}</p></div>', unsafe_allow_html=True)
    res = ss.get("last_result")
    calls = ((res or {}).get("ai") or {}).get("calls") or []
    if calls:
        md(section("AI calls in the latest analysis", "What was asked, whether the answer was used, and how long it took."))
        st.dataframe([{"Step": c["purpose"], "Result": "✓ used" if c["ok"] else f"✕ {c['error']}",
                       "Seconds": c["latency_s"], "Tokens": c.get("tokens")} for c in calls],
                     hide_index=True, width="stretch")


# ====================================================================== ML tyre model
def ml_section() -> None:
    card = _card()
    if not card:
        st.warning("The ML model hasn't been trained yet. Run `python scripts/train_tyre_model.py` (about 15 s).")
        return
    m = card["metrics"]
    gain = 100 * (1 - m["ml"]["mae_s"] / m["formula"]["mae_s"])
    md(kpis([("ML error per lap", f'{m["ml"]["mae_s"]:.3f}<small>s</small>', "unseen 2025–26 races", "ai",
              h("ml_metrics")),
             ("Formula error per lap", f'{m["formula"]["mae_s"]:.3f}<small>s</small>', "same laps", "", ""),
             ("Improvement", f"{gain:.0f}<small>%</small>", "lower error than the formula", "good", ""),
             ("Laps learned from", f'{card["final_model"]["laps"]:,}', f'{card["final_model"]["races"]} real races',
              "", "")], cols=4))
    md(f'<p class="pw-side-note">{esc(card["algorithm"])}. Trained on {card["train"]["laps"]:,} laps '
       f'({", ".join(map(str, card["train"]["seasons"]))}), tested on {card["test"]["laps"]:,} laps from '
       f'{card["test"]["races"]} races it never saw ({", ".join(map(str, card["test"]["seasons"]))}). '
       f'R² on the test races: {m["ml"]["r2"]:.2f} (lap times are noisy: traffic, mistakes, battles).</p>')
    a, b = st.columns(2, gap="medium")
    with a:
        md(section("Error on unseen races", h("ml_metrics")))
        st.altair_chart(charts.mae_bars(card), width="stretch")
    with b:
        md(section("Error by tyre", "Average error per lap on each dry compound, ML model vs formula."))
        st.altair_chart(charts.compound_bars(card), width="stretch")

    md(section("What the model learned about tyre wear", h("ml_curves")))
    curves = _curves()
    if curves:
        names = [tm.ALL] + [c for c in sorted(curves["circuits"]) if c != tm.ALL]
        c1, c2 = st.columns([1, 3])
        circuit = c1.selectbox("Circuit", names, key="ml_try_curve_circuit")
        c1.caption("Solid = learned from real laps; dashed = beyond the tyre ages in the data. Real data says the "
                   "three dry tyres behave more alike than the formula's textbook numbers: switch on the formula "
                   "to compare.")
        with c2:
            st.session_state.setdefault("ml_try_show_formula", False)
            show_f = st.toggle("Show the formula's curves (dotted)", key="ml_try_show_formula")
            formula = ({c: [tm.rule_tyre_adj(c, a) for a in range(46)] for c in tm.DRY} if show_f else None)
            st.altair_chart(charts.curves_chart(curves["circuits"][circuit], card["coverage_max_age"],
                                                formula=formula), width="stretch")
    a, b = st.columns(2, gap="medium")
    with a:
        md(section("Which inputs matter", h("ml_importance")))
        st.altair_chart(charts.importance_bars(card), width="stretch")
    with b:
        md(section("Predicted vs real (sample of unseen laps)", h("ml_scatter")))
        st.altair_chart(charts.scatter_chart(card), width="stretch")
    _try_model(card)
    with st.expander("Model card: how it was built and tested"):
        size = card.get("model_size_search", {})
        st.markdown(
            f"- **Target:** {card['target']}\n- **Inputs:** {', '.join(card['features'])}\n"
            f"- **Data filter:** {card['data_filter']}\n"
            f"- **Model size:** {size.get('chosen_trees')} trees, chosen on season {size.get('validation_season')} "
            f"with a model trained on {', '.join(map(str, size.get('trained_on', [])))} (the test seasons were never "
            "used to choose anything)\n"
            f"- **Curves for the simulator:** {card.get('curve_settings', '')}\n"
            f"- **Trained:** {card['trained_at']} with scikit-learn {card['sklearn_version']} in "
            f"{card['seconds_to_train']} s\n- **Retrain:** `python scripts/train_tyre_model.py`")
        if size.get("scores"):
            st.dataframe(pd.DataFrame(size["scores"]).rename(columns={"trees": "Trees",
                                                                     "validation_mae_s": "Error on 2024 (s)"}),
                         hide_index=True)


def _try_model(card) -> None:
    md(section("Try the model", h("ml_try")))
    model = _model()
    if model is None:
        st.info("The saved model was made with a different scikit-learn version, so it needs a quick retrain.")
        if st.button("Train the ML model now (about 15 s)", key="ml_train_btn"):
            with st.spinner("Training on the built-in races…"):
                tm.train_and_save()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        return
    for k, v in {"ml_try_age": 15, "ml_try_progress": 50, "ml_try_gap": 5.0, "ml_try_stint": 2}.items():
        st.session_state.setdefault(k, v)
    c1, c2, c3 = st.columns(3)
    circuit = c1.selectbox("Circuit", sorted(card["circuits"]), key="ml_try_circuit")
    tyre = c1.radio("Tyre", list(tm.DRY), key="ml_try_tyre", horizontal=True, format_func=str.title)
    age = c2.slider("Tyre age (laps)", 0, 45, key="ml_try_age")
    progress = c2.slider("Race progress (%)", 0, 100, key="ml_try_progress")
    gap = c3.slider("Gap to the car ahead (s)", 0.0, 5.0, step=0.5, key="ml_try_gap")
    stint = c3.number_input("Stint number", 1, 4, key="ml_try_stint")
    pred = model.predict(tyre, age, progress / 100, circuit, gap_ahead=gap, stint=int(stint))
    new = model.predict(tyre, 0, progress / 100, circuit, gap_ahead=gap, stint=int(stint))
    md(kpis([("Predicted lap vs typical", f"{pred:+.2f}<small>s</small>", "negative = faster than usual", "ai", ""),
             ("Cost of tyre age", f"{pred - new:+.2f}<small>s</small>", f"vs a new {tyre.lower()} at the same moment",
              "", ""),
             ("Formula says (tyre only)", f"{tm.rule_tyre_adj(tyre, age) - tm.rule_tyre_adj(tyre, 0):+.2f}<small>s</small>",
              "hand-written wear for comparison", "", "")], cols=3))


# ====================================================================== agents vs real teams
def eval_section() -> None:
    ev = _eval()
    if not ev:
        st.warning("No evaluation yet. Run `python scripts/evaluate_agents.py` (about a minute).")
        return
    f, m = ev["summary"]["formula"], ev["summary"]["ml"]
    md(f'<p class="pw-side-note">{esc(ev["method"])} {tip(h("eval"))}</p>')
    md(kpis([("Same call as the team", f'{f["same_call_pct"]:.0f}<small>%</small>',
              f'formula · ML model {m["same_call_pct"]:.0f}%', "best", h("eval")),
             ("Same tyre as the team", f'{f["same_tyre_pct"]:.0f}<small>%</small>',
              f'formula · ML model {m["same_tyre_pct"]:.0f}%', "", ""),
             ("Same number of stops", f'{f["same_stop_count_pct"]:.0f}<small>%</small>', "both tyre models", "", ""),
             ("Time-to-flag error, clean races", f'{m["median_time_error_clean_s"]:.1f}<small>s</small>',
              f'ML model · formula {f["median_time_error_clean_s"]:.1f} s ({m["clean_runs"]} races)', "ai",
              h("eval_clean"))], cols=4))
    a, b = st.columns([1.2, 1], gap="medium")
    with a:
        md(section("How the agents' call compared with the team's", h("eval")))
        st.altair_chart(charts.eval_verdicts(ev, evaluation.VERDICT_TEXT), width="stretch")
    with b:
        md(section("Same call as the team, by season", "Share of moments where the agents' call matched."))
        st.altair_chart(charts.eval_seasons(ev, "by_season", "Same call (%)"), width="stretch")
    md(section("Race-time error on clean races, by season", h("eval_clean"),
               sub="The ML model is trained on every season, so only 2025–2026 are a fully fair test here "
                   "(the unseen-season test is on the ML tab)."))
    st.altair_chart(charts.eval_seasons(ev, "clean_time_error_by_season", "Median error (s)"), width="stretch")
    st.caption("Why do the agents disagree with teams? Teams know things the model doesn't: tyre temperatures, "
               "rivals' plans, team orders, damage. A Safety Car after the decision changes the best call. And "
               "the model has only a few plan shapes (now, +3 laps, events). Agreement is a sanity check, not "
               "a score to maximise.")
    with st.expander(f"All {len(ev['rows'])} runs"):
        df = pd.DataFrame(ev["rows"])
        c1, c2 = st.columns(2)
        seasons = c1.multiselect("Season", sorted(df["year"].unique()), key="ml_try_eval_season")
        model = c2.radio("Tyre model", ["formula", "ml"], key="ml_try_eval_model", horizontal=True,
                         format_func=lambda x: "Formula" if x == "formula" else "ML model")
        view = df[(df["tyre_model"] == model) & (df["year"].isin(seasons) if seasons else True)]
        view = view.assign(verdict=view["verdict"].map(evaluation.VERDICT_TEXT))
        st.dataframe(view[["race", "driver", "finish", "decision_lap", "agents_call", "team_call", "verdict",
                           "model_error_s"]].rename(columns={
                               "race": "Race", "driver": "Driver", "finish": "Finished", "decision_lap": "After lap",
                               "agents_call": "Agents' call", "team_call": "Team did", "verdict": "Result",
                               "model_error_s": "Time error (s)"}), hide_index=True, width="stretch", height=380)


def model_lab() -> None:
    t1, t2, t3 = st.tabs(["🧠 AI model (agent brain)", "📈 ML tyre model", "🏁 Agents vs real teams"])
    with t1:
        ai_section()
    with t2:
        ml_section()
    with t3:
        eval_section()

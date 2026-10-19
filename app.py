"""PitWall AI — Streamlit entry point.  Run:  streamlit run app.py

Top navigation with one page per job (files in views/, content in ui/pages.py).
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="PitWall AI", page_icon="🏁", layout="wide", initial_sidebar_state="collapsed")

from ui import components, tables  # noqa: E402
from ui.state import ai_status, init_state, keep_widget_state, restore_inputs  # noqa: E402
from ui.styles import CSS, esc  # noqa: E402

PAGES = [  # (file, title, icon); each page's web address is its file name, e.g. /model_lab
    ("race_control", "Race Control", ":material/sports_score:"),
    ("agents", "Agents", ":material/hub:"),
    ("chat", "Ask the Pit Wall", ":material/forum:"),
    ("strategy_lab", "Strategy Lab", ":material/science:"),
    ("model_lab", "Model Lab", ":material/model_training:"),
    ("setup", "Race Setup", ":material/tune:"),
    ("records", "Records", ":material/history:"),
    ("about", "About", ":material/info:"),
]


def header() -> None:
    ss = st.session_state
    base = ss["raw_base"]
    s = ai_status()
    left, right = st.columns([1.1, 2], vertical_alignment="center")
    left.markdown('<div class="pw-brand"><div class="pw-logo">PW</div><div><div class="nm">PitWall <span class="ai">AI'
                  '</span></div><div class="sub">Agentic F1 race strategy</div></div></div>', unsafe_allow_html=True)
    pills = [f'<span class="pw-pill"><span class="pw-dot"></span>{esc(base.get("scenario_name", ""))}</span>']
    if base.get("data_source") in ("replay", "fastf1"):
        pills.append('<span class="pw-pill alert"><span class="pw-dot"></span><b>Real race</b>&nbsp;replay</span>'
                     if not (base.get("replay") or {}).get("real_events") else
                     '<span class="pw-pill alert"><span class="pw-dot"></span><b>Hindsight</b>&nbsp;real events on</span>')
    else:
        pills.append(f'<span class="pw-pill"><span class="pw-dot"></span>'
                     f'{esc(tables.SOURCE.get(base.get("data_source"), "Synthetic"))}</span>')
    pills.append(f'<span class="pw-pill"><span class="pw-dot"></span>Tyres <b>{esc(ss.get("w_tyre_model", "Formula"))}'
                 f'</b></span>')
    cls = "ai" if s["on"] and s["ready"] else ("alert" if s["on"] else "off")
    pills.append(f'<span class="pw-pill {cls}" title="{esc(s["detail"])}"><span class="pw-dot"></span>'
                 f'<b>{esc(s["text"])}</b></span>')
    right.markdown(f'<div class="pw-status">{"".join(pills)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="pw-divider"></div>', unsafe_allow_html=True)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    keep_widget_state()                 # inputs survive switching pages
    if not st.session_state.get("pending_run"):
        restore_inputs()                # after a live run's restart
    pages = {name: st.Page(f"views/{name}.py", title=t, icon=i, default=(name == "race_control"))
             for name, t, i in PAGES}
    pg = st.navigation(list(pages.values()), position="top")
    pending = st.session_state.get("pending_run")
    if pending and pending.get("where", "agents") == "agents" and pg.url_path != "agents":
        st.switch_page(pages["agents"])     # watch the agents work live
    header()
    for level, msg in st.session_state.pop("flash", []):
        getattr(st, level)(msg)
    components.banners(st.session_state.get("last_result"))
    pg.run()


main()

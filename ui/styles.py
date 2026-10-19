"""Pit-wall timing-screen styling and small HTML building blocks.

Palette follows F1 timing-screen conventions: purple = best (the recommendation),
green = valid, amber = caution, red = rejected; tyre chips use real compound colours.
"""
from __future__ import annotations

import html

TYRE_COLORS = {"SOFT": "#FF4D4D", "MEDIUM": "#FFD23F", "HARD": "#F2F4F7",
               "INTERMEDIATE": "#3DDC6B", "WET": "#3D8BFF"}

CSS = """
<style>
:root{
  --pw-bg:#0F161F; --pw-panel:#17212D; --pw-panel-2:#1F2B3A; --pw-line:#2B3A4D;
  --pw-text:#EEF2F7; --pw-muted:#AEBBCB; --pw-faint:#7C8CA2;
  --pw-best:#B37BFF; --pw-ok:#3DD68C; --pw-warn:#F2C14E; --pw-bad:#FF6B6B; --pw-red:#E10600; --pw-ai:#5CC8FF;
  --pw-r1:6px; --pw-r2:12px;
}
html, body, [class*="st-"]{font-size:16.5px;}
[data-testid="stAppDeployButton"], .stAppDeployButton{display:none !important;}
.block-container{padding-top:4.6rem; padding-bottom:4rem; max-width:1320px;}
h1,h2,h3,h4{letter-spacing:-0.01em;}
p, li{line-height:1.55;}
.pw-num{font-variant-numeric:tabular-nums;}

/* ---------- top navigation (Streamlit header) ---------- */
header[data-testid="stHeader"]{background:rgba(15,22,31,.92); backdrop-filter:blur(8px);
        border-bottom:1px solid var(--pw-line);}
header[data-testid="stHeader"] a{font-weight:600;}

/* ---------- brand bar ---------- */
.pw-brand{display:flex; align-items:center; gap:.8rem;}
.pw-logo{width:42px; height:42px; border-radius:10px; background:linear-gradient(135deg,var(--pw-red),#7a0f0a);
         display:flex; align-items:center; justify-content:center; font-weight:900; color:#fff;
         letter-spacing:.02em; box-shadow:0 6px 18px -6px rgba(225,6,0,.7); flex:none;}
.pw-brand .nm{font-size:1.45rem; font-weight:900; line-height:1; letter-spacing:-0.02em;}
.pw-brand .nm .ai{color:var(--pw-best);}
.pw-brand .sub{color:var(--pw-muted); font-size:.85rem; margin-top:.2rem;}
.pw-status{display:flex; gap:.45rem; flex-wrap:wrap; justify-content:flex-end; align-items:center;}
.pw-pill{display:inline-flex; align-items:center; gap:.45rem; background:var(--pw-panel);
         border:1px solid var(--pw-line); border-radius:999px; padding:.32rem .75rem;
         font-size:.86rem; color:var(--pw-muted); white-space:nowrap; max-width:100%; overflow:hidden;
         text-overflow:ellipsis;}
.pw-pill b{color:var(--pw-text); font-weight:600;}
.pw-pill.alert{border-color:var(--pw-warn);}
.pw-pill.ai{border-color:rgba(92,200,255,.55);}
.pw-pill.off{border-style:dashed;}
.pw-dot{width:8px; height:8px; border-radius:50%; background:var(--pw-ok); flex:none;}
.pw-pill.alert .pw-dot{background:var(--pw-warn);} .pw-pill.ai .pw-dot{background:var(--pw-ai);
        box-shadow:0 0 0 0 rgba(92,200,255,.6); animation:pwPing 2s ease-out infinite;}
.pw-pill.off .pw-dot{background:var(--pw-faint);}
@keyframes pwPing{0%{box-shadow:0 0 0 0 rgba(92,200,255,.55);} 80%,100%{box-shadow:0 0 0 7px rgba(92,200,255,0);}}
.pw-divider{height:1px; background:var(--pw-line); margin:.8rem 0 1.1rem;}

/* ---------- page titles ---------- */
.pw-page{margin:.2rem 0 1rem;}
.pw-page .h1{font-size:2.15rem; font-weight:900; margin:0; padding:0; letter-spacing:-0.02em; line-height:1.15;}
.pw-page p{color:var(--pw-muted); font-size:1.02rem; margin:.35rem 0 0; max-width:75ch;}

/* ---------- step cards ---------- */
.pw-stephead{display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .55rem;}
.pw-stephead .n{width:1.7rem; height:1.7rem; border-radius:50%; background:var(--pw-best); color:#14091F;
        font-weight:900; display:flex; align-items:center; justify-content:center; font-size:.9rem; flex:none;}
.pw-stephead .h3{font-size:1.2rem; font-weight:800; margin:0; padding:0; display:flex; align-items:center;}
.pw-stephead .hint{color:var(--pw-muted); font-size:.9rem; margin-left:auto;}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--pw-line) !important; border-radius:var(--pw-r2) !important;}

/* ---------- KPI tiles ---------- */
.pw-kpis{display:grid; grid-template-columns:repeat(var(--cols,4),minmax(0,1fr)); gap:10px; margin:.4rem 0 .9rem;}
.pw-kpi{background:var(--pw-panel); border:1px solid var(--pw-line); border-radius:var(--pw-r2); padding:.8rem .95rem;
        animation:pwRise .45s cubic-bezier(.2,.7,.2,1) both; animation-delay:calc(var(--i,0) * .06s);}
.pw-kpi .k{color:var(--pw-muted); font-size:.85rem; display:flex; align-items:center;}
.pw-kpi .v{font-size:1.7rem; font-weight:800; font-variant-numeric:tabular-nums; margin-top:.15rem; line-height:1.15;}
.pw-kpi .v small{font-size:.9rem; color:var(--pw-muted); font-weight:600; margin-left:.2rem;}
.pw-kpi .d{color:var(--pw-muted); font-size:.82rem; margin-top:.2rem;}
.pw-kpi.good .v{color:var(--pw-ok);} .pw-kpi.best .v{color:var(--pw-best);} .pw-kpi.ai .v{color:var(--pw-ai);}
@keyframes pwRise{from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;}}

/* ---------- tooltip "?" ---------- */
.pw-tip{position:relative; display:inline-flex; align-items:center; justify-content:center;
        width:18px; height:18px; margin-left:.4rem; border-radius:50%; border:1px solid var(--pw-faint);
        color:var(--pw-muted); font-size:11px; font-weight:700; line-height:1; cursor:help;
        vertical-align:.1em; transition:border-color .15s, color .15s, background .15s;
        letter-spacing:0; text-transform:none; flex:none;}
.pw-tip:hover,.pw-tip:focus-visible{color:#fff; border-color:var(--pw-best); background:rgba(179,123,255,.18);
        outline:none;}
.pw-tip .pw-tip-body{position:absolute; top:calc(100% + 9px); left:-10px; z-index:9999;
        width:max-content; max-width:320px; padding:.65rem .8rem; border-radius:8px;
        background:#0A1017; border:1px solid var(--pw-line); box-shadow:0 10px 30px rgba(0,0,0,.5);
        color:var(--pw-text); font-size:.86rem; font-weight:400; line-height:1.5; text-align:left;
        white-space:normal; opacity:0; visibility:hidden; transform:translateY(-4px);
        transition:opacity .16s ease, transform .16s ease, visibility .16s; pointer-events:none;}
.pw-tip .pw-tip-body::before{content:""; position:absolute; top:-6px; left:14px; width:10px; height:10px;
        background:#0A1017; border-left:1px solid var(--pw-line); border-top:1px solid var(--pw-line);
        transform:rotate(45deg);}
.pw-tip.end .pw-tip-body{left:auto; right:-10px;}
.pw-tip.end .pw-tip-body::before{left:auto; right:14px;}
.pw-tip:hover .pw-tip-body,.pw-tip:focus-visible .pw-tip-body{opacity:1; visibility:visible; transform:none;}
[data-testid="stMarkdownContainer"]:has(.pw-tip){overflow:visible;}

/* ---------- section headings ---------- */
.pw-sec{margin:1.5rem 0 .6rem;}
.pw-sec .h3{font-size:1.25rem; font-weight:800; margin:0; padding:0; display:flex; align-items:center; line-height:1.3;}
.pw-sec p{color:var(--pw-muted); margin:.25rem 0 0; font-size:.95rem; max-width:80ch;}
.pw-sec.first{margin-top:.2rem;}

/* ---------- agent pipeline diagram ---------- */
.pw-flow{display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:0; margin:.4rem 0 1.6rem; position:relative;}
.pw-node{position:relative; margin:0 14px 0 0; background:var(--pw-panel); border:1px solid var(--pw-line);
         border-radius:var(--pw-r2); padding:.75rem .8rem .8rem; min-height:150px;
         animation:pwRise .5s cubic-bezier(.2,.7,.2,1) both; animation-delay:calc(var(--i) * .12s);}
.pw-node::after{content:""; position:absolute; right:-14px; top:50%; width:14px; height:2px; background:var(--pw-faint);}
.pw-node::before{content:""; position:absolute; right:-15px; top:calc(50% - 4px); border:5px solid transparent;
         border-left:6px solid var(--pw-faint); transform:translateX(9px); z-index:2;}
.pw-node.last{margin-right:0;} .pw-node.last::after, .pw-node.last::before{display:none;}
.pw-node .top{display:flex; align-items:center; gap:.4rem; font-size:.78rem; color:var(--pw-muted);}
.pw-node .ic{font-size:1.25rem;}
.pw-node .nm{font-weight:800; font-size:1rem; margin:.25rem 0 .15rem; line-height:1.2;}
.pw-node .st{font-size:.84rem; color:var(--pw-text); line-height:1.4;}
.pw-node .st .x{color:var(--pw-muted);}
.pw-node .bar{position:absolute; left:10px; right:10px; top:0; height:3px; border-radius:0 0 3px 3px;
         background:var(--c,var(--pw-ok));}
.pw-badge{display:inline-flex; align-items:center; gap:.25rem; font-size:.72rem; font-weight:700;
         padding:.08rem .45rem; border-radius:999px; border:1px solid var(--pw-line); color:var(--pw-muted);
         margin-left:auto; white-space:nowrap;}
.pw-badge.ai{color:var(--pw-ai); border-color:rgba(92,200,255,.5); background:rgba(92,200,255,.08);}
.pw-badge.rule{color:var(--pw-muted);}
.pw-badge.py{color:var(--pw-ok); border-color:rgba(61,214,140,.45);}
.pw-loop{grid-column:2 / 6; margin:.35rem 14px 0 0; border:2px dashed var(--pw-warn); border-top:none;
         border-radius:0 0 12px 12px; height:18px; position:relative; opacity:.9;}
.pw-loop span{position:absolute; left:50%; top:8px; transform:translateX(-50%); background:var(--pw-bg);
         color:var(--pw-warn); font-size:.8rem; font-weight:700; padding:0 .5rem; white-space:nowrap;}
.pw-loop.none{border-color:var(--pw-line);} .pw-loop.none span{color:var(--pw-faint);}

/* ---------- live agent stages ---------- */
.pw-stages{display:grid; grid-template-columns:repeat(5,1fr); gap:6px; margin:.3rem 0 .6rem;}
.pw-stage{background:var(--pw-panel); border:1px solid var(--pw-line); border-radius:var(--pw-r1);
          padding:.6rem .8rem; transition:border-color .2s, background .2s;}
.pw-stage .nm{font-weight:700; font-size:.95rem;}
.pw-stage .st{color:var(--pw-muted); font-size:.82rem; margin-top:.1rem;}
.pw-stage.active{border-color:var(--pw-best); background:rgba(179,123,255,.10);
          animation:pwPulse 1.1s ease-in-out infinite;}
.pw-stage.active .st{color:var(--pw-best); font-weight:700;}
.pw-stage.done{border-color:rgba(61,214,140,.55);}
.pw-stage.done .st{color:var(--pw-ok);}
@keyframes pwPulse{0%,100%{box-shadow:0 0 0 0 rgba(179,123,255,.0);} 50%{box-shadow:0 0 0 4px rgba(179,123,255,.25);}}
[data-testid="stExpander"] summary p{font-size:1rem;}

/* ---------- timing tower (race summary) ---------- */
.pw-tower{display:grid; grid-template-columns:1.6fr repeat(6,1fr); background:var(--pw-panel);
          border:1px solid var(--pw-line); border-radius:var(--pw-r2); margin:.4rem 0 .4rem;}
.pw-cell{padding:.7rem .95rem .75rem; border-left:1px solid var(--pw-line); min-width:0;}
.pw-cell:first-child{border-left:none;}
.pw-cell .k{color:var(--pw-muted); font-size:.84rem; display:flex; align-items:center;}
.pw-cell .v{font-size:1.4rem; font-weight:700; font-variant-numeric:tabular-nums; margin-top:.1rem;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.pw-cell .v small{font-size:.88rem; color:var(--pw-muted); font-weight:600; margin-left:.25rem;}
.pw-prog{height:4px; background:var(--pw-line); border-radius:4px; margin-top:.45rem; overflow:hidden;}
.pw-prog i{display:block; height:100%; background:var(--pw-text); border-radius:4px; width:var(--p);
           transition:width .6s ease;}
.pw-v-warn{color:var(--pw-warn);} .pw-v-wet{color:#6FB1FF;}

/* ---------- tyre chip ---------- */
.pw-tyre{display:inline-flex; align-items:center; justify-content:center; width:1.5em; height:1.5em;
         border-radius:50%; border:3px solid var(--tc); color:var(--pw-text); font-size:.7em;
         font-weight:900; margin-right:.35rem; vertical-align:.12em; background:#0B1118;}

/* ---------- recommendation ---------- */
.pw-call{background:linear-gradient(180deg,rgba(179,123,255,.08),rgba(179,123,255,0) 70%), var(--pw-panel);
         border:1px solid var(--pw-line); border-radius:var(--pw-r2);
         padding:1.15rem 1.3rem 1.2rem; position:relative; overflow:visible;}
.pw-call::before{content:""; position:absolute; left:0; top:14px; bottom:14px; width:4px;
         border-radius:0 4px 4px 0; background:var(--cc, var(--pw-best));}
.pw-call .q{color:var(--pw-muted); font-size:.92rem; display:flex; align-items:center;}
.pw-call .act{font-size:2rem; font-weight:900; line-height:1.15; margin:.35rem 0 .3rem; letter-spacing:-0.015em;}
.pw-call.reveal .act{animation:pwWipe .7s cubic-bezier(.2,.7,.2,1) both;}
@keyframes pwWipe{from{clip-path:inset(0 100% 0 0); opacity:.4;} to{clip-path:inset(0 0 0 0); opacity:1;}}
.pw-call .meta{color:var(--pw-muted); font-size:.88rem; display:flex; gap:.5rem; flex-wrap:wrap; align-items:center;}
.pw-call.review{--cc:var(--pw-warn);} .pw-call.review .act{color:var(--pw-warn);}

.pw-stats{display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:.75rem 0 .2rem;}
.pw-stat{background:var(--pw-panel-2); border-radius:var(--pw-r1); padding:.65rem .8rem;}
.pw-stat .k{color:var(--pw-muted); font-size:.82rem; display:flex; align-items:center;}
.pw-stat .v{font-size:1.3rem; font-weight:700; font-variant-numeric:tabular-nums; margin-top:.1rem;}
.pw-stat .v small{font-size:.82rem; color:var(--pw-muted); font-weight:600;}

.pw-risk{display:inline-flex; align-items:center; gap:.4rem; padding:.2rem .65rem; border-radius:999px;
         font-weight:700; font-size:.85rem; border:1px solid currentColor;}
.pw-risk.LOW{color:var(--pw-ok);} .pw-risk.MEDIUM{color:var(--pw-warn);} .pw-risk.HIGH{color:var(--pw-bad);}
.pw-riskline{display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; margin:.75rem 0 .1rem;
             color:var(--pw-muted); font-size:.92rem;}

.pw-list{list-style:none; padding:0; margin:.2rem 0 0;}
.pw-list li{position:relative; padding:.42rem 0 .42rem 1.15rem; border-bottom:1px solid var(--pw-line);
            font-size:.97rem; line-height:1.5;}
.pw-list li:last-child{border-bottom:none;}
.pw-list li::before{content:""; position:absolute; left:0; top:.95rem; width:6px; height:6px;
            border-radius:50%; background:var(--lc, var(--pw-muted));}
.pw-list.alt li::before{--lc:var(--pw-faint);}
.pw-list li.bad::before{background:var(--pw-bad);}
.pw-list.ai li::before{background:var(--pw-ai);}

/* ---------- radio ---------- */
.pw-radio{background:#0B121A; border:1px solid var(--pw-line); border-radius:var(--pw-r2); padding:.95rem 1.05rem 1.05rem;}
.pw-radio .hd{display:flex; align-items:center; gap:.6rem; color:var(--pw-muted); font-size:.85rem;}
.pw-wave{display:inline-flex; align-items:flex-end; gap:2px; height:14px;}
.pw-wave i{width:3px; background:var(--pw-ok); border-radius:2px; height:30%;
           animation:pwWave .9s ease-in-out 4; animation-delay:calc(var(--d) * .09s);}
@keyframes pwWave{0%,100%{height:25%;} 50%{height:100%;}}
.pw-radio .txt{font-size:1.12rem; line-height:1.55; margin-top:.5rem; font-style:italic;}

/* ---------- before / after ---------- */
.pw-ba{background:var(--pw-panel); border:1px solid var(--pw-line); border-radius:var(--pw-r2); padding:.9rem 1.05rem;}
.pw-ba .row{display:grid; grid-template-columns:4.5rem 1fr; gap:.5rem; padding:.3rem 0; font-size:.95rem;}
.pw-ba .row span{color:var(--pw-muted);}
.pw-ba .tag{display:inline-block; font-size:.8rem; font-weight:700; padding:.1rem .55rem; border-radius:999px;
            margin-bottom:.35rem;}
.pw-ba .tag.chg{background:rgba(242,193,78,.15); color:var(--pw-warn);}
.pw-ba .tag.same{background:rgba(61,214,140,.13); color:var(--pw-ok);}
.pw-ba .why{color:var(--pw-text); font-size:.94rem; margin-top:.4rem; padding-top:.5rem; border-top:1px solid var(--pw-line);}

.pw-reveal{animation:pwReveal .55s cubic-bezier(.2,.7,.2,1) both; margin-bottom:.4rem;}
@keyframes pwReveal{from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:none;}}
[data-testid="stChatMessage"]{animation:pwShow .25s ease both; opacity:0;}
@keyframes pwShow{to{opacity:1;}}

/* ---------- cards ---------- */
.pw-card{background:var(--pw-panel); border:1px solid var(--pw-line); border-radius:var(--pw-r2);
         padding:.85rem 1.05rem; margin-bottom:.6rem;}
.pw-card h4{margin:0 0 .35rem; font-size:1.02rem; display:flex; align-items:center; gap:.5rem;}
.pw-card p{margin:.2rem 0; color:var(--pw-muted); font-size:.94rem; line-height:1.5;}
.pw-card.ai{border-color:rgba(92,200,255,.45); background:linear-gradient(180deg,rgba(92,200,255,.07),rgba(92,200,255,0) 80%), var(--pw-panel);}
.pw-card.ai h4{color:var(--pw-ai);}
.pw-chips{display:flex; flex-wrap:wrap; gap:.35rem; margin:.35rem 0;}
.pw-chip{font-size:.82rem; padding:.14rem .55rem; border-radius:999px; background:var(--pw-panel-2);
         color:var(--pw-text); border:1px solid var(--pw-line);}
.pw-chip.ok{border-color:rgba(61,214,140,.5);} .pw-chip.bad{border-color:rgba(255,107,107,.55);}
.pw-chip.warn{border-color:rgba(242,193,78,.6);} .pw-chip.ai{border-color:rgba(92,200,255,.55); color:var(--pw-ai);}
.pw-steps{counter-reset:s; list-style:none; padding:0; margin:.3rem 0;}
.pw-steps li{counter-increment:s; position:relative; padding:.5rem 0 .5rem 2.3rem; font-size:.97rem;
             line-height:1.5; border-bottom:1px solid var(--pw-line);}
.pw-steps li::before{content:counter(s); position:absolute; left:0; top:.5rem; width:1.6rem; height:1.6rem;
             border-radius:50%; background:var(--pw-panel-2); color:var(--pw-best); font-weight:800;
             display:flex; align-items:center; justify-content:center; font-size:.82rem;}
.pw-formula{background:#0B121A; border:1px solid var(--pw-line); border-radius:var(--pw-r1);
            padding:.65rem .85rem; font-size:.95rem; line-height:1.65;}

/* ---------- Streamlit widgets ---------- */
.stTabs [data-baseweb="tab-list"]{gap:.25rem; border-bottom:1px solid var(--pw-line);}
.stTabs [data-baseweb="tab"]{padding:.6rem 1rem; border-radius:6px 6px 0 0; font-weight:600;
        transition:background .15s, color .15s;}
.stTabs [data-baseweb="tab"]:hover{background:var(--pw-panel); color:var(--pw-text);}
.stTabs [aria-selected="true"]{background:var(--pw-panel);}
.stButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a{font-weight:700;
        transition:transform .12s ease, box-shadow .2s ease, background .2s ease, border-color .2s ease;}
.stButton > button:hover, .stDownloadButton > button:hover{transform:translateY(-1px);}
.stButton > button:active{transform:translateY(0) scale(.98);}
.stButton > button[kind="primary"]:hover{box-shadow:0 6px 22px -6px rgba(179,123,255,.75);}
[data-testid="stPageLink"] a{border:1px solid var(--pw-line); border-radius:8px; padding:.3rem .7rem;
        background:var(--pw-panel);}
[data-testid="stPageLink"] a:hover{border-color:var(--pw-best);}
[data-testid="stExpander"] details{border-color:var(--pw-line) !important; border-radius:var(--pw-r2) !important;
        background:var(--pw-panel);}
[data-testid="stExpander"] summary{font-weight:600;}
[data-testid="stDataFrame"]{border:1px solid var(--pw-line); border-radius:var(--pw-r2); overflow:hidden;}
div[data-testid="stAlert"]{border-radius:var(--pw-r2);}
div[role="radiogroup"][aria-orientation="horizontal"], .stRadio [role="radiogroup"]{gap:.4rem;}
.pw-sub{font-weight:700; font-size:1.02rem; margin:.35rem 0 .3rem; display:flex; align-items:center;}
.pw-side-note{color:var(--pw-muted); font-size:.88rem; line-height:1.5; margin:.2rem 0 .6rem;}

@media (max-width: 1100px){
  .pw-flow{grid-template-columns:repeat(3,minmax(0,1fr)); row-gap:12px;}
  .pw-loop{display:none;}
}
@media (max-width: 900px){
  .pw-stages{grid-template-columns:repeat(2,1fr);}
  .pw-tower{grid-template-columns:repeat(2,1fr);}
  .pw-cell{border-left:none; border-top:1px solid var(--pw-line);}
  .pw-stats{grid-template-columns:repeat(2,1fr);}
  .pw-kpis{grid-template-columns:repeat(2,minmax(0,1fr));}
  .pw-flow{grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media (prefers-reduced-motion: reduce){
  *, *::before, *::after{animation-duration:.01ms !important; animation-iteration-count:1 !important;
        transition-duration:.01ms !important; animation-delay:0s !important;}
}
</style>
"""


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def tip(text: str, end: bool = False) -> str:
    """Hover/focus '?' tooltip. `end=True` opens the bubble leftwards (for right-hand edges)."""
    if not text:
        return ""
    return (f'<span class="pw-tip{" end" if end else ""}" tabindex="0" role="button" '
            f'aria-label="Help: {esc(text)}">?<span class="pw-tip-body" role="tooltip">{esc(text)}</span></span>')


def section(title: str, help_text: str = "", sub: str = "", first: bool = False) -> str:
    return (f'<div class="pw-sec{" first" if first else ""}"><div class="h3">{esc(title)}{tip(help_text)}</div>'
            + (f"<p>{esc(sub)}</p>" if sub else "") + "</div>")


def page_title(title: str, lede: str = "") -> str:
    return f'<div class="pw-page"><div class="h1">{esc(title)}</div>' + (f"<p>{esc(lede)}</p>" if lede else "") + "</div>"


def step_head(n, title: str, help_text: str = "", hint: str = "") -> str:
    return (f'<div class="pw-stephead"><span class="n">{esc(n)}</span><div class="h3">{esc(title)}{tip(help_text)}</div>'
            + (f'<span class="hint">{esc(hint)}</span>' if hint else "") + "</div>")


def kpis(items, cols: int = 4) -> str:
    """items: (label, value_html, detail, css_class, help)."""
    cells = []
    for i, it in enumerate(items):
        label, value, detail, cls, helptext = (list(it) + ["", "", ""])[:5]
        cells.append(f'<div class="pw-kpi {cls}" style="--i:{i}"><div class="k">{esc(label)}{tip(helptext)}</div>'
                     f'<div class="v">{value}</div>' + (f'<div class="d">{esc(detail)}</div>' if detail else "")
                     + "</div>")
    return f'<div class="pw-kpis" style="--cols:{cols}">{"".join(cells)}</div>'


def tyre_chip(compound: str) -> str:
    c = (compound or "").upper()
    return (f'<span class="pw-tyre" style="--tc:{TYRE_COLORS.get(c, "#93A1B3")}" '
            f'title="{esc(c.title())}">{esc(c[:1])}</span>')


def fmt_race_time(seconds) -> str:
    """3398.99 -> '56:38.99' (minutes:seconds) for timing-screen readability."""
    if seconds is None or seconds != seconds:
        return "–"
    m, s = divmod(float(seconds), 60)
    return f"{int(m)}:{s:05.2f}"

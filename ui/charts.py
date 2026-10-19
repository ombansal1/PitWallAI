"""Altair charts built only from calculated simulation outputs (themed for the pit-wall palette)."""
from __future__ import annotations

from typing import Any, Dict, List

import altair as alt
import pandas as pd

from .styles import TYRE_COLORS

FONT = "Titillium Web, Segoe UI, sans-serif"
TEXT, MUTED, GRID = "#EEF2F7", "#AEBBCB", "#2B3A4D"
# Status encoding (validated on the #1A2533 surface): recommended / valid / rejected.
VERDICT_COLORS = {"Recommended": "#A66BF5", "Valid": "#7D8CA0", "Rejected": "#E0565B"}
# Categorical series (validated dark-mode slots, all-pairs pass for 3 series).
SERIES = ["#3987E5", "#D95926", "#199E70"]


def _theme(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(background="transparent", font=FONT)
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID,
                            tickColor=GRID, labelFontSize=12, titleFontSize=12, titleFontWeight=600)
            .configure_legend(labelColor=TEXT, titleColor=MUTED, labelFontSize=12, titleFontSize=12,
                              orient="bottom", symbolType="circle")
            .configure_text(color=TEXT, font=FONT))


def _verdict(status: str) -> str:
    return {"SELECTED": "Recommended", "ACCEPTED": "Valid"}.get(status, "Rejected")


def bar_rows(result: Dict[str, Any], max_rows: int = 10):
    """Valid options, plus rejected ones that were competitive (no slower than the slowest valid
    option + 5 s). Hopeless rejected options are left out so they don't squash the scale."""
    rows = [r for r in result["candidates"] if r["remaining_time_s"] is not None and r["delta_to_best_s"] is not None]
    valid = [r for r in rows if r["status"] in ("SELECTED", "ACCEPTED")]
    cap = max((r["delta_to_best_s"] for r in valid), default=0.0) + 5.0
    shown = valid + [r for r in rows if r not in valid and r["delta_to_best_s"] <= cap]
    shown = shown[:max_rows]
    return shown, len(rows) - len(shown)


def delta_bar_chart(result: Dict[str, Any], max_rows: int = 10) -> alt.Chart:
    rows, _ = bar_rows(result, max_rows)
    df = pd.DataFrame(rows)
    df["verdict"] = df["status"].map(_verdict)
    df["order"] = df["verdict"].map({"Recommended": 0, "Valid": 1, "Rejected": 2})
    df = df.sort_values(["order", "delta_to_best_s"])
    df["value_label"] = df.apply(lambda r: "Recommended" if r["verdict"] == "Recommended"
                                 else f"{r['delta_to_best_s']:+.1f} s", axis=1)
    df["why"] = df["issues"].replace("", "Passed every check")
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=list(df["label"]), title=None,
                axis=alt.Axis(labelLimit=260, labelColor=TEXT, labelFontSize=12.5)),
        tooltip=[alt.Tooltip("label:N", title="Strategy"), alt.Tooltip("verdict:N", title="Critic"),
                 alt.Tooltip("delta_to_best_s:Q", title="Gap to recommended (s)", format="+.2f"),
                 alt.Tooltip("why:N", title="Notes")])
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.62}).encode(
        x=alt.X("delta_to_best_s:Q", title="Seconds slower than the recommendation",
                scale=alt.Scale(nice=True)),
        color=alt.Color("verdict:N", title="Critic verdict",
                        scale=alt.Scale(domain=list(VERDICT_COLORS), range=list(VERDICT_COLORS.values()))))
    labels = base.mark_text(align="left", dx=6, fontSize=12, fontWeight=600, color=MUTED).encode(
        x=alt.X("delta_to_best_s:Q"), text="value_label:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(x="x:Q")
    return _theme((zero + bars + labels).properties(height=alt.Step(34)))


def gap_trace_chart(result: Dict[str, Any], top_n: int = 3) -> alt.Chart:
    """Running gap of the next-best valid plans to the recommended plan, lap by lap."""
    sel = result["selected_id"]
    others = [r for r in result["candidates"] if r["status"] == "ACCEPTED"][:top_n]
    base = {row["lap"]: row["cumulative_ref_s"] for row in result["simulations"][sel]["laps"]}
    data = [{"lap": row["lap"], "strategy": r["label"], "delta_s": row["cumulative_ref_s"] - base[row["lap"]]}
            for r in others for row in result["simulations"][r["id"]]["laps"]]
    zero = alt.Chart(pd.DataFrame({"y": [0], "t": ["Recommended plan"]})).mark_rule(
        color="#A66BF5", strokeWidth=2).encode(y="y:Q")
    zero_lbl = alt.Chart(pd.DataFrame({"y": [0], "t": ["Recommended plan"]})).mark_text(
        align="left", dy=-8, x=4, color="#C9A4FF", fontWeight=600, fontSize=12).encode(y="y:Q", text="t:N")
    if not data:
        return _theme((zero + zero_lbl).properties(height=300))
    df = pd.DataFrame(data)
    names = list(dict.fromkeys(df["strategy"]))
    hover = alt.selection_point(fields=["lap"], nearest=True, on="pointerover", empty=False)
    lap_dom = [int(df["lap"].min()), int(df["lap"].max())]
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("lap:Q", title="Lap", scale=alt.Scale(domain=lap_dom, nice=False),
                axis=alt.Axis(tickMinStep=1, format="d")),
        y=alt.Y("delta_s:Q", title="Seconds behind recommended"),
        color=alt.Color("strategy:N", title=None, scale=alt.Scale(domain=names, range=SERIES[:len(names)]),
                        legend=alt.Legend(columns=1, labelLimit=360)))
    pts = lines.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("strategy:N", title="Strategy"), alt.Tooltip("lap:Q", title="Lap"),
                 alt.Tooltip("delta_s:Q", title="Behind recommended (s)", format="+.2f")]).add_params(hover)
    rule = alt.Chart(df).mark_rule(color=MUTED, strokeDash=[2, 3]).encode(x="lap:Q").transform_filter(hover)
    # autosize "fit" shares the height with the legend, so leave room for it
    return _theme((zero + zero_lbl + lines + rule + pts).properties(height=300 + 26 * len(names)))


def lap_time_chart(result: Dict[str, Any]) -> alt.Chart:
    df = pd.DataFrame(result["simulations"][result["selected_id"]]["laps"])
    df["tyre"] = df["compound"].str.title()
    df["shown"] = df["lap_time_s"] + df["pit_loss_s"]
    df["note"] = df.apply(lambda r: ", ".join(x for x in (
        "pit stop" if r["pit_loss_s"] > 0 else "", "Safety Car" if r["safety_car"] else "",
        "wet" if r["condition"] == "wet" else "") if x) or "green, dry", axis=1)
    tyres = [t for t in TYRE_COLORS if t.title() in set(df["tyre"])]
    return _theme(alt.Chart(df).mark_bar(cornerRadiusEnd=3, width={"band": 0.75}).encode(
        x=alt.X("lap:O", title="Lap", axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y("shown:Q", title="Lap time incl. pit lane (s)", scale=alt.Scale(zero=False)),
        color=alt.Color("tyre:N", title="Tyre", scale=alt.Scale(domain=[t.title() for t in tyres],
                                                               range=[TYRE_COLORS[t] for t in tyres])),
        tooltip=[alt.Tooltip("lap:O", title="Lap"), alt.Tooltip("tyre:N", title="Tyre"),
                 alt.Tooltip("tyre_age:Q", title="Tyre age (laps)"), alt.Tooltip("note:N", title="Conditions"),
                 alt.Tooltip("lap_time_s:Q", title="Lap time (s)", format=".2f"),
                 alt.Tooltip("pit_loss_s:Q", title="Pit lane (s)", format=".2f")],
    ).properties(height=240))


COND_COLORS = {"Green-flag lap": "#3987E5", "Pit-stop lap": "#199E70", "Safety Car / VSC / red flag": "#D95926"}


def reality_chart(cmp: Dict[str, Any]) -> alt.Chart:
    """Per-lap real minus model time for the team's real strategy; colour = what the lap was."""
    rows = []
    for r in cmp["lap_compare"]:
        if r["real_s"] is None or r["model_s"] is None:
            continue
        kind = ("Safety Car / VSC / red flag" if r["condition"] != "green" else
                "Pit-stop lap" if r.get("pit") else "Green-flag lap")
        rows.append({"lap": r["lap"], "diff": r["real_s"] - r["model_s"], "kind": kind,
                     "real": r["real_s"], "model": r["model_s"]})
    df = pd.DataFrame(rows)
    kinds = [k for k in COND_COLORS if k in set(df["kind"])] if len(df) else list(COND_COLORS)
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=2, width={"band": 0.8}).encode(
        x=alt.X("lap:O", title="Lap", axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y("diff:Q", title="Real minus model (s, + = real was slower)"),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=kinds, range=[COND_COLORS[k] for k in kinds])),
        tooltip=[alt.Tooltip("lap:O", title="Lap"), alt.Tooltip("kind:N", title="Lap type"),
                 alt.Tooltip("real:Q", title="Real (s)", format=".2f"),
                 alt.Tooltip("model:Q", title="Model (s)", format=".2f"),
                 alt.Tooltip("diff:Q", title="Difference (s)", format="+.2f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED).encode(y="y:Q")
    return _theme((zero + bars).properties(height=260))


EVENT_COLORS = {"Safety Car": "#D95926", "VSC": "#C98500", "Rain": "#3987E5"}


def scenario_timeline(payload: Dict[str, Any]) -> alt.Chart:
    """Plain race timeline: a 'Race events' row (Safety Car / VSC / rain) above one row per plan
    showing tyre stints, with 'box L24' labels at each pit stop. Labels sit on the bars (no legend)."""
    first, total = payload["first"], payload["total"]
    events = [{"row": "Race events", "x0": max(p["start"], first) - 0.5, "x1": p["end"] + 0.5,
               "what": "Safety Car" if p["kind"] == "SC" else "VSC",
               "label": ("SC" if p["kind"] == "SC" else "VSC") + f" {p['start']}–{p['end']}" if p["end"] != p["start"]
               else ("SC" if p["kind"] == "SC" else "VSC") + f" {p['start']}",
               "mid": (max(p["start"], first) + p["end"]) / 2} for p in payload["neutral"]]
    events += [{"row": "Race events", "x0": r["start"] - 0.5, "x1": r["end"] + 0.5, "what": "Rain",
                "label": f"Rain {r['start']}–{r['end']}", "mid": (r["start"] + r["end"]) / 2} for r in payload["wet"]]
    stints, stops = [], []
    for plan in payload["plans"]:
        for st in plan["stints"]:
            stints.append({"row": plan["name"], "x0": st["from_lap"] - 0.5, "x1": st["to_lap"] + 0.5,
                           "tyre": st["compound"].title(), "laps": f"laps {st['from_lap']}–{st['to_lap']}",
                           "mid": (st["from_lap"] + st["to_lap"]) / 2,
                           "label": st["compound"].title() if st["to_lap"] - st["from_lap"] >= 5 else st["compound"][:1]})
        for st in plan["stints"][:-1]:
            stops.append({"row": plan["name"], "x": st["to_lap"] + 0.5, "label": f"box L{st['to_lap']}"})
    rows = (["Race events"] if events else []) + [p["name"] for p in payload["plans"]]
    x = alt.X("x0:Q", title="Lap", scale=alt.Scale(domain=[first - 0.5, total + 0.5], nice=False),
              axis=alt.Axis(format="d", tickMinStep=1))
    y = alt.Y("row:N", sort=rows, title=None, scale=alt.Scale(domain=rows, paddingInner=0.45),
              axis=alt.Axis(labelColor=TEXT, labelFontSize=12.5, labelLimit=200, ticks=False, domain=False,
                            grid=False))
    layers = []
    if events:
        edf = pd.DataFrame(events)
        layers.append(alt.Chart(edf).mark_bar(cornerRadius=3, opacity=0.9).encode(
            x=x, x2="x1:Q", y=y,
            color=alt.Color("what:N", legend=None, scale=alt.Scale(domain=list(EVENT_COLORS),
                                                                   range=list(EVENT_COLORS.values()))),
            tooltip=[alt.Tooltip("label:N", title="Event")]))
        layers.append(alt.Chart(edf).mark_text(fontSize=11, fontWeight=700, color="#0F1720").encode(
            x="mid:Q", y=y, text="label:N"))
    sdf = pd.DataFrame(stints)
    tyres = [t for t in TYRE_COLORS if t.title() in set(sdf["tyre"])]
    layers.append(alt.Chart(sdf).mark_bar(cornerRadius=3).encode(
        x=x, x2="x1:Q", y=y,
        color=alt.Color("tyre:N", legend=None, scale=alt.Scale(domain=[t.title() for t in tyres],
                                                               range=[TYRE_COLORS[t] for t in tyres])),
        tooltip=[alt.Tooltip("row:N", title="Plan"), alt.Tooltip("tyre:N", title="Tyre"),
                 alt.Tooltip("laps:N", title="Laps")]))
    layers.append(alt.Chart(sdf).mark_text(fontSize=11, fontWeight=800, color="#0F1720").encode(
        x="mid:Q", y=y, text="label:N"))
    if stops:
        pdf = pd.DataFrame(stops)
        layers.append(alt.Chart(pdf).mark_tick(color=TEXT, thickness=2, size=26).encode(x="x:Q", y=y))
        layers.append(alt.Chart(pdf).mark_text(align="left", dx=4, dy=-15, fontSize=11, fontWeight=700,
                                               color=TEXT).encode(x="x:Q", y=y, text="label:N"))
    chart = alt.layer(*layers).resolve_scale(color="independent")
    return _theme(chart.properties(height=62 * len(rows) + 30))


def options_chart(options: List[Dict[str, Any]]) -> alt.Chart:
    return delta_bar_chart({"candidates": options}, max_rows=6)


# ====================================================================== Model Lab
MODEL_COLORS = {"ML model": SERIES[0], "Formula": SERIES[1], "No model": SERIES[2]}


def _mscale(names):
    return alt.Scale(domain=list(names), range=[MODEL_COLORS[n] for n in names])


def mae_bars(card: Dict[str, Any]) -> alt.Chart:
    m = card["metrics"]
    df = pd.DataFrame([{"model": "ML model", "mae": m["ml"]["mae_s"]}, {"model": "Formula", "mae": m["formula"]["mae_s"]},
                       {"model": "No model", "mae": m["no_model"]["mae_s"]}])
    df["label"] = df["mae"].map(lambda v: f"{v:.3f} s")
    base = alt.Chart(df).encode(y=alt.Y("model:N", sort=list(df["model"]), title=None,
                                        axis=alt.Axis(labelColor=TEXT, labelFontSize=13)),
                                x=alt.X("mae:Q", title="Average error per lap (s) — lower is better",
                                        scale=alt.Scale(domain=[0, max(df["mae"]) * 1.25])))
    bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
        color=alt.Color("model:N", scale=_mscale(df["model"]), legend=None),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("mae:Q", title="MAE (s)", format=".3f")])
    text = base.mark_text(align="left", dx=6, fontSize=13, fontWeight=700, color=TEXT).encode(text="label:N")
    return _theme((bars + text).properties(height=150))


def compound_bars(card: Dict[str, Any]) -> alt.Chart:
    rows = []
    for comp, v in card["per_compound"].items():
        rows += [{"tyre": comp.title(), "model": "ML model", "mae": v["ml_mae_s"]},
                 {"tyre": comp.title(), "model": "Formula", "mae": v["formula_mae_s"]}]
    df = pd.DataFrame(rows)
    return _theme(alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("tyre:N", title=None, sort=["Soft", "Medium", "Hard"], axis=alt.Axis(labelAngle=0, labelColor=TEXT)),
        xOffset=alt.XOffset("model:N", sort=["ML model", "Formula"]),
        y=alt.Y("mae:Q", title="Error per lap (s)"),
        color=alt.Color("model:N", scale=_mscale(["ML model", "Formula"]), title=None),
        tooltip=[alt.Tooltip("tyre:N", title="Tyre"), alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("mae:Q", title="MAE (s)", format=".3f")]).properties(height=230))


def importance_bars(card: Dict[str, Any]) -> alt.Chart:
    df = pd.DataFrame(card["importance"])
    df["label"] = df["mae_increase_s"].map(lambda v: f"+{v:.3f} s")
    base = alt.Chart(df).encode(y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelColor=TEXT,
                                                                                         labelLimit=260)),
                                x=alt.X("mae_increase_s:Q", title="Extra error when this input is shuffled (s)",
                                        scale=alt.Scale(domain=[0, max(0.01, df["mae_increase_s"].max()) * 1.3])))
    bars = base.mark_bar(cornerRadiusEnd=4, height=18, color=MODEL_COLORS["ML model"]).encode(
        tooltip=[alt.Tooltip("feature:N", title="Input"), alt.Tooltip("mae_increase_s:Q", title="Extra error (s)",
                                                                       format=".3f")])
    text = base.mark_text(align="left", dx=5, fontSize=12, color=MUTED).encode(text="label:N")
    return _theme((bars + text).properties(height=34 * len(df) + 30))


def curves_chart(table: Dict[str, List[float]], cover: Dict[str, int], max_age: int = 45,
                 formula: Dict[str, List[float]] = None) -> alt.Chart:
    rows = []
    for comp, vals in table.items():
        lim = cover.get(comp, len(vals))
        for age, v in enumerate(vals[:max_age + 1]):
            rows.append({"tyre": comp.title(), "age": age, "loss": v,
                         "part": "learned from data" if age <= lim else "extended (beyond data)"})
    df = pd.DataFrame(rows)
    order = [c.title() for c in table]
    color = alt.Color("tyre:N", scale=alt.Scale(domain=order, range=[TYRE_COLORS[c.upper()] for c in order]),
                      title=None)
    tip = [alt.Tooltip("tyre:N", title="Tyre"), alt.Tooltip("age:Q", title="Tyre age (laps)"),
           alt.Tooltip("loss:Q", title="Slower than a new soft (s)", format=".2f"), alt.Tooltip("part:N", title="")]
    learned = alt.Chart(df[df["part"] == "learned from data"]).mark_line(strokeWidth=2.5).encode(
        x=alt.X("age:Q", title="Tyre age (laps)"), y=alt.Y("loss:Q", title="Slower than a new soft (s)"),
        color=color, tooltip=tip)
    ext_rows = df[df["part"] != "learned from data"]
    joins = df[df["part"] == "learned from data"].groupby("tyre").tail(1)   # connect solid and dashed parts
    ext = pd.concat([joins, ext_rows], ignore_index=True)
    extended = alt.Chart(ext).mark_line(strokeWidth=2, strokeDash=[5, 4], opacity=.8).encode(
        x="age:Q", y="loss:Q", color=color, tooltip=tip)
    pts = alt.Chart(df).mark_point(size=60, opacity=0, filled=True).encode(x="age:Q", y="loss:Q", color=color,
                                                                          tooltip=tip)
    layers = [learned, extended, pts]
    if formula:
        fdf = pd.DataFrame([{"tyre": c.title(), "age": a, "loss": v} for c, vals in formula.items()
                            for a, v in enumerate(vals[:max_age + 1])])
        layers.insert(0, alt.Chart(fdf).mark_line(strokeWidth=1.5, strokeDash=[1, 3], opacity=.55).encode(
            x="age:Q", y="loss:Q", color=color,
            tooltip=[alt.Tooltip("tyre:N", title="Formula tyre"), alt.Tooltip("age:Q", title="Tyre age"),
                     alt.Tooltip("loss:Q", title="Formula: slower than new soft (s)", format=".2f")]))
    return _theme(alt.layer(*layers).properties(height=320))


def scatter_chart(card: Dict[str, Any]) -> alt.Chart:
    df = pd.DataFrame(card["scatter"])
    lim = float(max(2.5, min(4.0, df[["actual_s", "ml_s"]].abs().quantile(.99).max())))
    df = df[(df["actual_s"].abs() <= lim) & (df["ml_s"].abs() <= lim)]
    pts = alt.Chart(df).mark_circle(size=22, opacity=.45, color=MODEL_COLORS["ML model"]).encode(
        x=alt.X("actual_s:Q", title="Real lap vs driver's typical lap (s)", scale=alt.Scale(domain=[-lim, lim])),
        y=alt.Y("ml_s:Q", title="ML prediction (s)", scale=alt.Scale(domain=[-lim, lim])),
        tooltip=[alt.Tooltip("actual_s:Q", title="Real (s)", format="+.2f"),
                 alt.Tooltip("ml_s:Q", title="Predicted (s)", format="+.2f"), alt.Tooltip("compound:N", title="Tyre")])
    diag = alt.Chart(pd.DataFrame({"a": [-lim, lim]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(x="a:Q", y="a:Q")
    return _theme((diag + pts).properties(height=320))


VERDICT_ORDER = ["Same call (tyre and lap within 2)", "Same tyre, different lap", "Different tyre",
                 "Different number of stops", "No call"]


def eval_verdicts(evald: Dict[str, Any], text: Dict[str, str]) -> alt.Chart:
    rows = []
    for key, name in (("formula", "Formula"), ("ml", "ML model")):
        s = evald["summary"].get(key) or {}
        n = max(1, s.get("runs", 1))
        for code, count in (s.get("verdicts") or {}).items():
            if count:
                rows.append({"verdict": text[code], "model": name, "pct": 100 * count / n, "count": count})
    df = pd.DataFrame(rows)
    order = [v for v in VERDICT_ORDER if v in set(df["verdict"])]
    return _theme(alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("verdict:N", sort=order, title=None, axis=alt.Axis(labelColor=TEXT, labelLimit=280)),
        yOffset=alt.YOffset("model:N", sort=["ML model", "Formula"]),
        x=alt.X("pct:Q", title="Share of race moments (%)"),
        color=alt.Color("model:N", scale=_mscale(["ML model", "Formula"]), title="Tyre model"),
        tooltip=[alt.Tooltip("verdict:N", title="Result"), alt.Tooltip("model:N", title="Tyre model"),
                 alt.Tooltip("pct:Q", title="%", format=".1f"), alt.Tooltip("count:Q", title="Moments")])
        .properties(height=78 * len(order) + 20))


def eval_seasons(evald: Dict[str, Any], field: str, title: str) -> alt.Chart:
    rows = []
    for key, name in (("formula", "Formula"), ("ml", "ML model")):
        for y, v in ((evald["summary"].get(key) or {}).get(field) or {}).items():
            rows.append({"season": str(y), "model": name, "value": v})
    df = pd.DataFrame(rows)
    color = alt.Color("model:N", scale=_mscale(["ML model", "Formula"]), title="Tyre model")
    base = alt.Chart(df).encode(x=alt.X("season:O", title=None, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y("value:Q", title=title), color=color,
                                tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("model:N", title="Model"),
                                         alt.Tooltip("value:Q", title=title, format=".1f")])
    return _theme((base.mark_line(strokeWidth=2) + base.mark_point(size=70, filled=True)).properties(height=240))

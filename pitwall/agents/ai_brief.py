"""What the AI model is told, and how its answers are checked.

`race_brief` turns the race state into a small, factual JSON brief (kept short so a free-plan token
budget lasts). `plan_from_ai` validates the Planner's JSON reply and converts it into a CandidateSpec.
Anything malformed is dropped and reported; plans that are well-formed but break racing rules are kept
on purpose, so the Strategy Critic visibly rejects them.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..config import DEFAULT_CONFIG, ModelConfig
from ..models import RaceState
from ..simulation import effective_max_stint, pit_loss_on_lap
from ..strategies import CandidateSpec

ANALYSES = ("track_status", "tyre_analysis", "gap_analysis", "weather_analysis")


def _span(p: Dict[str, Any], total: int) -> str:
    end = p.get("end") or total
    return f"{p['start']}" if end == p["start"] else f"{p['start']}-{end}"


def race_brief(state: RaceState, cfg: ModelConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    t = state.target
    used = set(t.compounds_used) | {t.compound}
    dry_used = {c for c in used if cfg.compounds[c].is_dry}
    wet_used = any(not cfg.compounds[c].is_dry for c in used)
    rule = ("satisfied" if wet_used or len(dry_used) >= 2
            else f"still needs a dry tyre other than {', '.join(sorted(dry_used)).lower()} (unless wet tyres are used)")
    life = effective_max_stint(t.compound, state.deg_multiplier, cfg)
    nearby = sorted(state.competitors, key=lambda c: abs(c.gap_to_leader_s - t.gap_to_leader_s))[:4]
    brief = {
        "decision_point": f"after lap {state.completed_laps} of {state.total_laps}",
        "laps_left": state.remaining_laps,
        "allowed_stop_laps": f"{state.first_lap}-{state.total_laps - 1} (a stop on lap P happens at the end of lap P)",
        "our_car": {"driver": t.name or t.code, "tyre": t.compound, "tyre_age_laps": t.tyre_age_laps,
                    "safe_life_left_laps": max(0, life - t.tyre_age_laps),
                    "new_sets_left": {c: n for c, n in t.available_sets.items() if n > 0},
                    "two_compound_rule": rule},
        "pit_stop_loss_s": {"green": round(state.pit_loss_s, 1),
                            "under_safety_car": round(state.pit_loss_s * cfg.sc_pit_loss_factor, 1),
                            "under_vsc": round(state.pit_loss_s * cfg.vsc_pit_loss_factor, 1)},
        "track_now": state.track_condition,
        "tyre_wear_multiplier": state.deg_multiplier,
        "safe_tyre_life_laps": {c: effective_max_stint(c, state.deg_multiplier, cfg) for c in cfg.compounds},
        "tyre_pace_note": "fresh soft fastest; medium about 0.45 s/lap slower, hard about 0.85 s/lap slower when "
                          "new, but they wear slower. Slicks on a wet track lose about 10 s/lap.",
        "safety_cars_ahead": [{"type": p["kind"], "laps": _span(p, state.total_laps),
                               "stop_cost_s": round(pit_loss_on_lap(state, max(p['start'], state.first_lap), cfg), 1)}
                              for p in state.neutral_ahead()],
        "rain_forecast": [{"wet_laps": _span(r, state.total_laps)} for r in state.wet_runs()],
        "pit_lane_closed_laps": list(state.pit_closed_laps),
        "cars_nearby": [{"driver": c.code, "gap_s": round(c.gap_to_leader_s - t.gap_to_leader_s, 1),
                         "tyre": c.compound, "tyre_age": c.tyre_age_laps} for c in nearby],
    }
    if state.events:
        brief["events_applied"] = [e.get("description") for e in state.events][-3:]
    return brief


PLANNER_SYSTEM = (
    "You are the Planner agent on a Formula 1 pit wall. A Python simulator will time every plan you "
    "propose and a strict Critic agent will reject any plan that breaks the rules, so think like a "
    "strategist: propose a varied, sensible set of pit-stop plans to test. Consider stopping now vs later, "
    "tyre choice, stopping when a Safety Car/VSC makes stops cheaper, switching to wet tyres for rain and "
    "back to slicks when it dries, and two-stop plans when one set cannot last. Tyre names: SOFT, MEDIUM, "
    "HARD, INTERMEDIATE, WET. At most 2 more stops. Reply with JSON only.")

PLANNER_SCHEMA = (
    '{"rationale": ["2-4 short reasons for this plan"],\n'
    ' "analyses": ["track_status","tyre_analysis","gap_analysis","weather_analysis" (pick what is useful)],\n'
    ' "include_stay_out": true|false, "include_box_now": true|false,\n'
    ' "delays": [laps after the first possible lap to box, e.g. 3, 6],\n'
    ' "compounds": [tyres to try for single stops; [] = every suitable tyre],\n'
    ' "event_stops": [{"lap": int, "why": "..."}],\n'
    ' "wet_transition": true|false, "return_to_slicks": true|false, "two_stop": true|false,\n'
    ' "custom_plans": [{"stops": [{"lap": int, "tyre": "MEDIUM"}], "why": "..."}]  (0-4 extra plans)}')


def _as_int(v) -> Optional[int]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else None


def _tyre(v, cfg: ModelConfig) -> Optional[str]:
    if not isinstance(v, str):
        return None
    name = v.strip().upper().rstrip("S")
    name = {"INTER": "INTERMEDIATE", "FULL WET": "WET", "INTERMEDIATE": "INTERMEDIATE"}.get(name, name)
    return name if name in cfg.compounds else None


def plan_from_ai(data: Dict[str, Any], state: RaceState, cfg: ModelConfig = DEFAULT_CONFIG
                 ) -> Tuple[Optional[CandidateSpec], List[str], List[str], List[str]]:
    """Validate the Planner's reply. Returns (spec or None, analyses, rationale, notes about dropped parts)."""
    notes: List[str] = []
    if not isinstance(data, dict):
        return None, [], [], ["reply was not a JSON object"]
    analyses = [a for a in (data.get("analyses") or []) if a in ANALYSES]
    for must in ("track_status", "tyre_analysis"):
        if must not in analyses:
            analyses.insert(0 if must == "track_status" else 1, must)
    if state.competitors and "gap_analysis" not in analyses:
        analyses.append("gap_analysis")
    if (state.wet_runs() or state.track_condition == "wet") and "weather_analysis" not in analyses:
        analyses.append("weather_analysis")
        notes.append("added the weather analysis (rain is in the race state)")
    rationale = [str(x).strip()[:220] for x in (data.get("rationale") or []) if str(x).strip()][:4]

    last_stop = state.total_laps - 1
    delays: List[int] = []
    for d in data.get("delays") or []:
        d = _as_int(d)
        if d is not None and 1 <= d and state.first_lap + d <= last_stop and d not in delays:
            delays.append(d)
        elif d is not None:
            notes.append(f"dropped a {d}-lap delay (outside the race)")
    delays = delays[:4]
    compounds = []
    for c in data.get("compounds") or []:
        name = _tyre(c, cfg)
        if name and name not in compounds:
            compounds.append(name)
        elif not name:
            notes.append(f"ignored unknown tyre '{c}'")
    if compounds and not any(state.target.available_sets.get(c, 0) for c in compounds):
        notes.append("none of the suggested tyres has a set left, so every suitable tyre is tried")
        compounds = []
    event_laps: List[Tuple[int, str]] = []
    for e in data.get("event_stops") or []:
        lap = _as_int(e.get("lap")) if isinstance(e, dict) else _as_int(e)
        why = (str(e.get("why", "")).strip()[:120] if isinstance(e, dict) else "") or "stop timed by the AI planner"
        if lap is not None and state.first_lap <= lap <= last_stop:
            if all(lap != x for x, _ in event_laps):
                event_laps.append((lap, why))
        elif lap is not None:
            notes.append(f"dropped an event stop on lap {lap} (outside laps {state.first_lap}-{last_stop})")
    custom: List[Dict[str, Any]] = []
    for p in (data.get("custom_plans") or [])[:4]:
        stops_in = p.get("stops") if isinstance(p, dict) else None
        if not isinstance(stops_in, list) or not stops_in:
            notes.append("dropped a custom plan with no stops")
            continue
        stops, problem = [], None
        for s in stops_in[:3]:
            lap = _as_int(s.get("lap")) if isinstance(s, dict) else None
            raw_tyre = (s.get("tyre") or s.get("compound")) if isinstance(s, dict) else None
            tyre = _tyre(raw_tyre, cfg)
            if lap is None:
                problem = "a stop without a lap number"
            elif tyre is None:
                problem = f"unknown tyre '{raw_tyre}'"
            if problem:
                break
            stops.append((lap, tyre))
        if problem:
            notes.append(f"dropped a custom plan with {problem}")
            continue
        stops.sort()
        why = str(p.get("why", "")).strip()[:160] or "proposed by the AI planner"
        custom.append({"stops": stops, "why": why})   # rule breaks are left for the Critic to reject
    spec = CandidateSpec(include_stay_out=bool(data.get("include_stay_out", True)),
                         include_pit_now=bool(data.get("include_box_now", True)),
                         delays=delays, compounds=compounds, event_laps=event_laps,
                         wet_transition=bool(data.get("wet_transition")) and state.rain_arrival() is not None,
                         return_to_slicks=bool(data.get("return_to_slicks")),
                         two_stop=bool(data.get("two_stop")), custom_plans=custom)
    return spec, analyses, rationale, notes


def merge_specs(old: CandidateSpec, new: CandidateSpec) -> Tuple[CandidateSpec, List[str]]:
    """Union of two specs (nothing already planned is lost). Returns (merged, list of additions)."""
    out = CandidateSpec(**{**old.to_dict()})
    out.event_laps = [tuple(x) for x in out.event_laps]
    added: List[str] = []
    if new.include_stay_out and not out.include_stay_out:
        out.include_stay_out = True
        added.append("added 'stay out'")
    if new.include_pit_now and not out.include_pit_now:
        out.include_pit_now = True
        added.append("added 'box this lap'")
    for d in new.delays:
        if d not in out.delays:
            out.delays.append(d)
            added.append(f"added a stop {d} laps later")
    if out.compounds and new.compounds != out.compounds:
        extra = [c for c in new.compounds if c not in out.compounds]
        if not new.compounds:
            out.compounds = []
            added.append("now tries every suitable tyre")
        elif extra:
            out.compounds += extra
            added.append("added tyres " + ", ".join(c.title() for c in extra))
    for lap, why in new.event_laps:
        if all(lap != x for x, _ in out.event_laps):
            out.event_laps.append((lap, why))
            added.append(f"added a stop on lap {lap} ({why})")
    for flag, text in (("wet_transition", "added switches to wet tyres for the rain"),
                       ("return_to_slicks", "added returns to slicks when it dries"),
                       ("two_stop", "added two-stop plans"), ("shift_closed_to_open", "moved stops off closed pit laps")):
        if getattr(new, flag) and not getattr(out, flag):
            setattr(out, flag, True)
            added.append(text)
    have = {tuple(p["stops"]) for p in out.custom_plans}
    for p in new.custom_plans:
        if tuple(p["stops"]) not in have:
            out.custom_plans.append(p)
            have.add(tuple(p["stops"]))
            added.append("added plan: " + ", ".join(f"lap {l} {c.title()}" for l, c in p["stops"]))
    return out, added


def results_digest(ws, limit: int = 6) -> Dict[str, Any]:
    """What the Planner is shown when revising: best valid options and why others failed."""
    rep = ws.critic_reports[-1]
    acc = sorted(rep["accepted"], key=lambda c: ws.simulations[c].remaining_time_s)
    best = ws.simulations[acc[0]].remaining_time_s if acc else None
    rejected: Dict[str, int] = {}
    for v in rep["verdicts"].values():
        for x in v["violations"]:
            rejected[x["code"]] = rejected.get(x["code"], 0) + 1
    return {"valid_options_fastest_first": [
        {"plan": ws.candidates[c].label, "seconds_slower_than_best": round(ws.simulations[c].remaining_time_s - best, 2)}
        for c in acc[:limit]],
        "rejections_by_rule": rejected,
        "critic_requests": [{"code": r["code"], "reason": r["reason"]} for r in rep["revision_requests"]]}


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)

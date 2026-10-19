"""Turn a plain-English race scenario into changes to the race inputs (deterministic, no AI model).

Understands, in any order and combination:
  * decision point:  "after lap 20 of 56", "lap 20/56", "15 laps to go", "a 58-lap race"
  * tyres:           "on mediums", "on 18-lap-old hards", "softs that are 12 laps old"
  * Safety Car:      "safety car on laps 24-27", "SC next lap for 3 laps", "safety car in 5 laps"
  * VSC:             "VSC on lap 30", "virtual safety car laps 30 to 31"
  * rain:            "rain from lap 35 to 45", "rain in 5 laps", "raining until lap 30", "wet track now"
  * tyre wear:       "1.8x tyre wear", "tyres wear 50% faster", "double wear"
  * pit stops:       "pit stop costs 25 seconds", "pit lane closed on laps 22 and 23"
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

EVENT_RE = re.compile(r"virtual safety car|virtual sc\b|\bvsc\b|virtual car|safety car|\bsc\b|"
                      r"\brain(?:s|ing|y)?\b|wet track|wet weather|\bshowers?\b|\bstorm\b|\bdrizzle\b")
RANGE_RE = re.compile(r"\blaps? (\d{1,3})\s*(?:-|–|to|until|through|and)\s*(?:lap )?(\d{1,3})\b")
SINGLE_RE = re.compile(r"(?:\b(?:on|from|at|starting(?: on| at| from)?|begins? (?:on|at)|out on)\s+|^\s*)laps? (\d{1,3})\b")
IN_RE = re.compile(r"\bin (\d{1,2}) laps?\b")
FOR_RE = re.compile(r"\bfor (\d{1,2}) laps?\b")
UNTIL_RE = re.compile(r"\b(?:until|till|up to|to|through) (?:the end of )?(?:lap )?(\d{1,3})\b")
TO_END_RE = re.compile(r"(until|to|till) the (end|finish|flag)|rest of the race")
NOW_RE = re.compile(r"\b(next lap|now|right now|immediately|already)\b")
COMP = {"soft": "SOFT", "medium": "MEDIUM", "hard": "HARD", "inter": "INTERMEDIATE",
        "intermediate": "INTERMEDIATE"}


@dataclass
class Scenario:
    changes: Dict[str, Any] = field(default_factory=dict)      # simple field overrides
    target: Dict[str, Any] = field(default_factory=dict)       # target-driver overrides
    neutral: List[Dict[str, Any]] = field(default_factory=list)
    wet: List[Dict[str, Any]] = field(default_factory=list)    # {'start', 'end' or None}
    wet_now: bool = False
    deg_factor: Optional[float] = None
    parts: List[str] = field(default_factory=list)             # plain-English list of what was understood
    assumptions: List[str] = field(default_factory=list)
    fresh: bool = False                                         # replace existing SC/VSC/rain inputs

    @property
    def empty(self) -> bool:
        return not (self.changes or self.target or self.neutral or self.wet or self.wet_now
                    or self.deg_factor)


def _segments(q: str) -> List[Tuple[str, str]]:
    """(keyword, text up to the next keyword or sentence break) for every event keyword."""
    hits = list(EVENT_RE.finditer(q))
    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(q)
        seg = q[m.end():end]
        seg = re.split(r"[;.]\s|,\s(?=[a-z])", seg)[0]
        out.append((m.group(0), seg))
    return out


def parse(text: str, raw: Dict[str, Any]) -> Scenario:
    q = " " + " ".join((text or "").lower().replace("–", "-").replace("’", "'").split()) + " "
    sc = Scenario()
    total = int(raw["total_laps"])
    done = int(raw["completed_laps"])

    # ---- decision point / race length
    m = re.search(r"(\d{2,3})[- ]lap race", q)
    if m:
        total = int(m.group(1))
        sc.changes["total_laps"] = total
    m = re.search(r"(?:after|on|at|it'?s|we'?re on|currently on|decision (?:at|after))?\s*lap (\d{1,3})\s*(?:of|/)\s*(\d{2,3})", q)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        sc.changes.update({"completed_laps": done, "total_laps": total})
    else:
        m = re.search(r"(\d{1,3}) laps? (?:to go|left|remaining)", q)
        if m:
            done = total - int(m.group(1))
            sc.changes["completed_laps"] = done
        else:
            m = re.search(r"\b(?:after|we'?re on|currently on|decision (?:at|after)) lap (\d{1,3})\b", q)
            if m:
                done = int(m.group(1))
                sc.changes["completed_laps"] = done
    if "completed_laps" in sc.changes or "total_laps" in sc.changes:
        sc.parts.append(f"Deciding after lap {done} of {total}")
        sc.fresh = True
    if re.search(r"\b(scenario|imagine|race where)\b", q):
        sc.fresh = True

    # ---- tyres
    tyre_age = None
    m = re.search(r"(\d{1,2})[- ]laps?[- ]old (soft|medium|hard|inter(?:mediate)?|wet)s?", q)
    if m:
        tyre_age, comp = int(m.group(1)), m.group(2)
    else:
        m = re.search(r"\b(soft|medium|hard|inter(?:mediate)?)s?\b(?: tyres?| tires?)?(?: that are| which are|,)? "
                      r"(?:aged )?(\d{1,2}) laps? old", q) or re.search(
            r"\bon (soft|medium|hard|inter(?:mediate)?|full wet)s?\b", q)
        comp = m.group(1) if m else None
        if m and m.lastindex and m.lastindex >= 2:
            tyre_age = int(m.group(2))
    if comp:
        code = "WET" if "wet" in comp else COMP.get(comp, COMP.get(comp.rstrip("s"), None))
        if code:
            sc.target["compound"] = code
            sc.parts.append(f"On {code.title()} tyres" + (f", {tyre_age} laps old" if tyre_age is not None else ""))
    if tyre_age is not None:
        sc.target["tyre_age_laps"] = tyre_age
    m = re.search(r"tyre age (?:of )?(\d{1,2})", q)
    if m and "tyre_age_laps" not in sc.target:
        sc.target["tyre_age_laps"] = int(m.group(1))
        sc.parts.append(f"Tyres {m.group(1)} laps old")

    # ---- Safety Car / VSC / rain periods
    for kw, seg in _segments(q):
        is_rain = kw.startswith(("rain", "wet", "shower", "storm", "drizzle"))
        kind = None if is_rain else ("VSC" if ("virtual" in kw or "vsc" in kw) else "SC")
        start = end = None
        r = RANGE_RE.search(seg)
        if r:
            start, end = int(r.group(1)), int(r.group(2))
        else:
            s1 = SINGLE_RE.search(seg)
            i1 = IN_RE.search(seg)
            if s1:
                start = int(s1.group(1))
            elif i1:
                start = done + int(i1.group(1)) + (0 if is_rain else 0)
            elif NOW_RE.search(seg):
                start = done + 1
            f1, u1 = FOR_RE.search(seg), UNTIL_RE.search(seg)
            if f1 and start is not None:
                end = start + int(f1.group(1)) - 1
            elif u1:
                end = int(u1.group(1))
                if start is None:
                    start = done + 1
        if is_rain:
            if start is None and end is None:
                if re.search(r"raining|wet track|rain now|it rains now|already", kw + seg) or NOW_RE.search(seg):
                    sc.wet_now = True
                    sc.parts.append("Raining now, until the finish")
                    continue
                start = done + 5
                sc.assumptions.append(f"rain starting in 5 laps (lap {start})")
            if end is None and not TO_END_RE.search(seg):
                sc.assumptions.append("rain lasting until the finish")
            sc.wet.append({"start": start, "end": end})
            sc.parts.append(f"Rain from lap {start}" + (f" to lap {end}" if end else " to the finish"))
        else:
            name = "Safety Car" if kind == "SC" else "VSC"
            if start is None:
                start = done + 1
                sc.assumptions.append(f"{name} starting next lap (lap {start})")
            if end is None:
                n = 4 if kind == "SC" else 2
                end = start + n - 1
                sc.assumptions.append(f"{name} lasting {n} laps")
            sc.neutral.append({"kind": kind, "start": start, "end": end})
            sc.parts.append(f"{name} on laps {start}–{end}" if end != start else f"{name} on lap {start}")

    # ---- tyre wear
    p = re.search(r"(\d{1,3})\s*% (?:more|higher|faster|extra|quicker)", q) if re.search(r"wear|deg", q) else None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|×|times)\s*(?:the )?(?:normal )?(?:tyre |tire )?(?:wear|deg|degradation)", q) or \
        re.search(r"(?:wear|deg|degradation)(?: is| of| at)? (?:x|×)?\s*(\d+(?:\.\d+)?)\s*(?:x|×|times)\b", q) or \
        re.search(r"(?:wear|deg|degradation)(?: is| of| at)? (?:x|×)\s*(\d+(?:\.\d+)?)\b", q)
    if p:
        sc.deg_factor = 1 + int(p.group(1)) / 100
    elif m:
        sc.deg_factor = float(m.group(1))
    elif re.search(r"double (the )?(tyre )?(wear|deg)", q):
        sc.deg_factor = 2.0
    elif re.search(r"(high|heavy|bad|fast) (tyre )?(wear|deg)|tyres? (are )?wearing (fast|quickly)", q):
        sc.deg_factor = 1.5
        sc.assumptions.append("'high wear' read as 1.5× normal")
    if sc.deg_factor is not None:
        sc.parts.append(f"Tyre wear ×{sc.deg_factor:g}")

    # ---- pit stops
    m = re.search(r"pit ?(?:stop|lane)? ?(?:loss|time)?\s*(?:is|of|costs?|takes?|=)\s*(\d+(?:\.\d+)?)\s*(?:s\b|sec|seconds)", q) \
        or re.search(r"pit (?:stops? )?(?:cost|take)s? (\d+(?:\.\d+)?)", q)
    if m:
        sc.changes["pit_loss_s"] = float(m.group(1))
        sc.parts.append(f"Pit stops cost {float(m.group(1)):g} s")
    m = re.search(r"pit lane (?:is )?closed (?:on |for )?laps? ([\d ,and-]+)", q)
    if m:
        laps = [int(x) for x in re.findall(r"\d{1,3}", m.group(1))]
        if laps:
            sc.changes["pit_closed_laps"] = laps
            sc.parts.append(f"Pit lane closed on lap{'s' if len(laps) > 1 else ''} {', '.join(map(str, laps))}")
    return sc


def apply(raw: Dict[str, Any], sc: Scenario) -> Dict[str, Any]:
    """New race-state dict with the scenario applied (the original is untouched)."""
    new = copy.deepcopy(raw)
    new.update({k: v for k, v in sc.changes.items() if k != "pit_closed_laps"})
    if "pit_closed_laps" in sc.changes:
        new["pit_closed_laps"] = sorted(set(sc.changes["pit_closed_laps"]))
    if sc.target:
        new["target"] = {**new["target"], **sc.target}
        if "compound" in sc.target:
            used = list(new["target"].get("compounds_used", []))
            if sc.target["compound"] not in used:
                used.append(sc.target["compound"])
            new["target"]["compounds_used"] = used
    lap_changed = "completed_laps" in sc.changes or "total_laps" in sc.changes
    if sc.fresh or lap_changed:
        new.update({"sc_start_lap": None, "sc_end_lap": None, "rain_start_lap": None, "rain_end_lap": None,
                    "neutral_periods": [], "wet_periods": [], "track_condition": "dry"})
        if lap_changed and isinstance(new.get("replay"), dict):
            new.pop("replay")   # a moved decision lap no longer matches the real race data
    total = int(new["total_laps"])
    new["neutral_periods"] = list(new.get("neutral_periods") or []) + [dict(p) for p in sc.neutral]
    wet = [{"start": w["start"], "end": w["end"] if w["end"] is not None else total} for w in sc.wet]
    new["wet_periods"] = list(new.get("wet_periods") or []) + wet
    if sc.wet_now:
        new["track_condition"] = "wet"
        new["rain_start_lap"] = new["rain_end_lap"] = None
    if sc.deg_factor is not None:
        new["deg_multiplier"] = round(float(sc.deg_factor), 4)
    new["scenario_name"] = (raw.get("scenario_name", "Scenario") + " (chat scenario)").replace(
        " (chat scenario) (chat scenario)", " (chat scenario)")
    return new

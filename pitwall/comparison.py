"""Previous-vs-current recommendation comparison with a numerical reason."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .config import DEFAULT_CONFIG, ModelConfig
from .models import PitStop, Strategy
from .simulation import check_strategy_constraints, simulate_strategy
from .validation import parse_race_state


def _strategy(defn: Dict[str, Any]) -> Strategy:
    return Strategy(defn["id"], defn["label"], defn["family"],
                    [PitStop(s["lap"], s["compound"]) for s in defn["stops"]])


def compare_results(prev: Dict[str, Any], curr: Dict[str, Any],
                    cfg: ModelConfig = DEFAULT_CONFIG) -> Optional[Dict[str, Any]]:
    if not prev or not curr or prev.get("status") == "invalid_input" or curr.get("status") == "invalid_input":
        return None
    p_rec, c_rec = prev["recommendation"], curr["recommendation"]
    same_point = (prev.get("decision_point") or {}).get("completed_laps") == \
        (curr.get("decision_point") or {}).get("completed_laps")
    out = {"previous_id": prev["analysis_id"], "current_id": curr["analysis_id"],
           "event": (curr.get("trigger_event") or {}).get("description", "input change"),
           "same_decision_point": same_point,
           "previous_action": p_rec["headline"], "current_action": c_rec["headline"],
           "previous_time_s": p_rec.get("numbers", {}).get("remaining_time_s"),
           "current_time_s": c_rec.get("numbers", {}).get("remaining_time_s"),
           "changed": p_rec.get("selected_id") != c_rec.get("selected_id")
           or p_rec["status"] != c_rec["status"]}

    # Re-evaluate the previous choice under the new race state (direct tool call).
    prev_sel = p_rec.get("selected_id")
    reason = ""
    if prev_sel and c_rec["status"] == "recommended":
        defn = prev["candidate_defs"][prev_sel]
        state = parse_race_state(curr["input_snapshot"], cfg)
        strat = _strategy(defn)
        violations = check_strategy_constraints(state, state.target, strat, cfg)
        if prev_sel in curr["simulations"]:
            t_prev_new = curr["simulations"][prev_sel]["remaining_time_s"]
        else:
            t_prev_new = round(simulate_strategy(state, state.target, strat, cfg).remaining_time_s, 3)
        out["previous_choice_now_s"] = t_prev_new
        out["previous_choice_now_feasible"] = not violations
        cur_t = c_rec["numbers"]["remaining_time_s"]
        if not out["changed"]:
            margin = c_rec["numbers"].get("margin_to_next_best_s")
            reason = (f"Unchanged: {c_rec['selected_label']} is still fastest at {cur_t:.2f} s"
                      + (f", {margin:.2f} s ahead of the next option." if margin is not None else "."))
        elif violations:
            reason = (f"Previous choice ({defn['label']}) is now infeasible: {violations[0]['message'].rstrip('.')}. "
                      f"New choice {c_rec['selected_label']} estimates {cur_t:.2f} s.")
        else:
            reason = (f"Previous choice ({defn['label']}) now estimates {t_prev_new:.2f} s; "
                      f"{c_rec['selected_label']} estimates {cur_t:.2f} s, "
                      f"{t_prev_new - cur_t:.2f} s faster under the new conditions.")
    elif c_rec["status"] != "recommended":
        reason = f"No validated strategy after the change: {c_rec.get('reason')}"
    elif not prev_sel:
        reason = f"Previously review required; now {c_rec['headline']} ({c_rec['numbers']['remaining_time_s']:.2f} s)."
    out["reason"] = reason
    if not same_point:
        out["reason"] += " (Note: decision points differ; times are not directly comparable.)"
    return out

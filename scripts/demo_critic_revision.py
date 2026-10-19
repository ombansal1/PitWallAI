"""DEMO — Strategy Critic rejection-and-revision paths (run: python scripts\\demo_critic_revision.py).

1. Planner revision: high-degradation scenario, every one-stop plan exceeds safe tyre life.
2. Executor revision: a test hook corrupts each first simulation attempt (pit loss counted twice);
   the Critic detects it and the Executor re-runs the affected candidates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows-safe when redirected

from pitwall.data_sources.synthetic import load_scenario  # noqa: E402
from pitwall.orchestrator import run_analysis  # noqa: E402


def show(title, r):
    print("=" * 88)
    print(title)
    print("=" * 88)
    for e in r["log"]:
        if e["kind"] in ("revision", "validation") or e["action"] in ("create_plan", "select_strategy", "finalize"):
            print(f"{e['step']:>3} {e['agent']:<20} {e['action']:<34} {e['detail'][:120]}")
    print(f"\nRevisions used: {r['retry_count']} of {r['max_revisions']}  →  {r['recommendation']['headline']}")
    for line in r["recommendation"]["summary"]:
        print("   ", line)
    print()


def corrupt_first_attempt(strategy_id, result, attempt):
    if attempt == 1:
        result.pit_loss_total_s += 21.5   # injected fault: pit loss counted twice
        result.remaining_time_s += 21.5
    return result


if __name__ == "__main__":
    show("DEMO 1 — Critic rejects all candidates → Planner revises the plan",
         run_analysis(load_scenario("high_degradation")))
    show("DEMO 2 — Critic rejects corrupted numbers → Executor re-runs (fault injected by demo)",
         run_analysis(load_scenario("dry_normal"), fault_injector=corrupt_first_attempt))

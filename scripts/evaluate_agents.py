"""Run the agents at ~470 real race moments and compare their call with the real team's call.

    python scripts/evaluate_agents.py            # all built-in races (a few minutes)
    python scripts/evaluate_agents.py 2024       # one season only

Writes data/eval/agents_vs_reality.json, which the Model Lab page shows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.data_sources.replays import load_index  # noqa: E402
from pitwall.evaluation import evaluate, save  # noqa: E402

if __name__ == "__main__":
    years = {int(a) for a in sys.argv[1:]}
    ids = [r["id"] for r in load_index()["races"] if not years or r["year"] in years]
    out = evaluate(ids, progress=print)
    save(out)
    for model, s in out["summary"].items():
        print(f"{model:8s} runs={s['runs']} same call={s['same_call_pct']}% same tyre={s['same_tyre_pct']}% "
              f"same number of stops={s['same_stop_count_pct']}% median time error={s['median_time_error_s']} s")

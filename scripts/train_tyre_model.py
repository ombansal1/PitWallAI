"""Train the ML tyre & lap-time model on the built-in real races and write data/ml/.

    python scripts/train_tyre_model.py

Writes model_card.json (metrics shown in the Model Lab), tyre_curves.json (used by the Strategy
Executor when the ML tyre model is switched on) and tyre_model.joblib (used by "Try the model").
Takes about 15-30 seconds on a laptop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.ml.tyre_model import train_and_save  # noqa: E402

if __name__ == "__main__":
    card = train_and_save(progress=print)
    m = card["metrics"]
    print(f"Test races ({', '.join(map(str, card['test']['seasons']))}): ML MAE {m['ml']['mae_s']:.3f} s, "
          f"formula {m['formula']['mae_s']:.3f} s, no model {m['no_model']['mae_s']:.3f} s; "
          f"R² {m['ml']['r2']:.2f}. Trained in {card['seconds_to_train']} s.")

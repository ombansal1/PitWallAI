"""Machine-learning tyre & lap-time model, trained on real F1 laps (the 162 built-in races).

What it learns
--------------
For every clean green-flag lap on dry tyres it predicts how much slower (or faster) that lap was than
the same driver's typical lap in that race, from: tyre compound, tyre age, race progress (fuel burn),
circuit, season, gap to the car ahead (dirty air) and stint number. Comparing each driver with themselves removes car and driver speed, so what is
left is mostly tyres and fuel.

Model: scikit-learn HistGradientBoostingRegressor (gradient-boosted decision trees, absolute-error loss so
odd laps don't dominate) with monotonic constraints: older tyres can only make a lap slower; later in the
race (less fuel) and more space to the car ahead can only make it faster.

Honest testing: the model size (number of trees) is chosen on the 2024 season using a model trained on
2019-2023; then it is trained on 2019-2024 and tested on 2025-2026 races it never saw, and compared with
(a) no model (always predict the driver's typical lap) and (b) PitWall's hand-written tyre formula.

How the agents use it: `tyre_curves()` turns the model into "seconds lost vs a new soft" per compound
and tyre age for each circuit. With "ML tyre model" switched on, the Strategy Executor uses those curves
instead of the formula. Safe tyre-life limits (the Critic's rule) stay the same either way.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONFIG, ModelConfig
from ..data_sources.replays import RACES_DIR, load_index

ML_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ml"
CARD_PATH = ML_DIR / "model_card.json"
CURVES_PATH = ML_DIR / "tyre_curves.json"
MODEL_PATH = ML_DIR / "tyre_model.joblib"

DRY = ("SOFT", "MEDIUM", "HARD")
COMPOUND_CODE = {c: i for i, c in enumerate(DRY)}
FEATURES = ["compound", "tyre_age", "progress", "circuit", "year", "gap_ahead", "stint"]
FEATURE_TEXT = {"compound": "Tyre compound", "tyre_age": "Tyre age (laps)", "progress": "Race progress (fuel burn)",
                "circuit": "Circuit", "year": "Season", "gap_ahead": "Gap to the car ahead (dirty air)",
                "stint": "Stint number (stops made + 1)"}
CATEGORICAL = [True, False, False, True, False, False, False]
# +1: can only make the lap slower; -1: can only make it faster; 0: free.
MONOTONIC = [0, 1, -1, 0, 0, -1, 0]
CLEAN_AIR_GAP_S = 5.0      # curves are predicted in clean air…
CURVE_STINT = 2            # …for a stint after a pit stop
TEST_YEARS = (2025, 2026)
MAX_AGE = 60
ALL = "All circuits (average)"
# Same venue, different race name.
CIRCUIT = {"Styrian Grand Prix": "Austrian Grand Prix", "70th Anniversary Grand Prix": "British Grand Prix",
           "Brazilian Grand Prix": "São Paulo Grand Prix"}


def circuit_of(event: str) -> str:
    return CIRCUIT.get(event, event)


class MLUnavailable(RuntimeError):
    pass


def _sklearn():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor
    except ImportError as e:  # pragma: no cover - sklearn is in requirements.txt
        raise MLUnavailable("scikit-learn is not installed (pip install -r requirements.txt)") from e


# ------------------------------------------------------------------------------------- dataset
def rule_tyre_adj(compound: str, age: float, cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    cp = cfg.compounds[compound]
    return cp.pace_offset_s + cp.deg_s_per_lap * age + cp.cliff_s_per_lap * max(0.0, age - cp.cliff_age_laps)


def race_laps(snapshot: Dict[str, Any], cfg: ModelConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Clean dry green-flag laps of one race, each compared with that driver's typical lap."""
    meta, L = snapshot["meta"], snapshot["laps"]
    df = pd.DataFrame({k: L[k] for k in ("drv", "lap", "time", "compound", "life", "status", "pin", "pout", "rain",
                                         "sesT", "stint")})
    total = int(meta.get("laps_winner") or df["lap"].max())
    # Gap to the car directly ahead on the road at the end of each lap (dirty air / traffic), capped at 5 s.
    order = df.dropna(subset=["sesT"]).sort_values(["lap", "sesT"])
    df["gap_ahead"] = order.groupby("lap")["sesT"].diff().clip(0, CLEAN_AIR_GAP_S).reindex(df.index)
    df["gap_ahead"] = df["gap_ahead"].fillna(CLEAN_AIR_GAP_S)
    df["stint"] = df["stint"].fillna(1).clip(1, 4)
    ok = ((df["lap"] >= 2) & df["time"].notna() & df["compound"].isin(DRY) & (df["status"].astype(str) == "1")
          & df["pin"].isna() & df["pout"].isna() & (df["rain"] != True) & df["life"].notna())  # noqa: E712
    df = df[ok].copy()
    if df.empty:
        return df
    ref = df.groupby("drv")["time"].transform("median")
    df = df[(df["time"] <= ref * 1.05) & (df["time"] >= ref * 0.95)]
    counts = df.groupby("drv")["time"].transform("size")
    df = df[counts >= 8].copy()
    if df.empty:
        return df
    df["ref"] = df.groupby("drv")["time"].transform("median")
    df["y"] = df["time"] - df["ref"]
    df["tyre_age"] = (df["life"] - 1).clip(lower=0).astype(int)
    df["progress"] = (df["lap"] / total).clip(0, 1)
    df["circuit"] = circuit_of(meta["event"])
    df["year"] = int(meta["year"])
    df["race"] = f"{meta['year']} {meta['event']}"
    fuel = cfg.real_fuel_effect_s_per_lap
    rule_raw = np.array([rule_tyre_adj(c, a, cfg) for c, a in zip(df["compound"], df["tyre_age"])]) - fuel * df["lap"].to_numpy()
    df["rule_raw"] = rule_raw
    df["rule_pred"] = df["rule_raw"] - df.groupby("drv")["rule_raw"].transform("median")
    return df[["race", "year", "circuit", "drv", "lap", "compound", "tyre_age", "progress", "gap_ahead", "stint",
               "time", "ref", "y", "rule_pred"]]


def build_dataset(race_ids: Optional[Iterable[str]] = None, races_dir: Path = RACES_DIR) -> pd.DataFrame:
    ids = list(race_ids) if race_ids is not None else [r["id"] for r in load_index()["races"]]
    frames = []
    for rid in ids:
        with open(races_dir / f"{rid}.json", encoding="utf-8") as f:
            frames.append(race_laps(json.load(f)))
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise MLUnavailable("no usable laps found")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------------------- model
def _matrix(df: pd.DataFrame, circuits: List[str]) -> np.ndarray:
    cmap = {c: i for i, c in enumerate(circuits)}
    return np.column_stack([
        df["compound"].map(COMPOUND_CODE).to_numpy(float),
        df["tyre_age"].to_numpy(float),
        df["progress"].to_numpy(float),
        df["circuit"].map(lambda c: cmap.get(c, np.nan)).to_numpy(float),   # unseen circuit = missing
        df["year"].to_numpy(float),
        df["gap_ahead"].to_numpy(float),
        df["stint"].to_numpy(float)])


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    err = p - y
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y - y.mean()) ** 2))
    return {"mae_s": round(float(np.mean(np.abs(err))), 4), "rmse_s": round(math.sqrt(float(np.mean(err ** 2))), 4),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else 0.0}


class TyreModel:
    """Fitted model + the circuit list it was trained with."""

    def __init__(self, est, circuits: List[str], last_year: int):
        self.est, self.circuits, self.last_year = est, circuits, last_year

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        return self.est.predict(_matrix(df, self.circuits))

    def predict(self, compound: str, tyre_age: float, progress: float, circuit: str, year: Optional[int] = None,
                gap_ahead: float = CLEAN_AIR_GAP_S, stint: int = CURVE_STINT) -> float:
        row = pd.DataFrame([{"compound": compound, "tyre_age": tyre_age, "progress": progress, "circuit": circuit,
                             "year": year or self.last_year, "gap_ahead": gap_ahead, "stint": stint}])
        return float(self.predict_frame(row)[0])


TREE_CHOICES = (25, 50, 75, 100, 150, 200, 300)


def fit(df: pd.DataFrame, n_trees: int = 50, seed: int = 42) -> TyreModel:
    HGB = _sklearn()
    circuits = sorted(df["circuit"].unique())
    est = HGB(loss="absolute_error", learning_rate=0.08, max_iter=int(n_trees), max_leaf_nodes=31,
              min_samples_leaf=60, l2_regularization=1.0, categorical_features=CATEGORICAL,
              monotonic_cst=MONOTONIC, early_stopping=False, random_state=seed)
    est.fit(_matrix(df, circuits), df["y"].to_numpy(float))
    return TyreModel(est, circuits, int(df["year"].max()))


def choose_trees(train: pd.DataFrame, choices=TREE_CHOICES) -> Tuple[int, List[Dict[str, float]]]:
    """Pick the number of trees by training on the earlier seasons and scoring on the LAST training
    season (a season the model hasn't seen, like the real test). Never looks at the test seasons."""
    last = int(train["year"].max())
    early, val = train[train["year"] < last], train[train["year"] == last]
    scores = []
    for n in choices:
        m = fit(early, n)
        scores.append({"trees": n, "validation_mae_s": round(float(np.mean(np.abs(m.predict_frame(val)
                                                                                  - val["y"].to_numpy()))), 4)})
    best = min(scores, key=lambda r: (r["validation_mae_s"], r["trees"]))
    return best["trees"], scores


def evaluate(model: TyreModel, test: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    from sklearn.inspection import permutation_importance
    y = test["y"].to_numpy(float)
    p_ml = model.predict_frame(test)
    out = {"ml": _metrics(y, p_ml), "formula": _metrics(y, test["rule_pred"].to_numpy(float)),
           "no_model": _metrics(y, np.zeros_like(y))}
    per_comp = {}
    for c in DRY:
        m = (test["compound"] == c).to_numpy()
        if m.sum() > 50:
            per_comp[c] = {"laps": int(m.sum()), "ml_mae_s": _metrics(y[m], p_ml[m])["mae_s"],
                           "formula_mae_s": _metrics(y[m], test["rule_pred"].to_numpy(float)[m])["mae_s"]}
    per_year = {}
    for yr in sorted(test["year"].unique()):
        m = (test["year"] == yr).to_numpy()
        per_year[str(int(yr))] = {"laps": int(m.sum()), "ml_mae_s": _metrics(y[m], p_ml[m])["mae_s"],
                                  "formula_mae_s": _metrics(y[m], test["rule_pred"].to_numpy(float)[m])["mae_s"]}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test), size=min(15000, len(test)), replace=False)
    sub = test.iloc[idx]
    imp = permutation_importance(model.est, _matrix(sub, model.circuits), sub["y"].to_numpy(float),
                                 n_repeats=3, random_state=seed, scoring="neg_mean_absolute_error")
    importance = sorted(({"feature": FEATURE_TEXT[f], "mae_increase_s": round(float(m), 4)}
                         for f, m in zip(FEATURES, imp.importances_mean)), key=lambda d: -d["mae_increase_s"])
    pick = rng.choice(len(test), size=min(1500, len(test)), replace=False)
    scatter = [{"actual_s": round(float(y[i]), 3), "ml_s": round(float(p_ml[i]), 3),
                "formula_s": round(float(test["rule_pred"].iloc[i]), 3), "compound": test["compound"].iloc[i]}
               for i in pick]
    return {"metrics": out, "per_compound": per_comp, "per_year": per_year, "importance": importance,
            "scatter": scatter}


def coverage(df: pd.DataFrame) -> Dict[str, int]:
    """Oldest tyre age with enough data per compound (95th percentile of the training laps)."""
    return {c: int(np.percentile(df.loc[df["compound"] == c, "tyre_age"], 95)) for c in DRY
            if (df["compound"] == c).any()}


def tyre_curves(model: TyreModel, cover: Dict[str, int], circuits: Optional[List[str]] = None,
                cfg: ModelConfig = DEFAULT_CONFIG) -> Dict[str, Dict[str, List[float]]]:
    """Seconds lost vs a NEW soft (same circuit, mid-race fuel) for ages 0..MAX_AGE, per circuit.
    Past the data coverage the curve continues with its recent slope (never flatter than the formula)."""
    circuits = circuits or model.circuits
    ages = np.arange(MAX_AGE + 1)
    rows = [{"compound": c, "tyre_age": a, "progress": 0.5, "circuit": ck, "year": model.last_year,
             "gap_ahead": CLEAN_AIR_GAP_S, "stint": CURVE_STINT} for ck in circuits for c in DRY for a in ages]
    pred = model.predict_frame(pd.DataFrame(rows)).reshape(len(circuits), len(DRY), len(ages))
    out: Dict[str, Dict[str, List[float]]] = {}
    for i, ck in enumerate(circuits):
        ref = pred[i, 0, 0]
        cur = {}
        for j, c in enumerate(DRY):
            # The absolute-error loss re-fits leaf values after the monotonic split search, which can leave
            # dips of a few thousandths of a second; a running maximum keeps "older is never faster".
            v = np.maximum.accumulate(pred[i, j] - ref)
            k = min(cover.get(c, MAX_AGE), MAX_AGE)
            if k < MAX_AGE:
                slope = max((v[k] - v[max(0, k - 8)]) / max(1, min(8, k)), cfg.compounds[c].deg_s_per_lap)
                v = v.copy()
                v[k + 1:] = v[k] + slope * (ages[k + 1:] - k)
            cur[c] = [round(float(x), 4) for x in v]
        out[ck] = cur
    out[ALL] = {c: [round(float(np.mean([out[ck][c][a] for ck in circuits])), 4) for a in ages] for c in DRY}
    return out


PROGRESS_POINTS = 21     # race progress 0, 0.05, …, 1


def progress_curves(model: TyreModel, circuits: Optional[List[str]] = None) -> Dict[str, List[float]]:
    """Learned effect of race progress (fuel burning off + track rubbering in), in seconds relative to the
    start of the race, at a fixed tyre (medium, 10 laps old, clean air). Non-increasing by construction."""
    circuits = circuits or model.circuits
    grid = np.linspace(0, 1, PROGRESS_POINTS)
    rows = [{"compound": "MEDIUM", "tyre_age": 10, "progress": g, "circuit": ck, "year": model.last_year,
             "gap_ahead": CLEAN_AIR_GAP_S, "stint": CURVE_STINT} for ck in circuits for g in grid]
    pred = model.predict_frame(pd.DataFrame(rows)).reshape(len(circuits), len(grid))
    out = {}
    for i, ck in enumerate(circuits):
        v = np.minimum.accumulate(pred[i] - pred[i, 0])
        out[ck] = [round(float(x), 4) for x in v]
    out[ALL] = [round(float(np.mean([out[ck][j] for ck in circuits])), 4) for j in range(len(grid))]
    return out


# ------------------------------------------------------------------------------------- training run + files
def train_and_save(race_ids: Optional[Iterable[str]] = None, out_dir: Path = ML_DIR, save_model: bool = True,
                   progress=None) -> Dict[str, Any]:
    """Build the dataset, train on seasons before TEST_YEARS, evaluate on TEST_YEARS, then refit on all
    seasons for the curves the agents use. Writes model_card.json, tyre_curves.json (+ the model)."""
    import sklearn
    say = progress or (lambda msg: None)
    t0 = time.time()
    say("Reading real laps from the built-in races…")
    df = build_dataset(race_ids)
    train, test = df[~df["year"].isin(TEST_YEARS)], df[df["year"].isin(TEST_YEARS)]
    if train.empty or test.empty:
        raise MLUnavailable("need races both before and in the test seasons")
    say(f"Choosing the model size on season {int(train['year'].max())} (held out from training)…")
    n_trees, tree_scores = choose_trees(train)
    say(f"Training {n_trees} trees on {len(train):,} laps ({train['race'].nunique()} races)…")
    model = fit(train, n_trees)
    say(f"Testing on {len(test):,} laps from {test['race'].nunique()} unseen races…")
    ev = evaluate(model, test)
    say("Refitting on every season for the strategy simulator…")
    final = fit(df, n_trees)
    cover = coverage(df)
    curves = tyre_curves(final, cover)
    progress = progress_curves(final)
    card = {
        "name": "PitWall tyre & lap-time model", "algorithm": "Gradient-boosted decision trees "
        "(scikit-learn HistGradientBoostingRegressor, absolute-error loss, monotonic in tyre age, race progress "
        "and gap to the car ahead)",
        "curve_settings": f"curves predicted mid-race, in clean air ({CLEAN_AIR_GAP_S:g} s gap), stint {CURVE_STINT}; "
                          "running maximum applied so older tyres are never faster; beyond the 95th-percentile "
                          "tyre age seen in the data the curve continues at its recent slope (at least the "
                          "formula's wear rate)",
        "target": "lap time minus the same driver's median clean lap in that race (seconds)",
        "features": [FEATURE_TEXT[f] for f in FEATURES],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__, "seconds_to_train": round(time.time() - t0, 1),
        "train": {"seasons": sorted(int(y) for y in train["year"].unique()), "laps": int(len(train)),
                  "races": int(train["race"].nunique())},
        "test": {"seasons": sorted(int(y) for y in test["year"].unique()), "laps": int(len(test)),
                 "races": int(test["race"].nunique())},
        "final_model": {"laps": int(len(df)), "races": int(df["race"].nunique()), "trees": int(final.est.n_iter_)},
        "model_size_search": {"validation_season": int(train["year"].max()), "trained_on": sorted(
            int(y) for y in train["year"].unique() if y < train["year"].max()), "scores": tree_scores,
            "chosen_trees": n_trees},
        "data_filter": "dry tyres, green flag only, no in/out laps, no rain, lap 2 onwards, within ±5% of the "
                       "driver's median, drivers with at least 8 such laps",
        "coverage_max_age": cover, "circuits": final.circuits, **ev,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CARD_PATH.name).write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / CURVES_PATH.name).write_text(json.dumps({"reference": "seconds slower than a new soft at the same "
                                                        "circuit, mid-race fuel load", "max_age": MAX_AGE,
                                                        "trained_at": card["trained_at"], "circuits": curves,
                                                        "progress": progress,
                                                        "progress_reference": "seconds relative to lap 1 as the "
                                                        "race goes on (fuel + track), at progress 0, 0.05 … 1"},
                                                       ensure_ascii=False), encoding="utf-8")
    if save_model:
        import joblib
        joblib.dump({"model": final, "sklearn": sklearn.__version__}, out_dir / MODEL_PATH.name, compress=3)
    say("Done.")
    return card


def load_card(path: Path = CARD_PATH) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_curves(path: Path = CURVES_PATH) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_model(path: Path = MODEL_PATH) -> Optional[TyreModel]:
    """The saved model, if it was made with the installed scikit-learn version (else None → retrain)."""
    try:
        import joblib
        import sklearn
        card = load_card(path.parent / CARD_PATH.name) or {}
        if card.get("sklearn_version") not in (None, sklearn.__version__):
            return None              # don't unpickle a model from another scikit-learn version
        blob = joblib.load(path)
        if blob.get("sklearn") != sklearn.__version__:
            return None
        return blob["model"]
    except Exception:
        return None


def curves_for(circuit: Optional[str], curves: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, List[float]]]:
    """Curves for a circuit (by race name), or the all-circuit average for unknown/synthetic tracks."""
    curves = curves or load_curves()
    if not curves:
        raise MLUnavailable("the ML tyre model hasn't been trained yet (python scripts/train_tyre_model.py)")
    table = curves["circuits"]
    key = circuit_of(circuit) if circuit else None
    if key not in table:
        key = ALL
    return key, table[key]


def tyre_model_payload(circuit: Optional[str]) -> Dict[str, Any]:
    """What gets attached to a race state (raw['tyre_model']) when the ML tyre model is switched on."""
    curves = load_curves()
    key, table = curves_for(circuit, curves)
    prog = (curves.get("progress") or {}).get(key)
    out = {"kind": "ml", "circuit": key, "curves": table,
           "label": f"ML tyre model ({'learned for ' + key if key != ALL else 'average of all circuits'})"}
    if prog:
        out["progress_curve"] = prog
    return out

"""Train and evaluate strikeout models on a strictly time-based split.

Models
------
1. **Naive baseline**: the pitcher's season-to-date K per start from prior
   games only (falls back to last season, then the league average).
2. **Linear regression** on all pre-game features (median imputation with
   missing-value indicators, standardized).
3. **Random forest** with a small grid over ``n_estimators``, ``max_depth`` and
   ``min_samples_leaf``, tuned with expanding-window season folds inside the
   training data (train on seasons before S, validate on S, for the last two
   training seasons). No random shuffling anywhere.

Split
-----
The test season is the newest season with at least ``MIN_TEST_STARTS`` starts;
training uses every earlier season. With 2021-2025 data that is train
2021-2024, test 2025. ``--test-season`` overrides it. After evaluation the
shipped model (``models/model.joblib``) is refit on every season, test included.

Leakage sanity check
--------------------
The models are also fit once with a known leaky feature, the current game's
batters faced, to show how much it inflates accuracy, and the leakage audit is
run on that leaky feature set to show the audit catches it.

Usage::

    python src/train.py
    python src/train.py --test-season 2025 --skip-leak-demo
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_features import (
    FEATURE_GROUPS,
    FEATURES,
    FEATURES_PATH,
    LEAKY_DEMO_COLUMN,
    TARGET,
    audit_leakage,
    format_audit,
)
from evaluate import regression_metrics
from fetch_data import load_raw

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "metrics"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "test_predictions.parquet"

SEED = 42
MIN_TEST_STARTS = 300
N_VALIDATION_SEASONS = 2
RF_GRID = {
    "rf__n_estimators": [200, 400],
    "rf__max_depth": [6, 10, 16],
    "rf__min_samples_leaf": [10, 30, 60],
}


def set_seeds(seed: int = SEED) -> None:
    """Fix Python and NumPy seeds so every run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def choose_test_season(df: pd.DataFrame, override: int | None = None) -> int:
    """Return the held-out season: the override, or the newest season with enough starts."""
    if override is not None:
        return override
    counts = df.groupby("season").size()
    full = counts[counts >= MIN_TEST_STARTS]
    if len(full) < 2:
        raise ValueError(f"Need at least two seasons with {MIN_TEST_STARTS}+ starts; have {counts.to_dict()}")
    return int(full.index.max())


def season_folds(train: pd.DataFrame, n_val: int = N_VALIDATION_SEASONS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV folds: fit on seasons before S, validate on S."""
    seasons = sorted(train["season"].unique())
    folds = []
    for val_season in seasons[-n_val:]:
        tr = np.flatnonzero(train["season"].to_numpy() < val_season)
        va = np.flatnonzero(train["season"].to_numpy() == val_season)
        if len(tr) and len(va):
            folds.append((tr, va))
    if not folds:
        raise ValueError("Training data needs at least two seasons for season-based tuning")
    return folds


def baseline_predict(df: pd.DataFrame, fallback: float) -> np.ndarray:
    """Naive baseline: season-to-date K per start from prior games (with fallbacks)."""
    return df["season_k_per_start"].fillna(fallback).to_numpy()


def make_linear() -> Pipeline:
    """Linear regression with median imputation, missing indicators, and scaling."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])


def make_forest(**params) -> Pipeline:
    """Random forest with median imputation and missing indicators."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)),
    ])


class LinearBoostBlend:
    """The average of the linear model and Poisson gradient boosting.

    The best option in ``src/compare_models.py`` (better than linear alone in
    both test seasons), and the model the app and ``predict_upcoming`` use.
    """

    def fit(self, X, y):
        self.linear_ = make_linear().fit(X, y)
        self.boost_ = HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=SEED,
        ).fit(X, y)
        return self

    def predict(self, X):
        return (self.linear_.predict(X) + self.boost_.predict(X)) / 2


def make_blend() -> LinearBoostBlend:
    """Linear regression + Poisson gradient boosting, averaged."""
    return LinearBoostBlend()


def tune_forest(train: pd.DataFrame, features: list[str]) -> tuple[Pipeline, dict, float]:
    """Grid-search the forest on season folds. Returns (refit model, best params, best CV MAE)."""
    search = GridSearchCV(
        make_forest(),
        RF_GRID,
        cv=season_folds(train),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
    )
    search.fit(train[features], train[TARGET])
    params = {k.removeprefix("rf__"): v for k, v in search.best_params_.items()}
    return search.best_estimator_, params, float(-search.best_score_)


def leak_demo(train: pd.DataFrame, test: pd.DataFrame, rf_params: dict, clean: dict) -> dict:
    """Fit with current-game batters faced added, and audit that leaky feature set."""
    leaky = FEATURES + [LEAKY_DEMO_COLUMN]
    lin = make_linear().fit(train[leaky], train[TARGET])
    rf = make_forest(**rf_params).fit(train[leaky], train[TARGET])
    y = test[TARGET].to_numpy()
    out = {
        "leaky_feature": LEAKY_DEMO_COLUMN,
        "linear": regression_metrics(y, lin.predict(test[leaky])),
        "random_forest": regression_metrics(y, rf.predict(test[leaky])),
    }
    for name in ("linear", "random_forest"):
        out[name]["mae_drop_vs_clean_pct"] = round(
            100 * (clean[name]["mae"] - out[name]["mae"]) / clean[name]["mae"], 1)

    raw = load_raw()
    out["audit_clean"] = audit_leakage(raw, FEATURES)
    out["audit_leaky"] = audit_leakage(raw, leaky)
    return out


def feature_set_ablation(train: pd.DataFrame, test: pd.DataFrame, rf_params: dict) -> dict:
    """Test-season MAE for nested feature sets, to show what each feature group adds."""
    sets = {
        "core": FEATURE_GROUPS["core"],
        "core + matchup": FEATURE_GROUPS["core"] + FEATURE_GROUPS["matchup"],
        "core + context": FEATURE_GROUPS["core"] + FEATURE_GROUPS["context"],
        "all": FEATURES,
    }
    y = test[TARGET].to_numpy()
    out = {}
    for name, cols in sets.items():
        lin = make_linear().fit(train[cols], train[TARGET])
        rf = make_forest(**rf_params).fit(train[cols], train[TARGET])
        out[name] = {
            "n_features": len(cols),
            "linear": regression_metrics(y, lin.predict(test[cols])),
            "random_forest": regression_metrics(y, rf.predict(test[cols])),
        }
    return out


def train_and_evaluate(df: pd.DataFrame, test_season: int | None = None, run_leak_demo: bool = True) -> dict:
    """Fit all models on the time split, save artifacts, and return a metrics dict."""
    set_seeds()
    df = df.sort_values(["game_date", "game_pk", "pitcher_id"]).reset_index(drop=True)
    test_season = choose_test_season(df, test_season)
    train = df[df["season"] < test_season].reset_index(drop=True)
    test = df[df["season"] == test_season].reset_index(drop=True)
    y_test = test[TARGET].to_numpy()
    print(f"Train: seasons {train['season'].min()}-{train['season'].max()} ({len(train):,} starts)")
    print(f"Test:  season {test_season} ({len(test):,} starts)")

    fallback = float(train[TARGET].mean())
    preds = {"baseline": baseline_predict(test, fallback)}

    linear = make_linear().fit(train[FEATURES], train[TARGET])
    preds["linear"] = linear.predict(test[FEATURES])

    print("Tuning random forest on season folds...")
    forest, rf_params, cv_mae = tune_forest(train, FEATURES)
    preds["random_forest"] = forest.predict(test[FEATURES])
    print(f"  best params {rf_params}, CV MAE {cv_mae:.3f}")

    metrics = {name: regression_metrics(y_test, p) for name, p in preds.items()}
    base_mae, base_rmse = metrics["baseline"]["mae"], metrics["baseline"]["rmse"]
    for m in metrics.values():
        m["mae_improvement_pct"] = round(100 * (base_mae - m["mae"]) / base_mae, 2)
        m["rmse_improvement_pct"] = round(100 * (base_rmse - m["rmse"]) / base_rmse, 2)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_seasons": sorted(int(s) for s in train["season"].unique()),
        "test_season": int(test_season),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "data_through": str(df["game_date"].max())[:10],
        "features": FEATURES,
        "rf_best_params": rf_params,
        "rf_cv_mae": round(cv_mae, 4),
        "metrics": metrics,
    }

    print("Comparing feature groups...")
    result["ablation"] = feature_set_ablation(train, test, rf_params)

    if run_leak_demo:
        print("Running leakage sanity check...")
        demo = leak_demo(train, test, rf_params, metrics)
        result["leak_demo"] = {k: v for k, v in demo.items() if not k.startswith("audit")}
        result["leakage_audit"] = {"clean_passed": demo["audit_clean"]["passed"],
                                   "leaky_passed": demo["audit_leaky"]["passed"]}
        print(format_audit(demo["audit_clean"], "Leakage audit, real features"))
        print(format_audit(demo["audit_leaky"], f"Leakage audit, with {LEAKY_DEMO_COLUMN} added"))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(linear, MODELS_DIR / "linear_eval.joblib")
    joblib.dump(forest, MODELS_DIR / "random_forest_eval.joblib")

    print("Refitting the shipped model on all seasons...")
    final = make_forest(**rf_params).fit(df[FEATURES], df[TARGET])
    joblib.dump({"model": final, "features": FEATURES, "trained_through": result["data_through"],
                 "params": rf_params}, MODELS_DIR / "model.joblib", compress=3)

    out = test[["game_pk", "game_date", "season", "pitcher_id", "pitcher_name", TARGET,
                "batters_faced", "season_k_rate"]].copy()
    for name, p in preds.items():
        out[f"pred_{name}"] = p
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS_PATH, index=False)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "latest.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result


def print_table(result: dict) -> None:
    """Print the model comparison table."""
    rows = pd.DataFrame(result["metrics"]).T[["mae", "rmse", "r2", "mae_improvement_pct", "rmse_improvement_pct"]]
    print(f"\nTest season {result['test_season']} ({result['n_test']:,} starts)")
    print(rows.to_string())
    if "ablation" in result:
        print("\nTest MAE by feature set (linear / random forest):")
        for name, a in result["ablation"].items():
            print(f"  {name:<16} {a['n_features']:>2} features  "
                  f"{a['linear']['mae']:.3f} / {a['random_forest']['mae']:.3f}")
    if "leak_demo" in result:
        d = result["leak_demo"]
        print(f"\nWith leaky '{d['leaky_feature']}': RF MAE {d['random_forest']['mae']:.3f} "
              f"({d['random_forest']['mae_drop_vs_clean_pct']}% lower than the honest model), "
              f"R² {d['random_forest']['r2']:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-season", type=int, default=None)
    parser.add_argument("--skip-leak-demo", action="store_true")
    args = parser.parse_args(argv)

    if not FEATURES_PATH.exists():
        print("No feature table. Run `python src/build_features.py` first.")
        return 1
    df = pd.read_parquet(FEATURES_PATH)
    result = train_and_evaluate(df, args.test_season, run_leak_demo=not args.skip_leak_demo)
    print_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

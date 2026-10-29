"""Which model and setup predicts strikeouts best?

Every candidate is fit on all seasons before a test season and scored on that
season (2025 and 2026 by default), the same split ``src/train.py`` uses, on
the same pre-game features. Candidates:

* ``linear (current)``: the shipped linear regression
* ``ridge`` / ``lasso``: linear with L2 / L1 shrinkage (alpha picked by CV on
  the training seasons)
* ``poisson regression``: a GLM for counts (log link), since strikeouts are counts
* ``random forest``: the forest from ``src/train.py`` with its tuned settings
* ``gradient boosting``: scikit-learn HistGradientBoosting with squared,
  absolute-error and Poisson losses
* ``two-stage BF x K rate``: predict batters faced and K per batter faced
  separately, then multiply (how deep he goes times how often he strikes
  hitters out)
* ``blend linear + boosting``: the average of the linear and Poisson-boosting
  predictions
* ``recency half-life 3 / 8``: linear with the recency-weighted K stats
  recomputed with a 3- or 8-start half-life instead of 5

Usage::

    python src/compare_models.py
    python src/compare_models.py --seasons 2026
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, PoissonRegressor, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_features import FEATURES, FEATURES_PATH, TARGET
from evaluate import regression_metrics
from train import baseline_predict, make_forest, make_linear

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "model_comparison.md"
SEED = 42
RF_PARAMS = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 10}


def linear_with(estimator) -> Pipeline:
    """Same preprocessing as the shipped linear model, with another final estimator."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("lr", estimator),
    ])


def boosting(loss: str) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss=loss, learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.1, random_state=SEED)


def add_ewm(df: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Recency-weighted K rate and K per start with another half-life (overwrites the columns)."""
    df = df.sort_values(["pitcher_id", "game_date", "game_pk"])
    g = df.groupby("pitcher_id")
    k = g[TARGET].transform(lambda s: s.shift(1).ewm(halflife=halflife).mean())
    bf = g["batters_faced"].transform(lambda s: s.shift(1).ewm(halflife=halflife).mean())
    df["k_per_start_ewm"] = k
    df["k_rate_ewm"] = k / bf.where(bf > 0)
    return df.sort_values(["game_date", "game_pk", "pitcher_id"])


def fit_predict(name: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    X, y, Xt = train[FEATURES], train[TARGET], test[FEATURES]
    if name == "linear (current)":
        return make_linear().fit(X, y).predict(Xt)
    if name == "ridge":
        return linear_with(RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(X, y).predict(Xt)
    if name == "lasso":
        return linear_with(LassoCV(alphas=30, cv=3, random_state=SEED, max_iter=5000)).fit(X, y).predict(Xt)
    if name == "poisson regression":
        return linear_with(PoissonRegressor(alpha=1e-3, max_iter=1000)).fit(X, y).predict(Xt)
    if name == "random forest":
        return make_forest(**RF_PARAMS).fit(X, y).predict(Xt)
    if name.startswith("gradient boosting"):
        loss = name.split("(")[1].rstrip(")")
        return boosting(loss).fit(X, y).predict(Xt)
    if name == "two-stage BF x K rate":
        bf = make_linear().fit(X, train["batters_faced"]).predict(Xt)
        bf_train = train["batters_faced"].where(train["batters_faced"] > 0)
        ok = bf_train.notna()
        rate = make_linear().fit(X[ok], (y / bf_train)[ok], lr__sample_weight=bf_train[ok]).predict(Xt)
        return bf * rate
    raise ValueError(name)


MODELS = [
    "linear (current)", "ridge", "lasso", "poisson regression", "random forest",
    "gradient boosting (squared_error)", "gradient boosting (absolute_error)", "gradient boosting (poisson)",
    "two-stage BF x K rate",
]


def evaluate_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    train, test = df[df["season"] < season], df[df["season"] == season]
    y = test[TARGET].to_numpy()
    preds, rows = {}, {}
    for name in MODELS:
        t0 = time.time()
        preds[name] = np.clip(fit_predict(name, train, test), 0, None)
        rows[name] = {**regression_metrics(y, preds[name]), "seconds": round(time.time() - t0, 1)}
        print(f"  {name}: MAE {rows[name]['mae']:.4f}")
    blend = (preds["linear (current)"] + preds["gradient boosting (poisson)"]) / 2
    rows["blend linear + boosting"] = {**regression_metrics(y, blend), "seconds": np.nan}
    for hl in (3, 8):
        d = add_ewm(df, hl)
        tr, te = d[d["season"] < season], d[d["season"] == season]
        p = np.clip(make_linear().fit(tr[FEATURES], tr[TARGET]).predict(te[FEATURES]), 0, None)
        rows[f"linear, recency half-life {hl}"] = {**regression_metrics(te[TARGET].to_numpy(), p), "seconds": np.nan}
    base = baseline_predict(test, float(train[TARGET].mean()))
    rows["naive baseline"] = {**regression_metrics(y, base), "seconds": np.nan}
    return pd.DataFrame(rows).T[["mae", "rmse", "r2", "seconds"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    args = parser.parse_args(argv)
    missing = [c for c in ("k_rate_ewm", "k_per_start_ewm") if c not in FEATURES]
    if missing:
        print(f"FEATURES lacks {missing}; rebuild with the current build_features.py first.")
        return 1

    df = pd.read_parquet(FEATURES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "pitcher_id"]).reset_index(drop=True)

    lines = ["# Model comparison", "", "Fit on all seasons before the test season; lower MAE is better.", ""]
    tables = []
    for season in args.seasons:
        print(f"Test season {season}...")
        table = evaluate_season(df, season)
        tables.append(table["mae"].rename(season))
        print(table.round(4).to_string())
        lines += [f"## {season}", "", "| Model | MAE | RMSE | R² |", "|---|---|---|---|"]
        lines += [f"| {n} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['r2']:.4f} |" for n, r in table.iterrows()]
        lines.append("")
    summary = pd.concat(tables, axis=1)
    summary["mean MAE"] = summary.mean(axis=1)
    summary = summary.sort_values("mean MAE")
    print("\nMAE by season, best first\n" + summary.round(4).to_string())
    cols = [str(c) for c in summary.columns]
    lines += ["## MAE by season, best first", "", f"| Model | {' | '.join(cols)} |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {n} | " + " | ".join(f"{v:.4f}" for v in r) + " |" for n, r in summary.iterrows()]
    lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nWrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Metrics, plots, feature importance, and a written interpretation.

Reads the test-set predictions and models written by ``train.py`` and produces:

* ``reports/results.md``: comparison table (MAE, RMSE, R², % improvement over
  the baseline), leakage check results, a noise-floor analysis, and a short
  interpretation with the numbers filled in.
* ``reports/figures/feature_importance.png``: permutation importance of the
  random forest on the test season (plus impurity importance for reference).
* ``reports/figures/predicted_vs_actual.png``
* ``reports/figures/residuals.png``
* ``metrics/latest.json`` gains a ``noise`` section.

Noise floor
-----------
Each plate appearance is close to a coin flip with the pitcher's true K rate.
Even a model that knew that rate *and* exactly how many batters the starter
would face could not beat binomial noise: ``Var(K) = BF * p * (1 - p)``. The
average of that over the test starts (``p`` = the pre-game season K rate,
``BF`` = actual batters faced) is the irreducible "coin-flip" variance, and
``1 - floor / Var(K)`` is an upper bound on R² for any pre-game model. The
gap between a model's MSE and that floor is everything else: unknown
workload (hooks, pitch limits), umpires, matchups, and skill estimation error.

Usage::

    python src/evaluate.py
    python src/evaluate.py --gate    # exit 1 if RF test MAE is worse than baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_PATH = ROOT / "metrics" / "latest.json"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "test_predictions.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
RF_EVAL_PATH = ROOT / "models" / "random_forest_eval.joblib"

MODEL_LABELS = {"baseline": "Naive baseline", "linear": "Linear regression", "random_forest": "Random forest"}
# Validated categorical slots 1-3 (blue, orange, aqua); text stays in neutral ink.
COLORS = {"baseline": "#2a78d6", "linear": "#eb6834", "random_forest": "#1baf7a"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEED = 42


def regression_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE and R² rounded to 4 decimals."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def noise_analysis(y, bf, rate, preds: dict[str, np.ndarray]) -> dict:
    """Split single-game K variance into binomial coin-flip noise and the rest.

    ``bf`` is actual batters faced and ``rate`` the pitcher's pre-game K/BF
    estimate; see the module docstring for the reasoning.
    """
    y, bf = np.asarray(y, float), np.asarray(bf, float)
    rate = np.nan_to_num(np.asarray(rate, float), nan=float(np.nansum(y) / np.nansum(bf)))
    var_y = float(np.var(y))
    floor = float(np.mean(bf * rate * (1 - rate)))
    explainable = var_y - floor
    out = {
        "var_k": round(var_y, 3),
        "mean_k": round(float(np.mean(y)), 3),
        "binomial_noise_floor": round(floor, 3),
        "noise_share_of_variance": round(floor / var_y, 3) if var_y > 0 else None,
        "r2_ceiling_estimate": round(1 - floor / var_y, 3) if var_y > 0 else None,
        "models": {},
    }
    for name, p in preds.items():
        mse = float(np.mean((y - p) ** 2))
        out["models"][name] = {
            "mse": round(mse, 3),
            "share_of_explainable_variance_captured": round((var_y - mse) / explainable, 3)
            if explainable > 0 else None,
        }
    return out


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def plot_feature_importance(model, X: pd.DataFrame, y: np.ndarray, features: list[str], path: Path) -> pd.DataFrame:
    """Plot permutation importance on the test set (MAE increase when a feature is shuffled)."""
    perm = permutation_importance(model, X, y, scoring="neg_mean_absolute_error",
                                  n_repeats=10, random_state=SEED, n_jobs=-1)
    rf = model.named_steps["rf"]
    impurity = pd.Series(rf.feature_importances_[: len(features)], index=features)
    imp = pd.DataFrame({
        "permutation_mae_increase": perm.importances_mean,
        "permutation_std": perm.importances_std,
        "impurity": impurity.reindex(features).to_numpy(),
    }, index=features).sort_values("permutation_mae_increase")

    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(features) + 1.2))
    ax.barh(imp.index, imp["permutation_mae_increase"], xerr=imp["permutation_std"],
            color=COLORS["random_forest"], height=0.6, error_kw={"ecolor": INK_MUTED, "lw": 1})
    ax.set_xlabel("Increase in test MAE when the feature is shuffled (strikeouts)")
    ax.set_title("Random forest feature importance (permutation, test season)", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp.sort_values("permutation_mae_increase", ascending=False)


def plot_pred_vs_actual(df: pd.DataFrame, path: Path) -> None:
    """Scatter of random forest predictions vs. actual K, with integer K jittered."""
    rng = np.random.default_rng(SEED)
    y = df["strikeouts"].to_numpy(float)
    p = df["pred_random_forest"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(p, y + rng.uniform(-0.3, 0.3, len(y)), s=8, alpha=0.25,
               color=COLORS["random_forest"], linewidths=0)
    lim = [0, max(y.max(), p.max()) + 1]
    ax.plot(lim, lim, color=INK_MUTED, lw=1.5, ls="--", label="Perfect prediction")
    bins = pd.cut(p, np.arange(0, np.ceil(p.max()) + 0.5, 0.5))
    binned = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
    ax.plot(binned["p"], binned["y"], color=COLORS["linear"], lw=2, marker="o", ms=5,
            label="Mean actual K per 0.5-K prediction bin")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted strikeouts (random forest)")
    ax.set_ylabel("Actual strikeouts (jittered)")
    ax.set_title("Predicted vs. actual, test season", loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(df: pd.DataFrame, path: Path) -> None:
    """Overlaid residual (actual - predicted) distributions for all three models."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.arange(-9, 9.5, 0.5)
    for name in MODEL_LABELS:
        r = df["strikeouts"] - df[f"pred_{name}"]
        ax.hist(r, bins=bins, histtype="step", lw=2, color=COLORS[name],
                label=f"{MODEL_LABELS[name]} (sd {r.std():.2f})")
    ax.axvline(0, color=INK_MUTED, lw=1)
    ax.set_xlabel("Residual: actual minus predicted strikeouts")
    ax.set_ylabel("Starts")
    ax.set_title("Residual distribution, test season", loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def interpretation(result: dict, noise: dict, top_features: list[str]) -> str:
    """Write the plain-language interpretation with this run's numbers."""
    m = result["metrics"]
    best = min(("linear", "random_forest"), key=lambda k: m[k]["mae"])
    nb, nl = m["baseline"], m[best]
    ceiling = noise["r2_ceiling_estimate"]
    captured = noise["models"][best]["share_of_explainable_variance_captured"]
    leak = result.get("leak_demo")
    text = [
        f"The {MODEL_LABELS[best].lower()} is the best model: test MAE {nl['mae']:.2f} strikeouts vs. "
        f"{nb['mae']:.2f} for the naive baseline ({nl['mae_improvement_pct']:.1f}% better), "
        f"RMSE {nl['rmse']:.2f} vs. {nb['rmse']:.2f}, and R² {nl['r2']:.3f} vs. {nb['r2']:.3f}.",
        "",
        f"Most of the single-game variance is not predictable before first pitch. Strikeouts in the test "
        f"season have variance {noise['var_k']:.2f}. Plate-appearance coin-flip noise alone accounts for about "
        f"{noise['binomial_noise_floor']:.2f} of it ({noise['noise_share_of_variance']:.0%}), even for a model "
        f"that knew each pitcher's true K rate and exactly how many batters they would face, which caps R² near "
        f"{ceiling:.2f}. The {MODEL_LABELS[best].lower()} reaches R² {nl['r2']:.3f}, about {captured:.0%} of "
        f"the variance that is explainable in principle.",
        "",
        "The remaining error comes mostly from things no pre-game feature sees: how long the manager leaves the "
        "starter in (an early hook or a pitch-count limit removes batters faced, the biggest driver of the "
        "count), the home-plate umpire's zone, in-game injuries and blowouts, and plate-appearance-level "
        "randomness in matchups. Batters faced varies a lot from start to start and the model can only guess "
        "it from recent workload.",
        "",
        f"The most useful features on the test season were {', '.join(top_features[:3])} "
        f"(permutation importance).",
    ]
    if leak:
        rf = leak["random_forest"]
        text += ["", f"Leakage sanity check: adding the current game's batters faced drops random forest MAE to "
                     f"{rf['mae']:.2f} ({rf['mae_drop_vs_clean_pct']:.0f}% lower) and lifts R² to {rf['r2']:.2f}, "
                     f"and the leakage audit flags it. The honest features pass the same audit."]
    return "\n".join(text)


def ablation_table(result: dict) -> str:
    """Markdown table of test MAE for nested feature sets (core, + matchup, + context, all)."""
    ab = result.get("ablation")
    if not ab:
        return ""
    rows = ["| Feature set | Features | Linear MAE | RF MAE | RF R² |", "|---|---:|---:|---:|---:|"]
    for name, a in ab.items():
        rows.append(f"| {name} | {a['n_features']} | {a['linear']['mae']:.3f} | "
                    f"{a['random_forest']['mae']:.3f} | {a['random_forest']['r2']:.3f} |")
    return "\n".join(rows) + "\n\nCore = pitcher form, workload, rest, home/away, opponent K rate. " \
        "Matchup = lineup handedness and K rates, pitcher platoon splits, pitch mix vs. lineup whiff rates. " \
        "Context = weather, day/night, park and umpire factors, opponent last-10 form.\n"


def write_report(result: dict, noise: dict, imp: pd.DataFrame, path: Path) -> None:
    """Write reports/results.md."""
    table = pd.DataFrame(result["metrics"]).T
    table.index = [MODEL_LABELS[i] for i in table.index]
    table = table[["mae", "rmse", "r2", "mae_improvement_pct", "rmse_improvement_pct"]]
    table.columns = ["MAE", "RMSE", "R²", "MAE vs baseline (%)", "RMSE vs baseline (%)"]
    md_table = "| Model | " + " | ".join(table.columns) + " |\n|---|" + "---:|" * len(table.columns) + "\n"
    for name, row in table.iterrows():
        cells = [f"{v:.3f}" for v in row.iloc[:3]] + [f"{v:+.1f}" for v in row.iloc[3:]]
        md_table += f"| {name} | " + " | ".join(cells) + " |\n"

    audit = result.get("leakage_audit", {})
    lines = [
        "# Results",
        "",
        f"Train seasons {result['train_seasons'][0]}-{result['train_seasons'][-1]} "
        f"({result['n_train']:,} starts), test season {result['test_season']} ({result['n_test']:,} starts), "
        f"data through {result['data_through']}. Random forest params: `{result['rf_best_params']}`.",
        "",
        md_table,
        "## Interpretation",
        "",
        interpretation(result, noise, list(imp.index)),
        "",
        "## What each feature group adds",
        "",
        ablation_table(result),
        "## Leakage audit",
        "",
        f"- Real feature set: **{'passed' if audit.get('clean_passed') else 'FAILED'}**",
        f"- With current-game batters faced added: **{'passed' if audit.get('leaky_passed') else 'caught'}**",
        "",
        "## Figures",
        "",
        "![Feature importance](figures/feature_importance.png)",
        "![Predicted vs actual](figures/predicted_vs_actual.png)",
        "![Residuals](figures/residuals.png)",
        "",
    ]
    path.write_text("\n".join(lines))


def quality_gate(result: dict) -> bool:
    """True if the random forest's test MAE is no worse than the baseline's."""
    return result["metrics"]["random_forest"]["mae"] <= result["metrics"]["baseline"]["mae"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", action="store_true",
                        help="exit 1 if the random forest's test MAE is worse than the baseline's")
    args = parser.parse_args(argv)

    if not METRICS_PATH.exists() or not PREDICTIONS_PATH.exists():
        print("No training results. Run `python src/train.py` first.")
        return 1
    result = json.loads(METRICS_PATH.read_text())
    preds = pd.read_parquet(PREDICTIONS_PATH)
    y = preds["strikeouts"].to_numpy(float)

    noise = noise_analysis(y, preds["batters_faced"], preds["season_k_rate"],
                           {k: preds[f"pred_{k}"].to_numpy(float) for k in MODEL_LABELS})
    result["noise"] = noise
    result["quality_gate_passed"] = quality_gate(result)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    features = result["features"]
    test = pd.read_parquet(FEATURES_PATH)
    test = test[test["season"] == result["test_season"]]
    imp = plot_feature_importance(joblib.load(RF_EVAL_PATH), test[features], test["strikeouts"].to_numpy(float),
                                  features, FIG_DIR / "feature_importance.png")
    result["feature_importance"] = imp["permutation_mae_increase"].round(4).to_dict()
    plot_pred_vs_actual(preds, FIG_DIR / "predicted_vs_actual.png")
    plot_residuals(preds, FIG_DIR / "residuals.png")
    write_report(result, noise, imp, REPORTS_DIR / "results.md")
    METRICS_PATH.write_text(json.dumps(result, indent=2, default=float) + "\n")

    print((REPORTS_DIR / "results.md").read_text())
    if args.gate and not result["quality_gate_passed"]:
        rf, base = result["metrics"]["random_forest"]["mae"], result["metrics"]["baseline"]["mae"]
        print(f"QUALITY GATE FAILED: random forest MAE {rf:.3f} is worse than baseline {base:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

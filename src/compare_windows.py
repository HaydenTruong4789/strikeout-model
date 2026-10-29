"""Does training only on recent starts beat training on all history?

Simulates the live setup: every ``--refit-days`` days (5, matching the data
update), the linear regression is refit on starts before that day and predicts
the starts in the next ``--refit-days`` days. Training windows compared:

* ``all history``: every earlier start (what the app does now)
* ``last 3 / 6 / 12 months``: starts in that many calendar days
  before the refit day. Early in a season the short windows hold little more
  than the previous September and postseason.
* ``last ~1 season of starts``: the most recent ``SEASON_STARTS`` starts, so
  six months of *baseball* rather than six calendar months.

Two "last 10 starts" variants, both trained on all history:

* ``season stats -> last 10 starts``: the pitcher's season-to-date K rate and
  K per start are replaced by the same stats over his previous 10 starts
  (carried across seasons), so no pitcher feature reaches back further than
  10 starts. (His platoon split vs the lineup still uses all his history.)
* ``train on each pitcher's last 10 starts``: the training rows are only each
  pitcher's 10 most recent starts before the refit day.

``--last-n 5 6 7 8 9 10`` runs only those two variants for each N instead,
next to all history.

``--emphasis`` keeps all history but leans on each pitcher's last 5 starts:

* ``+ last-5 K per start``: his K per start over his last 5 starts added as a
  feature (last-5 K rate, batters faced and pitches are already features).
* ``+ recency-weighted K stats``: K rate and K per start over all his starts,
  each start weighted by 0.5 ** (starts ago / 5), added as features.
* ``weight last 5 starts x3``: training rows that are among each pitcher's 5
  most recent starts count three times as much in the fit.
* ``weight by recency (1-year half-life)``: every training row weighted by
  0.5 ** (days before the refit day / 365).

Plus the current shipped evaluation: one fit on all earlier seasons.

Apart from the first variant, the features are the same everywhere: most
already look only at recent games (last 3-5 starts, opponent's last 10-30 games).

Usage::

    python src/compare_windows.py                    # test seasons 2025 and 2026
    python src/compare_windows.py --seasons 2026
    python src/compare_windows.py --last-n 5 6 7 8 9 10
    python src/compare_windows.py --emphasis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import FEATURES, FEATURES_PATH, TARGET
from train import baseline_predict, make_linear

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "training_window.md"
LAST_N_REPORT_PATH = ROOT / "reports" / "last_n_starts.md"
EMPHASIS_REPORT_PATH = ROOT / "reports" / "recent_emphasis.md"
EWM_HALFLIFE_STARTS = 5
SEASON_STARTS = 4900
WINDOWS = {
    "all history": None,
    "last 12 months": pd.Timedelta(days=365),
    "last 6 months": pd.Timedelta(days=183),
    "last 3 months": pd.Timedelta(days=91),
    "last ~1 season of starts": SEASON_STARTS,
    "train on each pitcher's last 10 starts": ("pitcher_last", 10),
}


def last_n_features(n: int) -> list[str]:
    """FEATURES with season-to-date K rate and K per start swapped for their last-``n``-start versions."""
    swap = {"season_k_rate": f"k_rate_last{n}", "season_k_per_start": f"k_per_start_last{n}"}
    return [swap.get(f, f) for f in FEATURES]


def make_variants(last_n: list[int] | None, emphasis: bool = False) -> dict:
    """Variant name -> (training window, feature list[, sample-weight rule])."""
    if emphasis:
        return {
            "all history": (None, FEATURES),
            "+ last-5 K per start": (None, [*FEATURES, "k_per_start_last5"]),
            "+ recency-weighted K stats": (None, [*FEATURES, *(c for c in ("k_rate_ewm", "k_per_start_ewm")
                                                               if c not in FEATURES)]),
            "weight last 5 starts x3": (None, FEATURES, "last5x3"),
            "weight by recency (1-year half-life)": (None, FEATURES, "decay365"),
        }
    if not last_n:
        variants = {name: (window, FEATURES) for name, window in WINDOWS.items()}
        variants["season stats -> last 10 starts"] = (None, last_n_features(10))
        return variants
    variants = {"all history": (None, FEATURES)}
    for n in last_n:
        variants[f"season stats -> last {n} starts"] = (None, last_n_features(n))
    for n in last_n:
        variants[f"train on each pitcher's last {n} starts"] = (("pitcher_last", n), FEATURES)
    return variants


def add_last_n(df: pd.DataFrame, ns) -> pd.DataFrame:
    """K rate and K per start over each pitcher's previous ``n`` starts (current start excluded)."""
    df = df.sort_values(["pitcher_id", "game_date", "game_pk"])
    g = df.groupby("pitcher_id")
    for w in ns:
        k = g[TARGET].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())
        bf = g["batters_faced"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())
        n = g[TARGET].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).count())
        df[f"k_rate_last{w}"] = k / bf.where(bf > 0)
        df[f"k_per_start_last{w}"] = k / n.where(n > 0)
    return df.sort_values(["game_date", "game_pk", "pitcher_id"])


def add_ewm(df: pd.DataFrame, halflife: float = EWM_HALFLIFE_STARTS) -> pd.DataFrame:
    """Recency-weighted K rate and K per start over all of a pitcher's previous starts."""
    df = df.sort_values(["pitcher_id", "game_date", "game_pk"])
    g = df.groupby("pitcher_id")
    k = g[TARGET].transform(lambda s: s.shift(1).ewm(halflife=halflife).mean())
    bf = g["batters_faced"].transform(lambda s: s.shift(1).ewm(halflife=halflife).mean())
    df["k_per_start_ewm"] = k
    df["k_rate_ewm"] = k / bf.where(bf > 0)
    return df.sort_values(["game_date", "game_pk", "pitcher_id"])


def sample_weights(train: pd.DataFrame, day: pd.Timestamp, rule: str | None) -> np.ndarray | None:
    """Per-row training weights for a weighting rule (None means equal weights)."""
    if rule is None:
        return None
    if rule == "last5x3":
        recent = train.groupby("pitcher_id").cumcount(ascending=False) < 5
        return np.where(recent, 3.0, 1.0)
    if rule == "decay365":
        return 0.5 ** ((day - train["game_date"]).dt.days.to_numpy() / 365)
    raise ValueError(rule)


def training_rows(df: pd.DataFrame, day: pd.Timestamp, window) -> pd.DataFrame:
    """Starts before ``day`` inside the window."""
    past = df[df["game_date"] < day]
    if window is None:
        return past
    if isinstance(window, tuple):  # ("pitcher_last", n)
        return past.groupby("pitcher_id").tail(window[1])
    if isinstance(window, int):
        return past.tail(window)
    return past[past["game_date"] >= day - window]


def walk_forward(df: pd.DataFrame, season: int, refit_days: int, variants: dict) -> pd.DataFrame:
    """Predictions for every start in ``season`` under each variant."""
    test = df[df["season"] == season].copy()
    start, end = test["game_date"].min(), test["game_date"].max()
    for name in variants:
        test[name] = np.nan
    day = start
    while day <= end:
        block = (test["game_date"] >= day) & (test["game_date"] < day + pd.Timedelta(days=refit_days))
        if block.any():
            for name, (window, cols, *rule) in variants.items():
                train = training_rows(df, day, window)
                if train.empty:
                    continue
                weights = sample_weights(train, day, rule[0] if rule else None)
                model = make_linear().fit(train[cols], train[TARGET], lr__sample_weight=weights)
                test.loc[block, name] = np.clip(model.predict(test.loc[block, cols]), 0, None)
        day += pd.Timedelta(days=refit_days)

    prior = df[df["season"] < season]
    test["season fit, all earlier seasons"] = np.clip(
        make_linear().fit(prior[FEATURES], prior[TARGET]).predict(test[FEATURES]), 0, None)
    test["naive baseline"] = baseline_predict(test, float(prior[TARGET].mean()))
    return test


def summarize(test: pd.DataFrame, variants: dict) -> pd.DataFrame:
    cols = [*variants, "season fit, all earlier seasons", "naive baseline"]
    y = test[TARGET]
    month = test["game_date"].dt.month.clip(lower=4, upper=9)  # March into April, Oct into Sept
    rows = {}
    for c in cols:
        err = (y - test[c]).abs()
        rows[c] = {"MAE": err.mean(), "RMSE": np.sqrt(((y - test[c]) ** 2).mean()),
                   "MAE Apr-May": err[month <= 5].mean(), "MAE Jun-Sep": err[month >= 6].mean()}
    return pd.DataFrame(rows).T


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--refit-days", type=int, default=5)
    parser.add_argument("--last-n", type=int, nargs="+", help="compare last-N-start variants for these N")
    parser.add_argument("--emphasis", action="store_true", help="all history with extra weight on recent starts")
    args = parser.parse_args(argv)
    variants = make_variants(args.last_n, args.emphasis)

    df = pd.read_parquet(FEATURES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = add_ewm(add_last_n(df, sorted({5, 10, *(args.last_n or [])}))).reset_index(drop=True)

    lines = ["# Training window comparison", "",
             f"Linear regression refit every {args.refit_days} days on starts before that day; "
             "lower MAE is better.", ""]
    for season in args.seasons:
        table = summarize(walk_forward(df, season, args.refit_days, variants), variants).round(3)
        n = int((df["season"] == season).sum())
        print(f"\nTest season {season} ({n:,} starts)\n{table.to_string()}")
        lines += [f"## {season} ({n:,} starts)", "", "| Training data | MAE | RMSE | MAE Apr-May | MAE Jun-Sep |",
                  "|---|---|---|---|---|"]
        lines += [f"| {name} | {r['MAE']:.3f} | {r['RMSE']:.3f} | {r['MAE Apr-May']:.3f} | {r['MAE Jun-Sep']:.3f} |"
                  for name, r in table.iterrows()]
        lines.append("")
    path = EMPHASIS_REPORT_PATH if args.emphasis else LAST_N_REPORT_PATH if args.last_n else REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\nWrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

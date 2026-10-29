"""Test recent-form and finer pitch-matchup features one change at a time, with Hayden's rule.

Same method as ``src/test_chase.py``: baseline is the live blend with the
current features; each test adds one thing; fit on seasons before the test
season, score on 2025 and 2026; paired bootstrap over game dates on the MAE
difference. A change is kept only if it is better in both seasons with both
95% intervals below zero. Differences under 0.005 that don't hold in both
seasons are noise.

Before the tests, a descriptive check of bounce-back and momentum: the mean
residual (actual minus out-of-sample live-blend prediction) of a start, grouped
by how the pitcher's previous starts went. If the model already sees form
through its rolling and recency-weighted features, these residuals sit near 0.

Every new column also gets the leakage check: rebuilt from data cut off before
three dates, it must match exactly.

Usage::

    python src/test_form_matchup.py
    python src/test_form_matchup.py --boot 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import FEATURES, FEATURES_PATH, TARGET
from compare_features import NOISE, SEASONS, paired_bootstrap, predict, verdict
from fetch_data import load_game_logs, load_lineups
from form_features import FORM_FEATURES, add_form_features, build_runs_cache
from pitch_matchup_features import PITCH_MATCHUP_FEATURES, add_pitch_matchup_features, build_fine_cache
from train import make_blend

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "form_matchup_test.md"
NEW = FORM_FEATURES + PITCH_MATCHUP_FEATURES
TESTS = [
    ("+ last start K surprise", ["last_start_k_surprise"]),
    ("+ bad starts in last 5", ["bad_starts_last5"]),
    ("+ good starts in last 5", ["good_starts_last5"]),
    ("+ last start short (BF <= 15)", ["last_start_short"]),
    ("+ fine pitch-type x velo whiff matchup", ["matchup_whiff_fine"]),
    ("+ fine pitch-type x velo chase matchup", ["matchup_chase_fine"]),
    ("+ both fine matchups", PITCH_MATCHUP_FEATURES),
]
DESC_SEASONS = [2024, 2025, 2026]


def group_residuals(df: pd.DataFrame, col: str, labels: dict) -> list[dict]:
    """Mean residual, n and a normal 95% interval for each group of ``col``."""
    out = []
    for label, mask in labels.items():
        r = df.loc[mask(df[col]), "resid"].dropna()
        se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan
        out.append(dict(group=label, n=len(r), mean=r.mean(), lo=r.mean() - 1.96 * se, hi=r.mean() + 1.96 * se))
    return out


def descriptive(df: pd.DataFrame) -> list[tuple[str, list[dict]]]:
    """Residuals of out-of-sample predictions grouped by recent form."""
    parts = []
    for s in DESC_SEASONS:
        tr, te = df[df["season"] < s], df[df["season"] == s]
        pred = np.clip(make_blend().fit(tr[FEATURES], tr[TARGET]).predict(te[FEATURES]), 0, None)
        parts.append(te.assign(resid=te[TARGET].to_numpy(float) - pred))
    d = pd.concat(parts)
    counts = {"0": lambda x: x == 0, "1": lambda x: x == 1, "2": lambda x: x == 2, "3+": lambda x: x >= 3}
    flag = {"yes": lambda x: x == 1, "no": lambda x: x == 0}
    return [
        ("All starts", group_residuals(d.assign(_all=1), "_all", {"all": lambda x: x == 1})),
        ("Last start bad", group_residuals(d, "last_start_bad", flag)),
        ("Last start good", group_residuals(d, "last_start_good", flag)),
        ("Bad starts in last 5", group_residuals(d, "bad_starts_last5", counts)),
        ("Good starts in last 5", group_residuals(d, "good_starts_last5", counts)),
        ("Last 3 starts all bad", group_residuals(d, "bad_streak3", flag)),
        ("Last 3 starts all good", group_residuals(d, "good_streak3", flag)),
        ("Last start short (BF <= 15)", group_residuals(d, "last_start_short", flag)),
    ]


def leakage_check(df, logs, runs, lineups, fine, n_dates: int = 3) -> list[str]:
    """Rebuild the new columns with every input cut off before a few dates; values must match."""
    rng = np.random.default_rng(0)
    dates = sorted(rng.choice(df.loc[df["season"] >= 2025, "game_date"].unique(), n_dates, replace=False))
    out = []
    key = ["game_pk", "pitcher_id"]
    for d in dates:
        d = pd.Timestamp(d)
        rows = df[df["game_date"] == d].set_index(key).index
        hist = df[df["game_date"] <= d]
        # Today's own outcomes are blanked: a pre-game rebuild can't know them.
        hist = hist.assign(**{TARGET: hist[TARGET].where(hist["game_date"] < d)})
        lg = logs[pd.to_datetime(logs["game_date"]) < d]
        rn = runs[pd.to_datetime(runs["game_date"]) < d]
        fn = {k: v[pd.to_datetime(v["game_date"]) < d] for k, v in fine.items()}
        lu = lineups[pd.to_datetime(lineups["game_date"]) <= d]
        a = add_form_features(hist.copy(), lg, rn)
        a = add_pitch_matchup_features(a, lu, fn).set_index(key).loc[rows]
        b = df.set_index(key).loc[rows]
        for c in NEW:
            same = np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)
            out.append(f"{d.date()} {c}: {'ok' if same else 'MISMATCH'}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args(argv)

    print("Runs and fine pitch counts from raw Statcast...")
    runs = build_runs_cache()
    fine = build_fine_cache()
    logs, lineups = load_game_logs(), load_lineups()
    df = pd.read_parquet(FEATURES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "pitcher_id"]).reset_index(drop=True)
    df = add_form_features(df, logs, runs)
    df = add_pitch_matchup_features(df, lineups, fine)
    cover = df[NEW].notna().mean().round(3)
    corr = df[NEW + [TARGET]].corr()[TARGET].drop(TARGET).round(3)
    print("Coverage:\n" + cover.to_string())
    print("Correlation with strikeouts:\n" + corr.to_string())
    share = df[["bad_start", "good_start"]].mean().round(3)
    print("Share of starts bad / good:\n" + share.to_string())

    print("Descriptive residuals...")
    desc = descriptive(df)
    for title, rows in desc:
        print(f"  {title}: " + ", ".join(f"{r['group']} {r['mean']:+.3f} (n={r['n']})" for r in rows))

    print("Leakage check...")
    leak = leakage_check(df, logs, runs, lineups, fine)
    print("\n".join(leak))

    rows, base_pred = [], {}
    for name, extra in TESTS:
        t0, per = time.time(), []
        for s in SEASONS:
            tr, te = df[df["season"] < s], df[df["season"] == s]
            y = te[TARGET].to_numpy(float)
            if s not in base_pred:
                base_pred[s] = predict({"model": "blend"}, tr, te)
            b, n = base_pred[s], predict({"model": "blend", "cols": FEATURES + extra}, tr, te)
            diff, lo, hi = paired_bootstrap(y, b, n, te["game_date"], args.boot)
            per.append(dict(season=s, base=float(np.abs(y - b).mean()), new=float(np.abs(y - n).mean()),
                            diff=diff, lo=lo, hi=hi))
        v = verdict(per, "change")
        rows.append((name, per, v))
        print(f"  {name}: " + ", ".join(f"{r['season']} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                                        for r in per) + f" -> {v} ({time.time() - t0:.0f}s)")

    lines = ["# Recent form and fine pitch-matchup feature tests", "",
             "Baseline: the live blend with its current features. Each test adds one feature (or group). Fit on "
             "seasons before the test season, scored on it. Diff = MAE with the change minus baseline MAE "
             "(negative is better), with a 95% paired-bootstrap interval over game dates. Kept only if better in "
             f"both seasons with both intervals below zero. Under {NOISE} and not in both seasons is noise.", "",
             "| Test | " + " | ".join(f"MAE {s} base → new | Diff {s} [95% CI]" for s in SEASONS) + " | Verdict |",
             "|---" * (2 * len(SEASONS) + 2) + "|"]
    for name, per, v in rows:
        cells = " | ".join(f"{r['base']:.4f} → {r['new']:.4f} | {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                           for r in per)
        lines.append(f"| {name} | {cells} | {v} |")
    lines += ["", "## Bounce-back and momentum: residuals the model leaves", "",
              "Residual = actual K minus the live blend's out-of-sample prediction (fit on earlier seasons), "
              f"for {DESC_SEASONS[0]}-{DESC_SEASONS[-1]} starts. Positive means the pitcher beat the model. "
              f"A bad start: K/BF more than {0.08} below his recency-weighted K rate, 4+ runs while he pitched "
              "(from Statcast; game logs have no earned runs), or under 4 innings. A good start: K/BF 0.08 "
              "above, or 7+ K with 0-1 runs.", "",
              f"Share of all starts: bad {share['bad_start']:.1%}, good {share['good_start']:.1%}.", "",
              "| Previous starts | Group | n | Mean residual [95% CI] |", "|---|---|---|---|"]
    for title, grp in desc:
        for r in grp:
            lines.append(f"| {title} | {r['group']} | {r['n']} | {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
    lines += ["", "## Coverage and correlation with strikeouts", "", "| Feature | Coverage | Corr with K |", "|---|---|---|"]
    lines += [f"| {c} | {cover[c]:.3f} | {corr[c]:+.3f} |" for c in NEW]
    lines += ["", "## Leakage check (rebuilt with data cut off before each date)", "", *[f"- {x}" for x in leak], ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

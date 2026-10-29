"""Test chase-rate (O-Swing%) features one change at a time, with Hayden's rule.

Same method as ``src/compare_features.py``: baseline is the live blend with the
current features; each test adds one thing; fit on seasons before the test
season, score on 2025 and 2026; paired bootstrap over game dates on the MAE
difference. A change is kept only if it is better in both seasons with both
95% intervals below zero. Differences under 0.005 that don't hold in both
seasons are noise.

Also runs the leakage check for the new columns: each must be identical when
rebuilt from data cut off the day before a sample of game dates.

Usage::

    python src/test_chase.py
    python src/test_chase.py --boot 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import FEATURES, FEATURES_PATH, TARGET
from chase_features import CHASE_FEATURES, add_chase_features, build_chase_cache
from compare_features import NOISE, SEASONS, paired_bootstrap, predict, verdict
from fetch_data import load_lineups

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "chase_test.md"
TESTS = [
    ("+ pitcher chase rate, last 5 starts", ["p_chase_last5"]),
    ("+ pitcher chase rate, career (shrunk)", ["p_chase_career"]),
    ("+ pitcher two-strike chase rate", ["p_chase_2k_career"]),
    ("+ lineup chase rate", ["lineup_chase_rate"]),
    ("+ pitcher x lineup chase (log5)", ["matchup_chase_rate"]),
    ("+ all chase features", CHASE_FEATURES),
]


def leakage_check(df: pd.DataFrame, lineups: pd.DataFrame, chase: dict, n_dates: int = 3) -> list[str]:
    """Rebuild the features with chase data cut off before a few dates; values must match."""
    rng = np.random.default_rng(0)
    dates = sorted(rng.choice(df.loc[df["season"] >= 2025, "game_date"].unique(), n_dates, replace=False))
    out = []
    for d in dates:
        cut = {k: v[pd.to_datetime(v["game_date"]) < d] for k, v in chase.items()}
        lu = lineups[pd.to_datetime(lineups["game_date"]) <= d]
        rows = df[df["game_date"] == d]
        hist = df[df["game_date"] <= d]
        a = add_chase_features(hist, lu, cut).set_index(["game_pk", "pitcher_id"]).loc[rows.set_index(["game_pk", "pitcher_id"]).index]
        b = df.set_index(["game_pk", "pitcher_id"]).loc[a.index]
        for c in CHASE_FEATURES:
            same = np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)
            out.append(f"{pd.Timestamp(d).date()} {c}: {'ok' if same else 'MISMATCH'}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args(argv)

    print("Chase counts from raw Statcast...")
    chase = build_chase_cache()
    lineups = load_lineups()
    df = pd.read_parquet(FEATURES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "pitcher_id"]).reset_index(drop=True)
    df = add_chase_features(df, lineups, chase)
    cover = df[CHASE_FEATURES].notna().mean().round(3)
    print("Coverage:\n" + cover.to_string())
    corr = df[CHASE_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).round(3)
    print("Correlation with strikeouts:\n" + corr.to_string())

    print("Leakage check...")
    leak = leakage_check(df, lineups, chase)
    print("\n".join(leak))

    base = {"model": "blend"}
    rows, base_pred = [], {}
    for name, extra in TESTS:
        t0, per = time.time(), []
        for s in SEASONS:
            tr, te = df[df["season"] < s], df[df["season"] == s]
            y = te[TARGET].to_numpy(float)
            if s not in base_pred:
                base_pred[s] = predict(base, tr, te)
            b, n = base_pred[s], predict({"model": "blend", "cols": FEATURES + extra}, tr, te)
            diff, lo, hi = paired_bootstrap(y, b, n, te["game_date"], args.boot)
            per.append(dict(season=s, base=float(np.abs(y - b).mean()), new=float(np.abs(y - n).mean()),
                            diff=diff, lo=lo, hi=hi))
        v = verdict(per, "change")
        rows.append((name, per, v))
        print(f"  {name}: " + ", ".join(f"{r['season']} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                                        for r in per) + f" -> {v} ({time.time() - t0:.0f}s)")

    lines = ["# Chase rate (O-Swing%) feature tests", "",
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
    lines += ["", "## Coverage and correlation with strikeouts", "", "| Feature | Coverage | Corr with K |", "|---|---|---|"]
    lines += [f"| {c} | {cover[c]:.3f} | {corr[c]:+.3f} |" for c in CHASE_FEATURES]
    lines += ["", "## Leakage check (rebuilt with data cut off before each date)", "", *[f"- {x}" for x in leak], ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

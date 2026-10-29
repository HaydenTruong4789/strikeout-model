"""Test model changes one at a time, with a paired bootstrap on each difference.

Every test changes one thing against a baseline (the live blend with the
current features, unless it says otherwise), is fit on all seasons before a
test season, and is scored on that season (2025 and 2026). For each season the
MAE difference (change minus baseline, negative is better) gets a 95% interval
from a paired bootstrap that resamples game dates, since starts on the same day
share umpires, weather and lineups.

Rule: a change is kept only if it is better in both seasons with an interval
that excludes zero in both. A drop (removing features) is kept if it is not
significantly worse in either season, since a simpler model is preferred.
Differences under ``NOISE`` (0.005 MAE) that don't hold in both seasons are noise.

Tests:

1. Rest flags (short rest <= 4 days, layoff 10+, IL return 30+) in place of
   ``rest_days``, on the one-stage blend and on the two-stage model (batters
   faced x K per batter faced).
2. Short-rest starts (likely openers or bulk relief) and very short starts left
   out of the training rows (test rows are unchanged).
3. Park factor as home-vs-road for the same team, in place of the current one.
4. Temperature: a within-month check of its effect. Temperature and opponent
   last-10 form were dropped from the model after an earlier run of this script
   showed no loss without them; "+ temperature and opponent form" adds them back.
5. Blend vs linear regression alone.
6. Per-batter matchup: each hitter's K chance vs this pitcher, summed over his
   expected PAs.
7. Batters faced as its own model with workload features (previous pitch count,
   pitches per PA, team hook). Game totals and moneylines need odds data and
   are not included.
8. Probabilities: Poisson, negative binomial and empirical, scored on log loss
   and on calibration at 4.5, 5.5 and 6.5 K.
9. The earlier candidates (SwStr%, CSW%, pitches per PA, team hook, own whiffs
   by pitch, velo vs normal) and the suggested drops.

Also reports park factors by venue ID and the leakage audit on the new columns.

Usage::

    python src/compare_features.py
    python src/compare_features.py --skip-audit --boot 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import (
    CANDIDATE_FEATURE_GROUPS,
    CANDIDATE_FEATURES,
    FEATURES,
    FEATURES_PATH,
    TARGET,
    audit_leakage,
    format_audit,
    home_teams,
    to_date,
)
from distribution import Distribution, calibration_error, log_loss, prob_at_least
from fetch_data import load_raw
from train import make_blend, make_linear

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "feature_tests.md"
SEASONS = [2025, 2026]
NOISE = 0.005
N_BOOT = 2000
SEED = 42
BF = "batters_faced"
REST_FLAGS = ["short_rest", "long_layoff", "il_return"]
WORKLOAD = ["pitches_prev", "pitches_per_pa_last5", "team_starter_bf_last30"]
LOW_SIGNAL = ["wind_out_mph", "lineup_platoon_adv_share", "day_game", "is_lhp", "roof_closed"]
PROB_THRESHOLDS = [5, 6, 7]  # over 4.5, 5.5, 6.5 strikeouts
METHODS = ["poisson", "negbin", "empirical"]


def swap(cols: list[str], old: list[str], new: list[str]) -> list[str]:
    out = [c for c in cols if c not in old]
    return out + [c for c in new if c not in out]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def two_stage(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
              bf_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Linear batters faced x linear K per batter faced (weighted by BF). Returns (K, BF) predictions."""
    bf_train = train[BF].where(train[BF] > 0)
    ok = bf_train.notna()
    bf_cols = bf_cols or cols
    bf = np.clip(make_linear().fit(train[bf_cols], train[BF]).predict(test[bf_cols]), 0, None)
    rate = make_linear().fit(train.loc[ok, cols], (train[TARGET] / bf_train)[ok],
                             lr__sample_weight=bf_train[ok]).predict(test[cols])
    return np.clip(bf * np.clip(rate, 0, 1), 0, None), bf


def predict(spec: dict, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    cols, model = spec.get("cols", FEATURES), spec.get("model", "blend")
    if "train_filter" in spec:
        train = train[spec["train_filter"](train)]
    if model == "blend":
        return np.clip(make_blend().fit(train[cols], train[TARGET]).predict(test[cols]), 0, None)
    if model == "linear":
        return np.clip(make_linear().fit(train[cols], train[TARGET]).predict(test[cols]), 0, None)
    if model == "two-stage":
        return two_stage(train, test, cols, spec.get("bf_cols"))[0]
    if model == "bf":  # batters faced itself is the target
        return np.clip(make_linear().fit(train[cols], train[BF]).predict(test[cols]), 0, None)
    raise ValueError(model)


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #
def paired_bootstrap(y, base, new, dates, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """MAE(new) - MAE(base) and a 95% interval, resampling game dates."""
    d = pd.DataFrame({"date": np.asarray(dates), "diff": np.abs(y - new) - np.abs(y - base)})
    per = d.groupby("date")["diff"].agg(["sum", "count"])
    s, n = per["sum"].to_numpy(), per["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(s.sum() / n.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def verdict(rows: list[dict], kind: str) -> str:
    better = all(r["hi"] < 0 for r in rows)
    worse = any(r["lo"] > 0 for r in rows)
    if kind == "drop":
        return "drop (no significant loss)" if not worse else "keep the features"
    if better:
        return "keep"
    if worse:
        return "worse"
    small = all(abs(r["diff"]) < NOISE for r in rows)
    return "noise" if small else "not consistent"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def tests() -> list[dict]:
    """Each test: name, baseline spec, changed spec, kind ('change' or 'drop'), target."""
    blend, linear, ts = {"model": "blend"}, {"model": "linear"}, {"model": "two-stage"}
    flags = swap(FEATURES, ["rest_days"], REST_FLAGS)
    out = [
        dict(name="1a. rest flags replace rest_days (blend)", base=blend, new={"cols": flags}),
        dict(name="1b. rest flags replace rest_days (two-stage)", base=ts, new={**ts, "cols": flags}),
        dict(name="1c. rest flags, batters-faced model", base={"model": "bf"}, new={"model": "bf", "cols": flags},
             target=BF),
        dict(name="2a. leave short-rest starts out of training", base=blend,
             new={"train_filter": lambda d: d["short_rest"] != 1}),
        dict(name="2b. leave starts under 9 BF out of training", base=blend,
             new={"train_filter": lambda d: d[BF] >= 9}),
        dict(name="3. home-vs-road park factor replaces current", base=blend,
             new={"cols": swap(FEATURES, ["park_k_factor"], ["park_k_factor_hr"])}),
        dict(name="5. blend vs linear alone", base=linear, new=blend),
        dict(name="6. + per-batter matchup expected K", base=blend,
             new={"cols": FEATURES + ["matchup_expected_k", "lineup_log5_k_rate"]}),
        dict(name="7a. batters-faced model + workload", base={"model": "bf"},
             new={"model": "bf", "cols": FEATURES + WORKLOAD}, target=BF),
        dict(name="7b. two-stage with workload BF model", base=ts,
             new={**ts, "bf_cols": FEATURES + WORKLOAD}),
        dict(name="7c. two-stage (workload BF) vs one-stage blend", base=blend,
             new={**ts, "bf_cols": FEATURES + WORKLOAD}),
    ]
    for name, cols in CANDIDATE_FEATURE_GROUPS.items():
        if name in ("rest flags", "home-road park factor", "per-batter matchup"):
            continue
        extra = [c for c in cols if c not in FEATURES]
        out.append(dict(name=f"9. + {name}", base=blend, new={"cols": FEATURES + extra}))
    out += [
        dict(name="9. drop the 5 low-signal features", base=blend, new={"cols": swap(FEATURES, LOW_SIGNAL, [])},
             kind="drop"),
    ]
    return out


def run_tests(df: pd.DataFrame, seasons: list[int], n_boot: int) -> pd.DataFrame:
    cache: dict = {}

    def preds(spec, s, tr, te):
        key = (id(spec), s)  # specs live for the whole run; shared baselines fit once
        if key not in cache:
            cache[key] = predict(spec, tr, te)
        return cache[key]

    rows = []
    for t in tests():
        t0, per = time.time(), []
        for s in seasons:
            tr, te = df[df["season"] < s], df[df["season"] == s]
            y = te[t.get("target", TARGET)].to_numpy(float)
            b, n = preds(t["base"], s, tr, te), preds(t["new"], s, tr, te)
            diff, lo, hi = paired_bootstrap(y, b, n, te["game_date"], n_boot)
            per.append(dict(season=s, base=float(np.abs(y - b).mean()), new=float(np.abs(y - n).mean()),
                            diff=diff, lo=lo, hi=hi))
        row = {"test": t["name"], "target": "BF" if t.get("target") == BF else "K",
               "verdict": verdict(per, t.get("kind", "change"))}
        for r in per:
            s = r["season"]
            row.update({f"base {s}": r["base"], f"new {s}": r["new"], f"diff {s}": r["diff"],
                        f"CI {s}": f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"})
        rows.append(row)
        print(f"  {t['name']}: " + ", ".join(f"{r['season']} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                                            for r in per) + f" -> {row['verdict']} ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def test_probabilities(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Distributions fit on season S-1 out-of-sample predictions, scored on season S."""
    rows = {}
    for s in seasons:
        early, prev = df[df["season"] < s - 1], df[df["season"] == s - 1]
        tr, te = df[df["season"] < s], df[df["season"] == s]
        mu_prev = predict({}, early, prev)
        mu = predict({}, tr, te)
        y = te[TARGET].to_numpy(int)
        for m in METHODS:
            dist = Distribution(m).fit(prev[TARGET], mu_prev)
            pmf = dist.pmf(mu)
            row = {"log loss": log_loss(y, pmf)}
            for t in PROB_THRESHOLDS:
                p = prob_at_least(pmf, t)
                row[f"Brier {t - 0.5}"] = float(((p - (y >= t)) ** 2).mean())
                row[f"calib err {t - 0.5}"] = calibration_error(y, pmf, t)
            row["alpha"] = dist.alpha if m == "negbin" else np.nan
            rows[f"{m} {s}"] = row
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Checks behind the park and temperature points
# --------------------------------------------------------------------------- #
def park_checks(raw: dict, df: pd.DataFrame) -> list[str]:
    games = raw["games"].copy()
    names = games.dropna(subset=["venue_id"]).groupby("venue_id")["venue_name"].unique()
    renamed = names[names.map(len) > 1]
    lines = ["## Parks by venue ID", "",
             "The model's park factor has always been keyed by venue ID; only the patterns report grouped by name.",
             "", "Venue IDs with more than one name: "
             + ("; ".join(f"{int(v)}: {', '.join(map(str, n))}" for v, n in renamed.items()) or "none"), ""]
    home = home_teams(raw["game_logs"])
    games["home_id"] = games["game_pk"].map(home)
    rays = games[(games["home_id"] == 139) & (games["season"].isin([2024, 2025, 2026]))]
    t = rays.groupby(["season", "venue_name"]).size().rename("games").reset_index()
    lines += ["Rays home games by season and venue:", "", "| Season | Venue | Games |", "|---|---|---|"]
    lines += [f"| {r.season} | {r.venue_name} | {r.games} |" for r in t.itertuples()]
    latest = df[df["season"] == df["season"].max()].merge(games[["game_pk", "venue_name", "home_id"]], on="game_pk")
    latest = latest.drop_duplicates("game_pk")
    by = latest.groupby("venue_name")[["park_k_factor", "park_k_factor_hr"]].mean()
    by = by[latest.groupby("venue_name").size() >= 20].sort_values("park_k_factor_hr")
    show = pd.concat([by.tail(5).iloc[::-1], by.head(5)])
    lines += ["", f"Highest and lowest home-vs-road park factors, {int(df['season'].max())} (mean over games):", "",
              "| Venue | Current factor | Home vs road |", "|---|---|---|"]
    lines += [f"| {v} | {r.park_k_factor:.3f} | {r.park_k_factor_hr:.3f} |" for v, r in show.iterrows()]
    corr = latest[["park_k_factor", "park_k_factor_hr"]].corr().iloc[0, 1]
    lines += ["", f"Correlation between the two factors across games: {corr:.2f}.", ""]
    return lines


def temperature_checks(raw: dict, seasons: list[int], n_boot: int) -> list[str]:
    """K/PA change per 10 F, raw and within season-month, open-air games only."""
    g = raw["games"].copy()
    g["temp_f"] = pd.to_numeric(g["temp_f"], errors="coerce")
    cond = g["condition"].fillna("").astype(str).str.lower()
    g = g[~cond.isin(["dome", "roof closed"]) & g["temp_f"].notna() & ~g["season"].isin(seasons)]
    tg = raw["team_games"].groupby("game_pk")[["bat_k", "bat_pa"]].sum()
    g = g.merge(tg, left_on="game_pk", right_index=True)
    g["month"] = to_date(g["game_date"]).dt.month.clip(4, 9)
    g["rate"] = g["bat_k"] / g["bat_pa"]

    def slope(d: pd.DataFrame, within: bool) -> float:
        x, y, w = d["temp_f"], d["rate"], d["bat_pa"]
        if within:
            grp = [d["season"], d["month"]]
            x = x - (x * w).groupby(grp).transform("sum") / w.groupby(grp).transform("sum")
            y = y - (y * w).groupby(grp).transform("sum") / w.groupby(grp).transform("sum")
        else:
            x = x - np.average(x, weights=w)
            y = y - np.average(y, weights=w)
        return float(10 * 100 * (w * x * y).sum() / (w * x * x).sum())  # K% points per 10 F

    rng = np.random.default_rng(SEED)
    out = {}
    for within in (False, True):
        est = slope(g, within)
        boot = [slope(g.iloc[rng.integers(0, len(g), len(g))], within) for _ in range(min(n_boot, 500))]
        out[within] = (est, np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    temp = pd.cut(g["temp_f"], [-100, 65, 75, 85, 200], labels=["<65", "65-74", "75-84", "85+"])
    table = (g.assign(temp=temp).groupby(["month", "temp"], observed=True)
             .apply(lambda d: pd.Series({"k": d["bat_k"].sum() / d["bat_pa"].sum(), "pa": d["bat_pa"].sum()}))
             .reset_index())
    piv = table.pivot(index="month", columns="temp", values="k")
    pa = table.pivot(index="month", columns="temp", values="pa")
    lines = ["## Temperature within month (open-air games, training seasons)", "",
             f"K/PA change per 10 F, all games: {out[False][0]:+.2f} points "
             f"[{out[False][1]:+.2f}, {out[False][2]:+.2f}]; within the same season and month: "
             f"{out[True][0]:+.2f} points [{out[True][1]:+.2f}, {out[True][2]:+.2f}].", "",
             "| Month | " + " | ".join(map(str, piv.columns)) + " |", "|---" * (len(piv.columns) + 1) + "|"]
    for m, r in piv.iterrows():
        cells = [f"{100 * v:.1f}% ({int(pa.loc[m, c]):,})" if pd.notna(v) and pa.loc[m, c] >= 2000 else ""
                 for c, v in r.items()]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "Cells show K/PA (PA); cells under 2,000 PA are blank. Months 3 and 10 are folded into 4 and 9.", ""]
    return lines


# --------------------------------------------------------------------------- #
def table_md(res: pd.DataFrame, seasons: list[int]) -> list[str]:
    head = ["Test", "Target", *(f"{s} base -> new" for s in seasons), *(f"{s} diff [95% CI]" for s in seasons),
            "Verdict"]
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for _, r in res.iterrows():
        cells = [r["test"], r["target"]]
        cells += [f"{r[f'base {s}']:.4f} -> {r[f'new {s}']:.4f}" for s in seasons]
        cells += [f"{r[f'diff {s}']:+.4f} {r[f'CI {s}']}" for s in seasons]
        lines.append("| " + " | ".join(cells + [r["verdict"]]) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    parser.add_argument("--boot", type=int, default=N_BOOT)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args(argv)

    df = pd.read_parquet(FEATURES_PATH)
    missing = [c for c in CANDIDATE_FEATURES if c not in df]
    if missing:
        print(f"Feature table lacks {missing}; run `python src/build_features.py` first.")
        return 1
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "pitcher_id"]).reset_index(drop=True)
    raw = load_raw()
    print("Candidate coverage (share of starts with a value):")
    print(df[CANDIDATE_FEATURES].notna().mean().round(3).to_string())

    lines = ["# Model tests: one change at a time", "",
             "Each test is fit on all seasons before the test season and scored on it. Diff is the change's "
             "MAE minus the baseline's (negative is better), with a 95% paired-bootstrap interval over game "
             "dates. Kept only if better in both seasons with both intervals below zero; a drop is kept if "
             f"not significantly worse in either. Differences under {NOISE} that don't hold in both seasons "
             "are noise.", ""]
    if not args.skip_audit:
        report = audit_leakage(raw, CANDIDATE_FEATURES, n_dates=3)
        print(format_audit(report, "Leakage audit, candidate features"))
        lines += ["```", format_audit(report, "Leakage audit, candidate features"), "```", ""]

    print("\nTests...")
    res = run_tests(df, args.seasons, args.boot)
    lines += ["## Results", "", *table_md(res, args.seasons), ""]

    keep = [t for t, v in zip(res["test"], res["verdict"], strict=True) if v == "keep" or v.startswith("drop (")]
    lines += [f"Passed the rule: {', '.join(keep) or 'none'}.", ""]

    print("\nProbabilities...")
    probs = test_probabilities(df, args.seasons)
    print(probs.round(4).to_string())
    cols = list(probs.columns)
    lines += ["## Probabilities (blend means, distribution fit on the previous season)", "",
              "| Method, season | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {n} | " + " | ".join("" if pd.isna(v) else f"{v:.4f}" for v in r) + " |"
              for n, r in probs.iterrows()]
    lines.append("")

    lines += park_checks(raw, df)
    lines += temperature_checks(raw, args.seasons, args.boot)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-40:]))
    print(f"\nWrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

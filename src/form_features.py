"""Recent-form candidate features: how a pitcher's last few starts went.

Game logs have batters faced, outs and strikeouts but no earned runs, so runs
allowed come from raw Statcast: the batting team's score change on every pitch
he threw (``post_bat_score - bat_score``). That counts unearned runs and misses
inherited runners who score after he leaves, so it is "runs while he pitched".

A start is **bad** when any of:

* K/BF more than ``KRATE_GAP`` (0.08) below his recency-weighted K rate going in,
* 4+ runs while he pitched,
* under 4 innings (12 outs).

A start is **good** when K/BF is more than 0.08 above that rate, or he had 7+
strikeouts with 0-1 runs.

Candidate features (from earlier starts only):

* ``last_start_k_surprise``: last start's K minus his ``k_per_start_ewm`` going into it.
* ``bad_starts_last5`` / ``good_starts_last5``: counts over his previous five starts.
* ``last_start_short``: the last start had 15 or fewer batters faced.

``last_start_bad`` / ``last_start_good`` and ``bad_streak3`` / ``good_streak3``
(last three all bad / all good) are kept for the descriptive table only.

These live outside ``build_features.py`` until ``src/test_form_matchup.py`` shows they help.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from build_features import to_date
from chase_features import read_raw
from fetch_data import STATCAST_AGG_DIR, STATCAST_RAW_DIR

log = logging.getLogger("form_features")
RUNS_DIR = STATCAST_AGG_DIR / "runs"
KRATE_GAP = 0.08
FORM_FEATURES = ["last_start_k_surprise", "bad_starts_last5", "good_starts_last5", "last_start_short"]
STREAK_COLS = ["last_start_bad", "last_start_good", "bad_streak3", "good_streak3"]


def build_runs_cache(force: bool = False) -> pd.DataFrame:
    """Runs scored while each pitcher was on the mound, per pitcher-game (cached per raw chunk)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    raws = sorted(STATCAST_RAW_DIR.glob("*.parquet"))
    for path in raws:
        out = RUNS_DIR / path.name
        if out.exists() and not force:
            continue
        df = read_raw(path, ["game_pk", "game_date", "pitcher", "bat_score", "post_bat_score"])
        df["runs"] = (pd.to_numeric(df["post_bat_score"], errors="coerce")
                      - pd.to_numeric(df["bat_score"], errors="coerce")).clip(lower=0)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
        agg = df.groupby(["game_pk", "pitcher"], as_index=False).agg(game_date=("game_date", "first"),
                                                                     runs=("runs", "sum"))
        agg.rename(columns={"pitcher": "pitcher_id"}).to_parquet(out, index=False)
        log.info("runs for %s", path.name)
    runs = pd.concat([pd.read_parquet(f) for f in sorted(RUNS_DIR.glob("*.parquet"))], ignore_index=True)
    runs = runs.groupby(["game_pk", "pitcher_id"], as_index=False).agg(game_date=("game_date", "first"),
                                                                       runs=("runs", "sum"))
    runs["pitcher_id"] = runs["pitcher_id"].astype("int64")
    return runs


def add_form_features(df: pd.DataFrame, game_logs: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` (one row per start, with ``k_rate_ewm`` and ``k_per_start_ewm``) plus the form columns.

    Each start's own outcome is labeled first; every feature then shifts by one
    start, so a row only sees starts before it.
    """
    df = df.drop(columns=FORM_FEATURES + STREAK_COLS + ["bad_start", "good_start"], errors="ignore")
    df["game_date"] = to_date(df["game_date"])
    logs = game_logs[["game_pk", "pitcher_id", "batters_faced", "outs"]].copy()
    logs["pitcher_id"] = logs["pitcher_id"].astype("int64")
    logs = logs.drop_duplicates(["game_pk", "pitcher_id"])
    s = (df[["game_pk", "pitcher_id", "game_date", "strikeouts", "k_rate_ewm", "k_per_start_ewm"]]
         .assign(_i=np.arange(len(df)))
         .merge(logs, on=["game_pk", "pitcher_id"], how="left")
         .merge(runs[["game_pk", "pitcher_id", "runs"]], on=["game_pk", "pitcher_id"], how="left"))
    for c in ("strikeouts", "batters_faced", "outs", "runs"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.sort_values(["pitcher_id", "game_date", "game_pk"])

    kbf = s["strikeouts"] / s["batters_faced"].where(s["batters_faced"] > 0)
    played = s["strikeouts"].notna()
    bad = (kbf < s["k_rate_ewm"] - KRATE_GAP) | (s["runs"] >= 4) | (s["outs"] < 12)
    good = (kbf > s["k_rate_ewm"] + KRATE_GAP) | ((s["strikeouts"] >= 7) & (s["runs"] <= 1))
    s["bad_start"] = bad.astype(float).where(played)
    s["good_start"] = good.astype(float).where(played)
    s["k_surprise"] = s["strikeouts"] - s["k_per_start_ewm"]
    s["short"] = (s["batters_faced"] <= 15).astype(float).where(s["batters_faced"].notna())

    g = s.groupby("pitcher_id")
    prev = lambda c: g[c].shift(1)
    roll = lambda c, w: g[c].transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum())
    full = lambda c, w: g[c].transform(lambda x: x.shift(1).rolling(w, min_periods=w).sum())
    s["last_start_k_surprise"] = prev("k_surprise")
    s["last_start_short"] = prev("short")
    s["last_start_bad"] = prev("bad_start")
    s["last_start_good"] = prev("good_start")
    s["bad_starts_last5"] = roll("bad_start", 5)
    s["good_starts_last5"] = roll("good_start", 5)
    s["bad_streak3"] = (full("bad_start", 3) == 3).astype(float).where(full("bad_start", 3).notna())
    s["good_streak3"] = (full("good_start", 3) == 3).astype(float).where(full("good_start", 3).notna())
    # A pitcher's first start has no history: counts are NaN, not 0.
    first = g.cumcount() == 0
    s.loc[first, ["bad_starts_last5", "good_starts_last5"]] = np.nan

    s = s.sort_values("_i")
    for c in FORM_FEATURES + STREAK_COLS + ["bad_start", "good_start"]:
        df[c] = s[c].to_numpy()
    return df

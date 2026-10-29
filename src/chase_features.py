"""Chase (O-Swing%) summaries and candidate features, built from the raw Statcast cache.

A pitch is out of the zone when Statcast's ``zone`` is 11-14. O-Swing% is
swings at out-of-zone pitches / out-of-zone pitches.

Per pitcher-game and batter-game counts are cached in
``data/raw/statcast_agg/chase/{pitcher,batter}/`` (one file per raw chunk), so
re-running only summarizes new chunks.

Candidate features (all from games before the one being predicted):

* ``p_chase_last5``: O-Swing% the pitcher induced over his last five starts.
* ``p_chase_career``: the same over all his earlier games, shrunk to the league.
* ``p_chase_2k_career``: induced O-Swing% with two strikes, shrunk.
* ``lineup_chase_rate``: the opponent lineup's O-Swing% (each batter's earlier
  games, shrunk to the league, weighted by expected PAs per batting slot).
* ``matchup_chase_rate``: pitcher x lineup chase in odds-ratio (log5) form.

These live outside ``build_features.py`` until a test in ``src/test_chase.py``
shows they help.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import LINEUP_PA_WEIGHTS, opponent_lineups, prior_rolling, prior_totals, shrunk_rate, to_date
from fetch_data import STATCAST_AGG_DIR, STATCAST_RAW_DIR, SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS

log = logging.getLogger("chase_features")
CHASE_DIR = STATCAST_AGG_DIR / "chase"
COUNT_COLS = ["n_oz", "sw_oz", "wh_oz", "n_iz", "sw_iz", "n_oz_2k", "sw_oz_2k"]
SHRINK = {"pitcher": 600, "pitcher_2k": 250, "batter": 150}   # out-of-zone pitches of prior weight
RAW_COLS = ["game_pk", "game_date", "pitcher", "batter", "zone", "description", "strikes"]
CHASE_FEATURES = ["p_chase_last5", "p_chase_career", "p_chase_2k_career", "lineup_chase_rate", "matchup_chase_rate"]


def read_raw(path: Path, cols: list[str]) -> pd.DataFrame:
    """Read only ``cols`` from a raw Statcast chunk (missing columns come back as NaN)."""
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception:  # noqa: BLE001 - a column missing from an old chunk
        return pd.read_parquet(path).reindex(columns=cols)


def summarize_chase(pitches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-zone / in-zone pitch and swing counts per pitcher-game and batter-game."""
    df = pitches.dropna(subset=["zone"]).copy()
    zone = pd.to_numeric(df["zone"], errors="coerce")
    oz, iz = zone.between(11, 14), zone.between(1, 9)
    swing = df["description"].isin(SWING_DESCRIPTIONS)
    whiff = df["description"].isin(WHIFF_DESCRIPTIONS)
    two = pd.to_numeric(df["strikes"], errors="coerce") == 2
    df = df.assign(n_oz=oz, sw_oz=oz & swing, wh_oz=oz & whiff, n_iz=iz, sw_iz=iz & swing,
                   n_oz_2k=oz & two, sw_oz_2k=oz & two & swing)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    agg = {c: (c, "sum") for c in COUNT_COLS}
    pit = df.groupby(["game_pk", "pitcher"], as_index=False).agg(game_date=("game_date", "first"), **agg)
    bat = df.groupby(["game_pk", "batter"], as_index=False).agg(game_date=("game_date", "first"), **agg)
    return pit.rename(columns={"pitcher": "pitcher_id"}), bat.rename(columns={"batter": "batter_id"})


def build_chase_cache(force: bool = False) -> dict[str, pd.DataFrame]:
    """Summarize every raw chunk not yet cached; return all pitcher and batter chase counts."""
    out_p, out_b = CHASE_DIR / "pitcher", CHASE_DIR / "batter"
    out_p.mkdir(parents=True, exist_ok=True); out_b.mkdir(parents=True, exist_ok=True)
    raws = sorted(STATCAST_RAW_DIR.glob("*.parquet"))
    if not raws:
        raise RuntimeError(f"No raw Statcast chunks in {STATCAST_RAW_DIR}.")
    for path in raws:
        if not force and (out_p / path.name).exists() and (out_b / path.name).exists():
            continue
        pit, bat = summarize_chase(read_raw(path, RAW_COLS))
        bat.to_parquet(out_b / path.name, index=False)
        pit.to_parquet(out_p / path.name, index=False)
        log.info("chase counts for %s", path.name)
    load = lambda d: pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)
    return {"pitcher": load(out_p), "batter": load(out_b)}


def odds(x):
    x = np.clip(np.asarray(x, float), 1e-4, 1 - 1e-4)
    return x / (1 - x)


def add_chase_features(df: pd.DataFrame, lineups: pd.DataFrame, chase: dict[str, pd.DataFrame],
                       lineup_source: str = "posted") -> pd.DataFrame:
    """Return ``df`` (one row per start) with the chase candidate columns added."""
    df = df.drop(columns=CHASE_FEATURES, errors="ignore")  # rebuilt below; stale copies break the merge
    df["game_date"] = to_date(df["game_date"])
    if "p_hand" not in df:
        df["p_hand"] = np.where(df["is_lhp"] == 1, "L", np.where(df["is_lhp"] == 0, "R", None))
    pit = chase["pitcher"].copy()
    pit["game_date"] = to_date(pit["game_date"])
    pit = pit.groupby(["game_pk", "pitcher_id", "game_date"], as_index=False)[COUNT_COLS].sum()
    pit["pitcher_id"] = pit["pitcher_id"].astype("int64")

    # Pitcher: last five starts (only his starts) and all earlier games, shrunk to the league.
    order = df[["game_pk", "pitcher_id", "game_date"]].assign(_i=np.arange(len(df)))
    starts = order.merge(pit.drop(columns="game_date"), on=["game_pk", "pitcher_id"], how="left")
    starts = starts.sort_values(["pitcher_id", "game_date", "game_pk"])
    n5 = prior_rolling(starts, "n_oz", 5)
    s5 = prior_rolling(starts, "sw_oz", 5)
    starts["p_chase_last5"] = s5 / n5.where(n5 >= 40)
    starts = starts.sort_values("_i")
    df["p_chase_last5"] = starts["p_chase_last5"].to_numpy()

    hist = ["n_oz", "sw_oz", "n_oz_2k", "sw_oz_2k"]
    left = df[["pitcher_id", "game_date"]].copy()
    left = prior_totals(left, pit, ["pitcher_id"], hist, prefix="p_")
    left = prior_totals(left, pit, [], hist, prefix="lg_")
    lg = left["lg_sw_oz"] / left["lg_n_oz"].where(left["lg_n_oz"] > 0)
    lg2 = left["lg_sw_oz_2k"] / left["lg_n_oz_2k"].where(left["lg_n_oz_2k"] > 0)
    df["p_chase_career"] = shrunk_rate(left["p_sw_oz"], left["p_n_oz"], lg, SHRINK["pitcher"]).to_numpy()
    df["p_chase_2k_career"] = shrunk_rate(left["p_sw_oz_2k"], left["p_n_oz_2k"], lg2, SHRINK["pitcher_2k"]).to_numpy()
    df["lg_chase"] = lg.to_numpy()

    # Lineup: each batter's O-Swing% from earlier games, shrunk, weighted by batting slot.
    bat = chase["batter"].copy()
    bat["game_date"] = to_date(bat["game_date"])
    bat["batter_id"] = bat["batter_id"].astype("int64")
    lu = opponent_lineups(df, lineups, lineup_source)
    lu["batter_id"] = lu["batter_id"].astype("int64")
    lu = prior_totals(lu, bat, ["batter_id"], ["n_oz", "sw_oz"], prefix="b_")
    lu = prior_totals(lu, bat, [], ["n_oz", "sw_oz"], prefix="lg_")
    lg_b = lu["lg_sw_oz"] / lu["lg_n_oz"].where(lu["lg_n_oz"] > 0)
    lu["chase"] = shrunk_rate(lu["b_sw_oz"], lu["b_n_oz"], lg_b, SHRINK["batter"])
    lu["w"] = LINEUP_PA_WEIGHTS[(lu["batting_order"].clip(1, 9) - 1).astype(int)]
    g = [lu["game_pk"], lu["pitcher_id"]]
    w = lu["w"].where(lu["chase"].notna())
    lineup = ((lu["chase"] * w).groupby(g).sum(min_count=1) / w.groupby(g).sum(min_count=1)).rename("lineup_chase_rate")
    df = df.merge(lineup.reset_index(), on=["game_pk", "pitcher_id"], how="left")

    o = odds(df["p_chase_career"]) * odds(df["lineup_chase_rate"]) / odds(df["lg_chase"])
    df["matchup_chase_rate"] = np.where(np.isnan(o), np.nan, o / (1 + o))
    return df.drop(columns=["lg_chase"])

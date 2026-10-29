"""Finer pitch-type x velocity matchup features, built from the raw Statcast cache.

The live model's matchup uses five pitch classes. Here there are nine:

* ``ff95``: four-seam or sinker at 95+ mph, ``ff92``: 92-95, ``fflo``: under 92
* ``fc`` cutter, ``sl`` slider, ``st`` sweeper, ``cu`` curve / slurve / knuckle-curve
* ``ch`` changeup, ``fs`` splitter (and forkball)

Per batter (earlier games only), each class gets a whiff rate (whiffs per
swing) and a chase rate (swings at out-of-zone pitches, ``zone`` 11-14), both
shrunk to that class's league rate. The opposing lineup's rate is the
batting-slot PA-weighted mean (``LINEUP_PA_WEIGHTS``). The pitcher's own rate
per class (all earlier games, shrunk) is combined with the lineup's in
odds-ratio (log5) form against the league rate, and weighted by his usage of
each class over his last five starts:

* ``matchup_whiff_fine`` = sum(usage_c x log5(pitcher_whiff_c, lineup_whiff_c, league_c)) / sum(usage_c)
* ``matchup_chase_fine``: the same with chase rates.

Counts are cached per raw chunk in ``data/raw/statcast_agg/fine/{pitcher,batter}/``.
These live outside ``build_features.py`` until ``src/test_form_matchup.py`` shows they help.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from build_features import LINEUP_PA_WEIGHTS, opponent_lineups, prior_rolling, prior_totals, shrunk_rate, to_date
from chase_features import odds, read_raw
from fetch_data import STATCAST_AGG_DIR, STATCAST_RAW_DIR, SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS

log = logging.getLogger("pitch_matchup_features")
FINE_DIR = STATCAST_AGG_DIR / "fine"
FINE_CLASSES = ["ff95", "ff92", "fflo", "fc", "sl", "st", "cu", "ch", "fs"]
KINDS = ["n", "sw", "wh", "oz", "osw"]  # pitches, swings, whiffs, out-of-zone pitches, out-of-zone swings
COUNT_COLS = [f"{k}_{c}" for c in FINE_CLASSES for k in KINDS]
SHRINK = {"batter_whiff": 30, "batter_chase": 60, "pitcher_whiff": 80, "pitcher_chase": 150}
RAW_COLS = ["game_pk", "game_date", "pitcher", "batter", "pitch_type", "release_speed", "zone", "description"]
PITCH_MATCHUP_FEATURES = ["matchup_whiff_fine", "matchup_chase_fine"]


def fine_class(pitch_type: pd.Series, speed: pd.Series) -> pd.Series:
    """Label each pitch with one of ``FINE_CLASSES`` (NaN for anything else)."""
    out = pd.Series(np.nan, index=pitch_type.index, dtype="object")
    fb = pitch_type.isin({"FF", "SI", "FA"})
    out[fb & (speed >= 95)] = "ff95"
    out[fb & (speed >= 92) & (speed < 95)] = "ff92"
    out[fb & (speed < 92)] = "fflo"
    for cls, types in {"fc": {"FC"}, "sl": {"SL"}, "st": {"ST"}, "cu": {"CU", "KC", "SV", "CS"},
                       "ch": {"CH"}, "fs": {"FS", "FO"}}.items():
        out[pitch_type.isin(types)] = cls
    return out


def summarize(pitches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-class pitch, swing, whiff and chase counts per pitcher-game and batter-game."""
    df = pitches.copy()
    df["cls"] = fine_class(df["pitch_type"].astype(object),
                           pd.to_numeric(df["release_speed"], errors="coerce").astype(float))
    df = df.dropna(subset=["cls"])
    oz = pd.to_numeric(df["zone"], errors="coerce").astype(float).between(11, 14)
    sw = df["description"].isin(SWING_DESCRIPTIONS).astype(bool)
    df = df.assign(n=1, sw=sw.astype(int), wh=df["description"].isin(WHIFF_DESCRIPTIONS).astype(int),
                   oz=oz.astype(int), osw=(oz & sw).astype(int))
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

    def by(who: str) -> pd.DataFrame:
        t = df.pivot_table(index=["game_pk", who, "game_date"], columns="cls", values=KINDS, aggfunc="sum", fill_value=0)
        t.columns = [f"{k}_{c}" for k, c in t.columns]
        return t.reindex(columns=COUNT_COLS, fill_value=0).reset_index()

    return by("pitcher").rename(columns={"pitcher": "pitcher_id"}), by("batter").rename(columns={"batter": "batter_id"})


def build_fine_cache(force: bool = False) -> dict[str, pd.DataFrame]:
    """Summarize every raw chunk not yet cached; return all pitcher and batter class counts."""
    out_p, out_b = FINE_DIR / "pitcher", FINE_DIR / "batter"
    out_p.mkdir(parents=True, exist_ok=True); out_b.mkdir(parents=True, exist_ok=True)
    raws = sorted(STATCAST_RAW_DIR.glob("*.parquet"))
    if not raws:
        raise RuntimeError(f"No raw Statcast chunks in {STATCAST_RAW_DIR}.")
    for path in raws:
        if not force and (out_p / path.name).exists() and (out_b / path.name).exists():
            continue
        pit, bat = summarize(read_raw(path, RAW_COLS))
        bat.to_parquet(out_b / path.name, index=False)
        pit.to_parquet(out_p / path.name, index=False)
        log.info("fine pitch counts for %s", path.name)

    def load(d, key):
        x = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)
        x = x.groupby(["game_pk", key], as_index=False).agg(game_date=("game_date", "first"),
                                                            **{c: (c, "sum") for c in COUNT_COLS})
        x[key] = x[key].astype("int64")
        x["game_date"] = to_date(x["game_date"])
        return x
    return {"pitcher": load(out_p, "pitcher_id"), "batter": load(out_b, "batter_id")}


def _rates(t: pd.DataFrame, prefix: str, lg: pd.DataFrame, m_whiff: float, m_chase: float) -> dict[str, pd.Series]:
    """Shrunk whiff-per-swing and chase rates per class from ``prefix`` totals."""
    out = {}
    for c in FINE_CLASSES:
        out[f"whiff_{c}"] = shrunk_rate(t[f"{prefix}wh_{c}"], t[f"{prefix}sw_{c}"], lg[f"whiff_{c}"], m_whiff)
        out[f"chase_{c}"] = shrunk_rate(t[f"{prefix}osw_{c}"], t[f"{prefix}oz_{c}"], lg[f"chase_{c}"], m_chase)
    return out


def _league(t: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in FINE_CLASSES:
        out[f"whiff_{c}"] = t[f"lg_wh_{c}"] / t[f"lg_sw_{c}"].where(t[f"lg_sw_{c}"] > 0)
        out[f"chase_{c}"] = t[f"lg_osw_{c}"] / t[f"lg_oz_{c}"].where(t[f"lg_oz_{c}"] > 0)
    return pd.DataFrame(out, index=t.index)


def add_pitch_matchup_features(df: pd.DataFrame, lineups: pd.DataFrame, fine: dict[str, pd.DataFrame],
                               lineup_source: str = "posted") -> pd.DataFrame:
    """Return ``df`` (one row per start) with ``matchup_whiff_fine`` and ``matchup_chase_fine`` added."""
    df = df.drop(columns=PITCH_MATCHUP_FEATURES, errors="ignore").reset_index(drop=True)
    df["game_date"] = to_date(df["game_date"])
    if "p_hand" not in df:
        df["p_hand"] = np.where(df["is_lhp"] == 1, "L", np.where(df["is_lhp"] == 0, "R", None))
    pit, bat = fine["pitcher"], fine["batter"]

    # Usage per class over his last five starts.
    order = df[["game_pk", "pitcher_id", "game_date"]].assign(_i=np.arange(len(df)))
    n_cols = [f"n_{c}" for c in FINE_CLASSES]
    st = order.merge(pit[["game_pk", "pitcher_id", *n_cols]], on=["game_pk", "pitcher_id"], how="left")
    st = st.sort_values(["pitcher_id", "game_date", "game_pk"])
    for c in n_cols:
        st[c + "_l5"] = prior_rolling(st, c, 5)
    st = st.sort_values("_i")
    usage = st[[c + "_l5" for c in n_cols]].to_numpy(float)
    tot = usage.sum(axis=1, keepdims=True)
    usage = np.where(tot >= 100, usage / np.where(tot > 0, tot, 1), np.nan)

    # Pitcher's own per-class rates over all earlier games, shrunk to the league.
    rate_cols = [c for c in COUNT_COLS if not c.startswith("n_")]
    left = df[["pitcher_id", "game_date"]].copy()
    left = prior_totals(left, pit, ["pitcher_id"], rate_cols, prefix="p_")
    left = prior_totals(left, pit, [], rate_cols, prefix="lg_")
    lg = _league(left)
    p_rates = _rates(left, "p_", lg, SHRINK["pitcher_whiff"], SHRINK["pitcher_chase"])

    # Lineup: each batter's per-class rates from earlier games, PA-weighted by slot.
    lu = opponent_lineups(df, lineups, lineup_source)
    lu["batter_id"] = lu["batter_id"].astype("int64")
    lu = prior_totals(lu, bat, ["batter_id"], rate_cols, prefix="b_")
    lu = prior_totals(lu, bat, [], rate_cols, prefix="lg_")
    b_rates = pd.DataFrame(_rates(lu, "b_", _league(lu), SHRINK["batter_whiff"], SHRINK["batter_chase"]), index=lu.index)
    w = pd.Series(LINEUP_PA_WEIGHTS[(lu["batting_order"].clip(1, 9) - 1).astype(int)], index=lu.index)
    g = [lu["game_pk"], lu["pitcher_id"]]
    lineup = {}
    for col in b_rates:
        ww = w.where(b_rates[col].notna())
        lineup[col] = (b_rates[col] * ww).groupby(g).sum(min_count=1) / ww.groupby(g).sum(min_count=1)
    lineup = pd.DataFrame(lineup).reset_index()
    lineup.columns = ["game_pk", "pitcher_id", *b_rates.columns]
    lineup_rows = df[["game_pk", "pitcher_id"]].merge(lineup, on=["game_pk", "pitcher_id"], how="left")

    for kind, name in (("whiff", "matchup_whiff_fine"), ("chase", "matchup_chase_fine")):
        num = np.zeros(len(df)); den = np.zeros(len(df))
        for j, c in enumerate(FINE_CLASSES):
            p = p_rates[f"{kind}_{c}"].to_numpy(float)
            b = lineup_rows[f"{kind}_{c}"].to_numpy(float)
            l = lg[f"{kind}_{c}"].to_numpy(float)
            o = odds(p) * odds(b) / odds(l)
            v = o / (1 + o)
            u = usage[:, j]
            ok = ~np.isnan(v) & ~np.isnan(u) & ~np.isnan(p) & ~np.isnan(b) & ~np.isnan(l)
            num += np.where(ok, u * v, 0.0); den += np.where(ok, u, 0.0)
        df[name] = np.where(den > 0.5, num / np.where(den > 0, den, 1), np.nan)
    return df

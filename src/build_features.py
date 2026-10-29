"""Build the pre-game modeling table for starting-pitcher strikeout prediction.

One row per start. The target is that game's strikeouts. Every feature is
computed only from information available before the game:

* Pitcher rolling features use ``groupby(pitcher).shift(1)`` before
  ``rolling(...)``, so the current start never enters its own window.
* Season-to-date rates use ``shift(1).cumsum()`` within (pitcher, season).
* Everything built from other games (league averages, opponent form, batter
  history, park and umpire factors) uses totals from dates strictly before
  the game date (``merge_asof(..., allow_exact_matches=False)`` or a one-date
  shift), so even the first game of a doubleheader never feeds the second.
* The opponent's posted starting lineup, the home plate umpire, the venue and
  the game-time weather are known before first pitch and used as-is.

Feature groups
--------------
``CORE_FEATURES``: pitcher form, workload, rest, home/away, season K rate,
opponent K rate. ``MATCHUP_FEATURES``: the opposing lineup's strikeout and
whiff tendencies against this pitcher's hand and pitch mix, and the pitcher's
platoon splits. ``CONTEXT_FEATURES``: weather, day/night, park and umpire
strikeout factors, opponent recent form.

Small samples are shrunk toward the league: a batter's rate is
``(events + m * league_rate) / (opportunities + m)`` with ``m`` from ``SHRINK``.

Incomplete windows (documented choice: "carry across seasons")
--------------------------------------------------------------
Rolling windows run over a pitcher's last N starts regardless of season, with
``min_periods=1``, so the first April start uses last September's starts.
Only a pitcher's first start in the data has no rolling values; those are left
as NaN and the models impute training-set medians (with missing-value
indicator columns). ``n_prior_starts`` tells the model how much history stands
behind the rolling values.

Season-to-date K rate and K per start use the current season's prior starts.
For the first start of a season they fall back to the pitcher's most recent
earlier season, then to the league average of all starts before that date.

Days of rest are capped at ``REST_CAP`` (30); ``first_start_of_season`` flags
the off-season case.

Upcoming starts
---------------
``build_features(raw, upcoming=...)`` also builds rows for games not played
yet (probable starters from ``fetch_data.fetch_upcoming``). Their game stats
are left empty, so they only ever read history, and an opponent without a
posted lineup gets its lineup from its previous game. Only each pitcher's
next upcoming start is kept, so no upcoming row feeds another pitcher's.

Usage::

    python src/build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_data import BATTER_AGG_COLUMNS, CLASS_COLUMNS, GAME_COLUMNS, LINEUP_COLUMNS, PITCH_CLASSES, load_raw

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

TARGET = "strikeouts"
REST_CAP = 30
# Recency-weighted K stats: a start EWM_HALFLIFE starts ago counts half as much as the latest.
EWM_HALFLIFE = 5
OPP_WINDOW = 30
OPP_FORM_WINDOW = 10
OPP_MIN_GAMES = 5
# Pseudo-counts for shrinking small samples toward the league rate.
SHRINK = {"batter_k": 100, "batter_whiff": 50, "pitcher_platoon": 80, "park": 2000, "ump": 1500}
# Typical plate appearances per game by batting-order slot (leadoff bats most).
LINEUP_PA_WEIGHTS = np.array([4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95, 3.85])
LINEUP_WHIFF_CLASSES = ["fb_hi", "brk", "os"]

KEY_COLUMNS = ["game_pk", "game_date", "season", "pitcher_id", "pitcher_name", "team_id", "opponent_id"]

ROLLING_FEATURES = [
    "fb_velo_last3", "fb_velo_last5",
    "whiff_rate_last3", "whiff_rate_last5",
    "k_per_bf_last5",
    "bf_last5", "pitches_last5",
    "usage_fb_hi_last5", "usage_brk_last5", "usage_os_last5",
    "k_rate_ewm", "k_per_start_ewm",
]
CORE_FEATURES = [
    "fb_velo_last3", "fb_velo_last5", "whiff_rate_last3", "whiff_rate_last5", "k_per_bf_last5",
    "bf_last5", "pitches_last5",
    "rest_days", "first_start_of_season", "is_home",
    "season_k_rate", "season_k_per_start",
    "opp_k_rate_last30", "n_prior_starts",
    "k_rate_ewm", "k_per_start_ewm",
]
MATCHUP_FEATURES = [
    "is_lhp",
    "pitcher_k_rate_vs_lineup",
    "lineup_k_rate_vs_hand",
    "lineup_platoon_adv_share",
    "usage_fb_hi_last5", "usage_brk_last5", "usage_os_last5",
    *[f"lineup_whiff_{c}" for c in LINEUP_WHIFF_CLASSES],
    "matchup_whiff_rate",
]
CONTEXT_FEATURES = [
    "day_game", "wind_out_mph", "roof_closed",
    "park_k_factor", "ump_k_factor",
]
FEATURES = CORE_FEATURES + MATCHUP_FEATURES + CONTEXT_FEATURES
# Built and kept in the table but not used by the model unless promoted into a
# group above after ``src/compare_features.py`` shows they help.
CANDIDATE_FEATURE_GROUPS = {
    "swinging strike rate": ["swstr_rate_last5"],
    "CSW rate": ["csw_rate_last5"],
    "pitches per PA": ["pitches_per_pa_last5"],
    "team hook": ["team_starter_bf_last30"],
    "own whiffs by pitch": ["own_whiff_fb_last5", "own_whiff_brk_last5", "own_whiff_os_last5"],
    "velo vs normal": ["fb_velo_vs_career"],
    "rest flags": ["short_rest", "long_layoff", "il_return"],
    # Dropped from the model in src/compare_features.py (no loss in either test season).
    "temperature and opponent form": ["temp_f", "opp_k_rate_last10"],
    "previous pitch count": ["pitches_prev"],
    "home-road park factor": ["park_k_factor_hr"],
    "per-batter matchup": ["matchup_expected_k", "lineup_log5_k_rate"],
}
CANDIDATE_FEATURES = [c for group in CANDIDATE_FEATURE_GROUPS.values() for c in group if c not in FEATURES]
# Candidates built from the pitcher's own earlier starts (NaN at his first start).
PITCHER_CANDIDATES = [
    "swstr_rate_last5", "csw_rate_last5", "pitches_per_pa_last5", "own_whiff_fb_last5", "own_whiff_brk_last5",
    "own_whiff_os_last5", "fb_velo_vs_career", "short_rest", "long_layoff", "il_return", "pitches_prev",
]
# Rest-day cut points for the flags (days since his previous start).
SHORT_REST_MAX, LONG_LAYOFF_MIN, IL_RETURN_MIN = 4, 10, 30
FEATURE_GROUPS = {"core": CORE_FEATURES, "matchup": MATCHUP_FEATURES, "context": CONTEXT_FEATURES}

# Stats that describe the game being predicted. None may ever be a feature.
CURRENT_GAME_COLUMNS = {
    "strikeouts", "batters_faced", "outs", "pitches", "sc_pitches",
    "fb_count", "fb_velo_sum", "fb_velo", "swings", "whiffs", "called", "bat_k", "bat_pa",
    "pa_vs_l", "k_vs_l", "pa_vs_r", "k_vs_r", *CLASS_COLUMNS,
}
# Kept in the processed table (never as a feature) for the leakage demo in train.py.
LEAKY_DEMO_COLUMN = "batters_faced"


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def to_date(s: pd.Series) -> pd.Series:
    """Parse dates to a single datetime64[ns] dtype (merge_asof needs matching dtypes)."""
    return pd.to_datetime(s).astype("datetime64[ns]")


def numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Coerce columns to float64 in place and return the frame."""
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def cumulative_by_date(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Running totals of ``cols`` per key, one row per (key, date), inclusive of that date."""
    daily = df.groupby([*keys, "game_date"], as_index=False)[cols].sum()
    daily = daily.sort_values([*keys, "game_date"]).reset_index(drop=True)
    if keys:
        daily[cols] = daily.groupby(keys)[cols].cumsum()
    else:
        daily[cols] = daily[cols].cumsum()
    return daily


def prior_totals(left: pd.DataFrame, history: pd.DataFrame, keys: list[str], cols: list[str],
                 prefix: str = "") -> pd.DataFrame:
    """Attach totals of ``cols`` from ``history`` rows with the same keys and *earlier* dates.

    Rows with no earlier history get 0. Uses ``merge_asof`` with
    ``allow_exact_matches=False``, so same-date rows (including doubleheaders)
    are always excluded.
    """
    cum = cumulative_by_date(history, keys, cols).rename(columns={c: prefix + c for c in cols})
    # Statcast IDs arrive as nullable Int64, boxscore IDs as int64; merge_asof needs one dtype.
    cum = cum.astype({k: left[k].dtype for k in keys})
    left = left.assign(_order=np.arange(len(left))).sort_values("game_date")
    out = pd.merge_asof(left, cum.sort_values("game_date"), on="game_date",
                        by=keys or None, allow_exact_matches=False)
    out[[prefix + c for c in cols]] = out[[prefix + c for c in cols]].fillna(0.0)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def shrunk_rate(num: pd.Series, den: pd.Series, prior_rate: pd.Series | float, m: float) -> pd.Series:
    """Empirical-Bayes rate: (num + m * prior_rate) / (den + m)."""
    return (num + m * prior_rate) / (den + m)


# --------------------------------------------------------------------------- #
# Core pitcher features
# --------------------------------------------------------------------------- #
def prepare_starts(game_logs: pd.DataFrame, statcast: pd.DataFrame) -> pd.DataFrame:
    """Keep starts only, attach Statcast summaries, and sort chronologically per pitcher.

    Upcoming starts (``upcoming == 1``) are kept even though their stats are empty.
    """
    starts = game_logs[game_logs["started"] == 1].copy()
    if "upcoming" not in starts:
        starts["upcoming"] = 0
    played = starts[["strikeouts", "batters_faced"]].notna().all(axis=1)
    starts = starts[played | (starts["upcoming"] == 1)]
    starts["game_date"] = to_date(starts["game_date"])
    sc_cols = ["sc_pitches", "fb_count", "fb_velo_sum", "swings", "whiffs", "called", *CLASS_COLUMNS]
    sc = statcast.reindex(columns=["game_pk", "pitcher_id", *sc_cols])
    starts = starts.merge(sc, on=["game_pk", "pitcher_id"], how="left")
    numeric(starts, ["strikeouts", "batters_faced", "outs", "pitches", "is_home", "season", *sc_cols])
    starts["fb_velo"] = starts["fb_velo_sum"] / starts["fb_count"].where(starts["fb_count"] > 0)
    starts["classified"] = starts[[f"n_{c}" for c in PITCH_CLASSES]].sum(axis=1, min_count=1)
    return starts.sort_values(["pitcher_id", "game_date", "game_pk"]).reset_index(drop=True)


def prior_rolling(starts: pd.DataFrame, col: str, window: int, how: str = "sum") -> pd.Series:
    """Rolling stat over a pitcher's previous ``window`` starts, excluding the current one.

    ``shift(1)`` drops the current start; ``min_periods=1`` lets partial windows
    through so only a pitcher's first start is NaN.
    """
    return starts.groupby("pitcher_id")[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).agg(how)
    )


def add_rolling_features(starts: pd.DataFrame) -> pd.DataFrame:
    """Add velocity, whiff, K/BF, workload, pitch-mix and recency-weighted K features."""
    df = starts
    for n in (3, 5):
        df[f"fb_velo_last{n}"] = prior_rolling(df, "fb_velo", n, "mean")
        whiffs = prior_rolling(df, "whiffs", n)
        swings = prior_rolling(df, "swings", n)
        df[f"whiff_rate_last{n}"] = whiffs / swings.where(swings > 0)
    k5 = prior_rolling(df, "strikeouts", 5)
    bf5 = prior_rolling(df, "batters_faced", 5)
    df["k_per_bf_last5"] = k5 / bf5.where(bf5 > 0)
    df["bf_last5"] = prior_rolling(df, "batters_faced", 5, "mean")
    df["pitches_last5"] = prior_rolling(df, "pitches", 5, "mean")
    g = df.groupby("pitcher_id")
    k_ewm = g["strikeouts"].transform(lambda s: s.shift(1).ewm(halflife=EWM_HALFLIFE).mean())
    bf_ewm = g["batters_faced"].transform(lambda s: s.shift(1).ewm(halflife=EWM_HALFLIFE).mean())
    df["k_per_start_ewm"] = k_ewm
    df["k_rate_ewm"] = k_ewm / bf_ewm.where(bf_ewm > 0)
    total5 = prior_rolling(df, "classified", 5)
    for c in PITCH_CLASSES:
        df[f"usage_{c}_last5"] = prior_rolling(df, f"n_{c}", 5) / total5.where(total5 > 0)
    return add_candidate_rolling(df)


def add_candidate_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate pitcher features: SwStr%, CSW%, pitches per PA, own whiffs by pitch, velo vs career."""
    pitches5 = prior_rolling(df, "sc_pitches", 5)
    whiffs5 = prior_rolling(df, "whiffs", 5)
    called5 = prior_rolling(df, "called", 5)
    has_called = prior_rolling(df.assign(_c=df["called"].notna().astype(float)), "_c", 5) > 0
    df["swstr_rate_last5"] = whiffs5 / pitches5.where(pitches5 > 0)
    df["csw_rate_last5"] = ((called5 + whiffs5) / pitches5.where(pitches5 > 0)).where(has_called)
    box_pitches5 = prior_rolling(df, "pitches", 5)
    bf5 = prior_rolling(df, "batters_faced", 5)
    df["pitches_per_pa_last5"] = box_pitches5 / bf5.where(bf5 > 0)
    groups = {"fb": ["fb_lo", "fb_mid", "fb_hi"], "brk": ["brk"], "os": ["os"]}
    for name, classes in groups.items():
        wh = df[[f"wh_{c}" for c in classes]].sum(axis=1, min_count=1)
        sw = df[[f"sw_{c}" for c in classes]].sum(axis=1, min_count=1)
        wh5 = prior_rolling(df.assign(_v=wh), "_v", 5)
        sw5 = prior_rolling(df.assign(_v=sw), "_v", 5)
        df[f"own_whiff_{name}_last5"] = wh5 / sw5.where(sw5 >= 10)
    career = df.groupby("pitcher_id")["fb_velo"].transform(lambda s: s.shift(1).expanding().mean())
    df["fb_velo_vs_career"] = df["fb_velo_last3"] - career
    return df


def team_hook(starts: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Each team's average batters faced by its starters over its previous ``window`` game-days.

    A low value means a quick-hook team. Totals are per team per date, shifted
    one date, so a doubleheader's first game never feeds the second.
    """
    daily = (starts.groupby(["team_id", "game_date"])["batters_faced"]
             .agg(bf="sum", n="count").reset_index().sort_values(["team_id", "game_date"]))
    g = daily.groupby("team_id")
    bf = g["bf"].transform(lambda s: s.shift(1).rolling(window, min_periods=5).sum())
    n = g["n"].transform(lambda s: s.shift(1).rolling(window, min_periods=5).sum())
    daily["team_starter_bf_last30"] = bf / n.where(n > 0)
    return daily[["team_id", "game_date", "team_starter_bf_last30"]]


def add_rest_features(starts: pd.DataFrame) -> pd.DataFrame:
    """Add days since the previous start (capped), rest flags, the previous pitch count and a first-start flag."""
    df = starts
    g = df.groupby("pitcher_id")
    prev_season = g["season"].shift(1)
    rest = g["game_date"].diff().dt.days
    df["rest_days"] = rest.clip(upper=REST_CAP)
    # Flags for the non-linear part: short rest, a layoff, a likely IL return (NaN at a first start).
    df["short_rest"] = (rest <= SHORT_REST_MAX).astype("float64").where(rest.notna())
    df["long_layoff"] = (rest >= LONG_LAYOFF_MIN).astype("float64").where(rest.notna())
    df["il_return"] = (rest >= IL_RETURN_MIN).astype("float64").where(rest.notna())
    df["pitches_prev"] = g["pitches"].shift(1)
    df["first_start_of_season"] = (prev_season != df["season"]).astype("float64")
    df["n_prior_starts"] = g.cumcount().astype("float64")
    return df


def league_prior_averages(starts: pd.DataFrame) -> pd.DataFrame:
    """League K/BF and K per start using only dates strictly before each game date."""
    daily = (
        starts.groupby("game_date")
        .agg(k=("strikeouts", "sum"), bf=("batters_faced", "sum"), n=("strikeouts", "count"))
        .sort_index()
    )
    cum = daily.cumsum().shift(1)
    return pd.DataFrame({
        "league_k_rate": cum["k"] / cum["bf"],
        "league_k_per_start": cum["k"] / cum["n"],
    })


def add_season_features(starts: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date K rate and K per start, with prior-season then league fallbacks."""
    df = starts
    gs = df.groupby(["pitcher_id", "season"])
    prior_k = gs["strikeouts"].transform(lambda s: s.shift(1).cumsum())
    prior_bf = gs["batters_faced"].transform(lambda s: s.shift(1).cumsum())
    prior_n = gs.cumcount()
    season_k_rate = prior_k / prior_bf.where(prior_bf > 0)
    season_k_per_start = prior_k / prior_n.where(prior_n > 0)

    totals = (
        df.groupby(["pitcher_id", "season"])
        .agg(k=("strikeouts", "sum"), bf=("batters_faced", "sum"), n=("strikeouts", "size"))
        .reset_index()
        .sort_values(["pitcher_id", "season"])
    )
    gp = totals.groupby("pitcher_id")
    totals["prev_k_rate"] = gp["k"].shift(1) / gp["bf"].shift(1)
    totals["prev_k_per_start"] = gp["k"].shift(1) / gp["n"].shift(1)
    prev = df[["pitcher_id", "season"]].merge(
        totals[["pitcher_id", "season", "prev_k_rate", "prev_k_per_start"]],
        on=["pitcher_id", "season"], how="left",
    )

    league = league_prior_averages(df)
    df["league_k_rate"] = df["game_date"].map(league["league_k_rate"])
    league_per_start = df["game_date"].map(league["league_k_per_start"])

    df["season_k_rate"] = season_k_rate.fillna(prev["prev_k_rate"]).fillna(df["league_k_rate"])
    df["season_k_per_start"] = season_k_per_start.fillna(prev["prev_k_per_start"]).fillna(league_per_start)
    return df


def opponent_k_rates(team_games: pd.DataFrame, window: int = OPP_WINDOW,
                     name: str = "opp_k_rate_last30") -> pd.DataFrame:
    """Each team's batting K rate over its previous ``window`` game-days.

    Totals are summed per team per date first and then shifted one date, so a
    doubleheader's first game never feeds the second game's feature.
    """
    tg = team_games.copy()
    tg["game_date"] = to_date(tg["game_date"])
    numeric(tg, ["bat_k", "bat_pa"])
    daily = (
        tg.groupby(["team_id", "game_date"], as_index=False)[["bat_k", "bat_pa"]].sum(min_count=1)
        .sort_values(["team_id", "game_date"])
    )
    g = daily.groupby("team_id")
    k = g["bat_k"].transform(lambda s: s.shift(1).rolling(window, min_periods=OPP_MIN_GAMES).sum())
    pa = g["bat_pa"].transform(lambda s: s.shift(1).rolling(window, min_periods=OPP_MIN_GAMES).sum())
    daily[name] = k / pa.where(pa > 0)
    return daily[["team_id", "game_date", name]]


# --------------------------------------------------------------------------- #
# Matchup features: lineup handedness, platoon splits, pitch mix vs. whiffs
# --------------------------------------------------------------------------- #
def pitcher_hands(statcast: pd.DataFrame) -> pd.Series:
    """Each pitcher's throwing hand ("L"/"R"), the most common value in Statcast."""
    sc = statcast.dropna(subset=["p_throws"])
    if sc.empty:
        return pd.Series(dtype="object")
    return sc.groupby("pitcher_id")["p_throws"].agg(lambda s: s.mode().iloc[0])


def batter_hands(batter_games: pd.DataFrame) -> pd.Series:
    """Each batter's side: "S" if they bat both ways (10%+ of PAs each side), else "L"/"R".

    Handedness is a fixed trait known before any game, so all cached games are used.
    """
    if batter_games.empty:
        return pd.Series(dtype="object")
    tot = batter_games.groupby("batter_id")[["pa_as_l", "pa_as_r"]].sum()
    share_l = tot["pa_as_l"] / (tot["pa_as_l"] + tot["pa_as_r"]).where(lambda x: x > 0)
    hand = np.where(share_l >= 0.9, "L", np.where(share_l <= 0.1, "R", "S"))
    return pd.Series(hand, index=tot.index).where(share_l.notna())


def opponent_lineups(starts: pd.DataFrame, lineups: pd.DataFrame, source: str = "posted") -> pd.DataFrame:
    """The nine batters each starter faces, one row per (start, batter).

    ``source="posted"`` uses the opponent's posted lineup for that game (known a
    few hours before first pitch). ``source="previous"`` uses the opponent's
    lineup from its previous game, for predictions made a day ahead.
    """
    lu = lineups.copy()
    lu["game_date"] = to_date(lu["game_date"])
    if source == "previous":
        games = (lu[["team_id", "game_pk", "game_date"]].drop_duplicates()
                 .sort_values(["team_id", "game_date", "game_pk"]))
        games["prev_pk"] = games.groupby("team_id")["game_pk"].shift(1)
        prev = games.dropna(subset=["prev_pk"]).astype({"prev_pk": "int64"})
        lu = prev[["team_id", "game_pk", "prev_pk"]].merge(
            lu.drop(columns=["game_date"]).rename(columns={"game_pk": "prev_pk"}), on=["team_id", "prev_pk"]
        ).drop(columns="prev_pk")
    lu = lu.rename(columns={"team_id": "opponent_id"})[["game_pk", "opponent_id", "batter_id", "batting_order"]]
    pairs = starts[["game_pk", "pitcher_id", "opponent_id", "game_date", "p_hand"]]
    return pairs.merge(lu, on=["game_pk", "opponent_id"], how="inner")


def matchup_features(starts: pd.DataFrame, raw: dict[str, pd.DataFrame],
                     lineup_source: str = "posted") -> pd.DataFrame:
    """Lineup-level strikeout, platoon and pitch-class whiff features for each start."""
    keys = ["game_pk", "pitcher_id"]
    out = starts[keys].copy()
    for col in ("pitcher_k_rate_vs_lineup", "lineup_k_rate_vs_hand", "lineup_platoon_adv_share",
                "lineup_log5_k_rate", *[f"lineup_whiff_{c}" for c in PITCH_CLASSES],
                *[f"log5_slot{i}" for i in range(1, 10)]):
        out[col] = np.nan
    bg, sc, lineups = raw["batter_games"], raw["statcast"], raw["lineups"]
    if bg.empty or lineups.empty:
        return out

    count_cols = [c for c in BATTER_AGG_COLUMNS if c not in ("game_pk", "batter_id", "game_date", "bat_team")]
    bg = numeric(bg.copy(), count_cols)
    bg["game_date"] = to_date(bg["game_date"])
    lu = opponent_lineups(starts, lineups, lineup_source)
    lu["bat_hand"] = lu["batter_id"].map(batter_hands(bg))
    lu["weight"] = LINEUP_PA_WEIGHTS[(lu["batting_order"].clip(1, 9) - 1).astype(int)]

    # Batter history before this date, and the league's, for shrinkage.
    hist_cols = ["pa_vs_lhp", "k_vs_lhp", "pa_vs_rhp", "k_vs_rhp",
                 *[f"{k}_{c}" for c in PITCH_CLASSES for k in ("sw", "wh")]]
    lu = prior_totals(lu, bg, ["batter_id"], hist_cols, prefix="b_")
    lu = prior_totals(lu, bg, [], hist_cols, prefix="lg_")

    vs_l = (lu["p_hand"] == "L").to_numpy()
    lg_k = np.where(vs_l, lu["lg_k_vs_lhp"] / lu["lg_pa_vs_lhp"].where(lu["lg_pa_vs_lhp"] > 0),
                    lu["lg_k_vs_rhp"] / lu["lg_pa_vs_rhp"].where(lu["lg_pa_vs_rhp"] > 0))
    b_k = np.where(vs_l, lu["b_k_vs_lhp"], lu["b_k_vs_rhp"])
    b_pa = np.where(vs_l, lu["b_pa_vs_lhp"], lu["b_pa_vs_rhp"])
    lu["k_rate_vs_hand"] = shrunk_rate(b_k, b_pa, lg_k, SHRINK["batter_k"])
    for c in PITCH_CLASSES:
        lg = lu[f"lg_wh_{c}"] / lu[f"lg_sw_{c}"].where(lu[f"lg_sw_{c}"] > 0)
        lu[f"whiff_{c}"] = shrunk_rate(lu[f"b_wh_{c}"], lu[f"b_sw_{c}"], lg, SHRINK["batter_whiff"])

    # Platoon: a switch hitter always bats from the side opposite the pitcher.
    hand = lu["bat_hand"].to_numpy()
    side = np.where(hand == "S", np.where(vs_l, "R", "L"), hand)
    known = lu["bat_hand"].notna().to_numpy() & lu["p_hand"].notna().to_numpy()
    lu["platoon_adv"] = np.where(known, (side != lu["p_hand"].to_numpy()).astype(float), np.nan)

    # Pitcher's K rate against the side each batter will hit from.
    split_cols = ["pa_vs_l", "k_vs_l", "pa_vs_r", "k_vs_r"]
    sc_hist = numeric(sc[["pitcher_id", "game_date", *split_cols]].copy(), split_cols)
    sc_hist["game_date"] = to_date(sc_hist["game_date"])
    lu = prior_totals(lu, sc_hist, ["pitcher_id"], split_cols, prefix="p_")
    lu = prior_totals(lu, sc_hist, [], split_cols, prefix="plg_")
    bats_l = side == "L"
    p_k = np.where(bats_l, lu["p_k_vs_l"], lu["p_k_vs_r"])
    p_pa = np.where(bats_l, lu["p_pa_vs_l"], lu["p_pa_vs_r"])
    plg = np.where(bats_l, lu["plg_k_vs_l"] / lu["plg_pa_vs_l"].where(lu["plg_pa_vs_l"] > 0),
                   lu["plg_k_vs_r"] / lu["plg_pa_vs_r"].where(lu["plg_pa_vs_r"] > 0))
    lu["pitcher_k_vs_side"] = np.where(known, shrunk_rate(p_k, p_pa, plg, SHRINK["pitcher_platoon"]), np.nan)

    # Per-batter K chance vs this pitcher (odds ratio, "log5"): batter vs hand x pitcher vs side / league.
    def odds(x):
        x = np.clip(np.asarray(x, float), 1e-4, 1 - 1e-4)
        return x / (1 - x)
    o = odds(lu["k_rate_vs_hand"]) * odds(lu["pitcher_k_vs_side"]) / odds(lg_k)
    lu["log5_k"] = np.where(np.isnan(o), np.nan, o / (1 + o))

    groups = [lu["game_pk"], lu["pitcher_id"]]

    def wmean(col: str) -> pd.Series:
        v = lu[col]
        w = lu["weight"].where(v.notna())
        return (v * w).groupby(groups).sum(min_count=1) / w.groupby(groups).sum(min_count=1)

    agg = pd.DataFrame({
        "pitcher_k_rate_vs_lineup": wmean("pitcher_k_vs_side"),
        "lineup_k_rate_vs_hand": wmean("k_rate_vs_hand"),
        "lineup_platoon_adv_share": wmean("platoon_adv"),
        "lineup_log5_k_rate": wmean("log5_k"),
        **{f"lineup_whiff_{c}": wmean(f"whiff_{c}") for c in PITCH_CLASSES},
    })
    agg.index.names = keys
    slots = (lu.assign(slot=lu["batting_order"].clip(1, 9).astype(int))
             .pivot_table(index=keys, columns="slot", values="log5_k", aggfunc="mean")
             .reindex(columns=range(1, 10)))
    slots.columns = [f"log5_slot{i}" for i in slots.columns]
    agg = agg.join(slots)
    return out[keys].merge(agg.reset_index(), on=keys, how="left")


def add_matchup_features(starts: pd.DataFrame, raw: dict[str, pd.DataFrame], lineup_source: str) -> pd.DataFrame:
    """Attach pitcher hand, lineup features, and the pitch-mix x lineup-whiff matchup rate."""
    starts["p_hand"] = starts["pitcher_id"].map(pitcher_hands(raw["statcast"]))
    starts["is_lhp"] = (starts["p_hand"] == "L").astype(float).where(starts["p_hand"].notna())
    starts = starts.merge(matchup_features(starts, raw, lineup_source), on=["game_pk", "pitcher_id"], how="left")
    usage = starts[[f"usage_{c}_last5" for c in PITCH_CLASSES]].to_numpy(float)
    whiff = starts[[f"lineup_whiff_{c}" for c in PITCH_CLASSES]].to_numpy(float)
    both = ~np.isnan(usage) & ~np.isnan(whiff)
    weight = np.where(both, usage, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(both, usage * whiff, 0.0).sum(axis=1) / weight
    starts["matchup_whiff_rate"] = np.where(weight > 0, rate, np.nan)
    starts["matchup_expected_k"] = expected_lineup_k(starts)
    return starts.drop(columns=[f"log5_slot{i}" for i in range(1, 10)])


def slot_pa(bf: np.ndarray) -> np.ndarray:
    """Expected PA for each lineup slot (columns 1..9) when the starter faces ``bf`` batters.

    For whole ``bf``, slot i bats floor((bf - i) / 9) + 1 times (0 if bf < i);
    fractional ``bf`` interpolates between the neighbouring whole numbers.
    """
    bf = np.clip(np.asarray(bf, float), 0, None)[:, None]
    i = np.arange(1, 10)[None, :]

    def whole(n):
        return np.where(n >= i, np.floor((n - i) / 9) + 1, 0.0)

    lo = np.floor(bf)
    return whole(lo) + (bf - lo) * (whole(lo + 1) - whole(lo))


def expected_lineup_k(starts: pd.DataFrame) -> np.ndarray:
    """Sum over lineup slots of each batter's K chance vs this pitcher x his expected PA vs the starter.

    Expected batters faced is the pitcher's average over his last 5 starts. A
    slot with no K chance takes the average of the other slots.
    """
    p = starts[[f"log5_slot{i}" for i in range(1, 10)]].to_numpy(float)
    with np.errstate(invalid="ignore"):
        fill = np.nanmean(np.where(np.isnan(p).all(axis=1, keepdims=True), 0.0, p), axis=1, keepdims=True)
    known = ~np.isnan(p).all(axis=1)
    p = np.where(np.isnan(p), fill, p)
    k = (p * slot_pa(starts["bf_last5"].fillna(0).to_numpy())).sum(axis=1)
    return np.where(known & starts["bf_last5"].notna().to_numpy(), k, np.nan)


# --------------------------------------------------------------------------- #
# Context features: weather, day/night, park, umpire, opponent form
# --------------------------------------------------------------------------- #
def wind_out_mph(wind_mph: pd.Series, wind_dir: pd.Series) -> pd.Series:
    """Signed wind: positive blowing out, negative blowing in, 0 across or calm."""
    d = wind_dir.fillna("").astype(str).str.lower()
    sign = np.where(d.str.startswith("out"), 1.0, np.where(d.str.startswith("in"), -1.0, 0.0))
    return wind_mph.fillna(0.0) * sign


def key_k_factor(team_games: pd.DataFrame, games: pd.DataFrame, key: str, m: float) -> pd.Series:
    """Prior K/PA at each value of ``key`` (venue or umpire) relative to the league, shrunk.

    Uses only games on earlier dates. Returns a Series indexed by game_pk.
    """
    tg = numeric(team_games[["game_pk", "game_date", "bat_k", "bat_pa"]].copy(), ["bat_k", "bat_pa"])
    tg["game_date"] = to_date(tg["game_date"])
    per_game = tg.groupby(["game_pk", "game_date"], as_index=False)[["bat_k", "bat_pa"]].sum()
    per_game = per_game.merge(games[["game_pk", key]], on="game_pk", how="inner").dropna(subset=[key])
    if per_game.empty:
        return pd.Series(dtype="float64")
    left = per_game[["game_pk", "game_date", key]].copy()
    left = prior_totals(left, per_game, [key], ["bat_k", "bat_pa"], prefix="k_")
    left = prior_totals(left, per_game, [], ["bat_k", "bat_pa"], prefix="lg_")
    lg = left["lg_bat_k"] / left["lg_bat_pa"].where(left["lg_bat_pa"] > 0)
    return pd.Series((shrunk_rate(left["k_bat_k"], left["k_bat_pa"], lg, m) / lg).to_numpy(),
                     index=left["game_pk"].to_numpy())


def home_road_park_factor(team_games: pd.DataFrame, games: pd.DataFrame, home_team: pd.Series,
                          m: float) -> pd.Series:
    """Prior K/PA in the home team's games at this venue / K/PA in its road games, shrunk.

    Both rates count both teams' batters, so the home team's own hitters and
    pitchers appear on both sides of the ratio and cancel out. Keyed by
    (home team, venue ID), so a renamed park keeps its history and a team in
    a temporary park (the Rays at Steinbrenner Field in 2025) starts fresh,
    shrunk toward 1. Uses only games on earlier dates. Indexed by game_pk.
    """
    tg = numeric(team_games[["game_pk", "game_date", "team_id", "bat_k", "bat_pa"]].copy(), ["bat_k", "bat_pa"])
    tg["game_date"] = to_date(tg["game_date"])
    per_game = tg.groupby(["game_pk", "game_date"], as_index=False)[["bat_k", "bat_pa"]].sum()
    per_game["home_id"] = per_game["game_pk"].map(home_team)
    per_game = per_game.merge(games[["game_pk", "venue_id"]], on="game_pk", how="inner").dropna(
        subset=["home_id", "venue_id"])
    if per_game.empty:
        return pd.Series(dtype="float64")
    per_game = per_game.astype({"home_id": "int64", "venue_id": "int64"})
    # One row per team per game, flagged home or away.
    teams = tg[["game_pk", "team_id"]].drop_duplicates().merge(per_game, on="game_pk")
    teams["team_id"] = teams["team_id"].astype("int64")
    road = teams[teams["team_id"] != teams["home_id"]]

    left = per_game[["game_pk", "game_date", "home_id", "venue_id"]].copy()
    left = prior_totals(left, per_game, ["home_id", "venue_id"], ["bat_k", "bat_pa"], prefix="h_")
    left = prior_totals(left, road.drop(columns="home_id").rename(columns={"team_id": "home_id"}),
                        ["home_id"], ["bat_k", "bat_pa"], prefix="r_")
    left = prior_totals(left, per_game, [], ["bat_k", "bat_pa"], prefix="lg_")
    lg = left["lg_bat_k"] / left["lg_bat_pa"].where(left["lg_bat_pa"] > 0)
    home = shrunk_rate(left["h_bat_k"], left["h_bat_pa"], lg, m)
    away = shrunk_rate(left["r_bat_k"], left["r_bat_pa"], lg, m)
    return pd.Series((home / away).to_numpy(), index=left["game_pk"].to_numpy())


def home_teams(game_logs: pd.DataFrame) -> pd.Series:
    """Home team ID per game_pk, from the starters' home/away flag."""
    logs = game_logs[["game_pk", "team_id", "is_home"]].dropna()
    home = logs[pd.to_numeric(logs["is_home"], errors="coerce") == 1].drop_duplicates("game_pk")
    return pd.Series(home["team_id"].astype("int64").to_numpy(), index=home["game_pk"].astype("int64").to_numpy())


def add_context_features(starts: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach weather, day/night, park and umpire factors, and opponent recent form."""
    games = raw["games"]
    ctx_cols = ["day_game", "temp_f", "wind_out_mph", "roof_closed", "park_k_factor", "park_k_factor_hr",
                "ump_k_factor"]
    if games.empty:
        for col in ctx_cols:
            starts[col] = np.nan
    else:
        g = games.copy()
        g["temp_f"] = pd.to_numeric(g["temp_f"], errors="coerce")
        g["wind_mph"] = pd.to_numeric(g["wind_mph"], errors="coerce")
        cond = g["condition"].fillna("").astype(str).str.lower()
        g["roof_closed"] = cond.isin(["dome", "roof closed"]).astype(float)
        g["wind_out_mph"] = wind_out_mph(g["wind_mph"], g["wind_dir"]).where(g["roof_closed"] == 0, 0.0)
        g["day_game"] = (g["day_night"] == "day").astype(float).where(g["day_night"].notna())
        g["park_k_factor"] = g["game_pk"].map(key_k_factor(raw["team_games"], g, "venue_id", SHRINK["park"]))
        g["ump_k_factor"] = g["game_pk"].map(key_k_factor(raw["team_games"], g, "hp_umpire_id", SHRINK["ump"]))
        g["park_k_factor_hr"] = g["game_pk"].map(home_road_park_factor(
            raw["team_games"], g, home_teams(raw["game_logs"]), SHRINK["park"]))
        starts = starts.merge(g[["game_pk", *ctx_cols]], on="game_pk", how="left")

    form = opponent_k_rates(raw["team_games"], OPP_FORM_WINDOW, "opp_k_rate_last10")
    return starts.merge(form.rename(columns={"team_id": "opponent_id"}), on=["opponent_id", "game_date"], how="left")


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #
def previous_lineups(lineups: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    """Each team's lineup from its latest game before the date, for (game_pk, game_date, team_id) rows."""
    if lineups.empty or missing.empty:
        return pd.DataFrame(columns=LINEUP_COLUMNS)
    lu = lineups.copy()
    lu["game_date"] = to_date(lu["game_date"])
    last = (lu[["team_id", "game_pk", "game_date"]].drop_duplicates()
            .sort_values(["game_date", "game_pk"]).rename(columns={"game_pk": "prev_pk"}))
    want = missing.assign(game_date=to_date(missing["game_date"])).sort_values("game_date")
    want = want.astype({"team_id": last["team_id"].dtype})
    want = pd.merge_asof(want, last, on="game_date", by="team_id", allow_exact_matches=False).dropna(subset=["prev_pk"])
    prev = lu.drop(columns="game_date").rename(columns={"game_pk": "prev_pk"})
    out = want.astype({"prev_pk": prev["prev_pk"].dtype}).merge(prev, on=["team_id", "prev_pk"])
    return out[LINEUP_COLUMNS]


def add_upcoming(raw: dict[str, pd.DataFrame], upcoming: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Append upcoming starts to the raw tables, with empty game stats and filled-in lineups."""
    up = upcoming["starts"].copy()
    up["game_date"] = up["game_date"].astype(str).str[:10]
    up = up.sort_values(["game_date", "game_pk"]).drop_duplicates("pitcher_id")
    pks = set(up["game_pk"])
    logs = up[[c for c in raw["game_logs"].columns if c in up.columns]].assign(started=1, upcoming=1)
    out = dict(raw)
    out["game_logs"] = pd.concat([raw["game_logs"].assign(upcoming=0), logs], ignore_index=True)

    teams = upcoming["team_games"]
    teams = teams[teams["game_pk"].isin(pks)]
    out["team_games"] = pd.concat([raw["team_games"], teams], ignore_index=True)
    games = upcoming["games"]
    out["games"] = pd.concat([raw["games"], games[games["game_pk"].isin(pks)]], ignore_index=True)

    posted = upcoming["lineups"]
    posted = posted[posted["game_pk"].isin(pks)]
    have = set(zip(posted["game_pk"], posted["team_id"], strict=True))
    need = teams[[(pk, t) not in have for pk, t in zip(teams["game_pk"], teams["team_id"], strict=True)]]
    filled = previous_lineups(raw["lineups"], need[["game_pk", "game_date", "team_id"]])
    out["lineups"] = pd.concat([raw["lineups"], posted, filled], ignore_index=True)
    return out


EMPTY_TABLES = {
    "team_games": ["game_pk", "game_date", "season", "team_id", "opponent_id", "bat_pa", "bat_k"],
    "batter_games": BATTER_AGG_COLUMNS,
    "games": GAME_COLUMNS,
    "lineups": LINEUP_COLUMNS,
}


def build_features(raw: dict[str, pd.DataFrame], lineup_source: str = "posted",
                   upcoming: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Return the modeling table: keys, target, pre-game features, and the leakage-demo column.

    ``raw`` holds the cached tables from ``fetch_data.load_raw()``. Only
    ``game_logs`` and ``statcast`` are required; features from a missing table
    are left NaN. With ``upcoming`` (from ``fetch_data.fetch_upcoming``), rows
    for those starts are added with an empty target and ``upcoming == 1``.
    """
    raw = {**{k: pd.DataFrame(columns=v) for k, v in EMPTY_TABLES.items()}, **raw}
    if upcoming is not None and not upcoming["starts"].empty:
        raw = add_upcoming(raw, upcoming)
    starts = prepare_starts(raw["game_logs"], raw["statcast"])
    starts = add_rolling_features(starts)
    starts = add_rest_features(starts)
    starts = add_season_features(starts)
    starts = starts.merge(team_hook(starts), on=["team_id", "game_date"], how="left")

    opp = opponent_k_rates(raw["team_games"]).rename(columns={"team_id": "opponent_id"})
    starts = starts.merge(opp, on=["opponent_id", "game_date"], how="left")
    starts = add_matchup_features(starts, raw, lineup_source)
    starts = add_context_features(starts, raw)

    cols = KEY_COLUMNS + [TARGET, LEAKY_DEMO_COLUMN, "p_hand"] + FEATURES + CANDIDATE_FEATURES
    if upcoming is not None:
        cols += ["upcoming"]
    out = starts[cols].sort_values(["game_date", "game_pk", "pitcher_id"]).reset_index(drop=True)
    out["season"] = out["season"].astype(int)
    return out


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
def _perturb_date(raw: dict[str, pd.DataFrame], day) -> dict[str, pd.DataFrame]:
    """Copy the raw tables with every game stat on ``day`` set to absurd values.

    Pre-game inputs (lineups, venue, weather, umpire) are left alone: they are
    legitimately known before the game.
    """
    day = pd.Timestamp(day)
    out = {k: v.copy() for k, v in raw.items()}

    def on_day(df):
        return to_date(df["game_date"]) == day

    logs = out["game_logs"]
    hit = on_day(logs) & (logs["started"] == 1)
    logs.loc[hit, ["strikeouts", "batters_faced", "outs", "pitches"]] = [99, 99, 27, 200]

    sc = out["statcast"]
    s_hit = on_day(sc)
    sc.loc[s_hit, "fb_velo_sum"] = sc.loc[s_hit, "fb_count"] * 110.0
    sc.loc[s_hit, "whiffs"] = sc.loc[s_hit, "swings"]
    for side in ("l", "r"):
        if f"k_vs_{side}" in sc:
            sc.loc[s_hit, f"k_vs_{side}"] = sc.loc[s_hit, f"pa_vs_{side}"]
    for c in PITCH_CLASSES:
        if f"n_{c}" in sc:
            sc.loc[s_hit, f"wh_{c}"] = sc.loc[s_hit, f"sw_{c}"]
            sc.loc[s_hit, f"n_{c}"] = sc.loc[s_hit, "sc_pitches"] if c == "fb_hi" else 0

    if "team_games" in out:
        tg = out["team_games"]
        tg.loc[on_day(tg), ["bat_k", "bat_pa"]] = [99, 99]
    if "batter_games" in out and not out["batter_games"].empty:
        bg = out["batter_games"]
        b_hit = on_day(bg)
        bg.loc[b_hit, "k_vs_lhp"] = bg.loc[b_hit, "pa_vs_lhp"]
        bg.loc[b_hit, "k_vs_rhp"] = bg.loc[b_hit, "pa_vs_rhp"]
        for c in PITCH_CLASSES:
            bg.loc[b_hit, f"wh_{c}"] = bg.loc[b_hit, f"sw_{c}"]
    return out


def audit_leakage(raw: dict[str, pd.DataFrame], feature_cols: list[str], n_dates: int = 5,
                  seed: int = 42, lineup_source: str = "posted") -> dict:
    """Check a feature set for leakage of current-game information.

    Three checks:

    1. **Name check**: no feature is a current-game stat (``CURRENT_GAME_COLUMNS``).
    2. **Shift check**: every pitcher rolling feature is NaN at each pitcher's
       first start. If a window were not shifted, the first start would see itself.
    3. **Perturbation check**: for a few random dates, set every game stat on
       that date to absurd values (99 K, 99 batters faced, 110 mph, every swing
       a whiff, every batter PA a strikeout, opponent 99 K), rebuild, and confirm
       those rows' features are unchanged.

    Returns a dict with per-check results and ``passed``.
    """
    base = build_features(raw, lineup_source)
    report: dict = {"features_checked": list(feature_cols)}

    report["current_game_features"] = [c for c in feature_cols if c in CURRENT_GAME_COLUMNS]

    first = base.sort_values(["pitcher_id", "game_date", "game_pk"]).groupby("pitcher_id").head(1)
    rolling_in_set = [c for c in feature_cols if c in ROLLING_FEATURES or c in PITCHER_CANDIDATES]
    report["unshifted_rolling_features"] = [c for c in rolling_in_set if first[c].notna().any()]

    rng = np.random.default_rng(seed)
    dates = np.sort(base["game_date"].unique())
    candidates = dates[1:] if len(dates) > 1 else dates
    picked = rng.choice(candidates, size=min(n_dates, len(candidates)), replace=False)
    changed: set[str] = set()
    keys = ["game_pk", "pitcher_id"]
    for day in picked:
        rebuilt = build_features(_perturb_date(raw, day), lineup_source)
        a = base[base["game_date"] == day].set_index(keys)[feature_cols].sort_index()
        b = rebuilt[rebuilt["game_date"] == day].set_index(keys)[feature_cols].sort_index()
        for col in feature_cols:
            if not np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True):
                changed.add(col)
    report["perturbed_dates"] = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in picked]
    report["features_changed_by_current_game"] = sorted(changed)

    report["passed"] = not (
        report["current_game_features"]
        or report["unshifted_rolling_features"]
        or report["features_changed_by_current_game"]
    )
    return report


def format_audit(report: dict, title: str = "Leakage audit") -> str:
    """Render an audit report as a short human-readable block."""
    def line(name, bad):
        return f"  [{'FAIL' if bad else 'PASS'}] {name}" + (f": {', '.join(bad)}" if bad else "")

    lines = [
        f"{title} ({len(report['features_checked'])} features)",
        line("no current-game stats used as features", report["current_game_features"]),
        line("rolling features shifted (NaN at first start)", report["unshifted_rolling_features"]),
        line(f"features unchanged when current game is altered ({len(report['perturbed_dates'])} dates)",
             report["features_changed_by_current_game"]),
        f"  => {'PASSED' if report['passed'] else 'LEAKAGE DETECTED'}",
    ]
    return "\n".join(lines)


def main() -> int:
    raw = load_raw()
    if raw["game_logs"].empty:
        print("No cached game logs. Run `python src/fetch_data.py` first.")
        return 1
    features = build_features(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)

    print(f"Wrote {len(features):,} starts to {FEATURES_PATH.relative_to(ROOT)}")
    print(features.groupby("season").size().rename("starts").to_string())
    print("\nMissing share per feature:")
    print(features[FEATURES].isna().mean().round(3).to_string())
    print()
    report = audit_leakage(raw, FEATURES)
    print(format_audit(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

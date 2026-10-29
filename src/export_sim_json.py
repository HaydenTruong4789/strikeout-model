"""Export upcoming-start predictions as JSON for the game simulation app (sim/index.html).

Reads ``data/processed/upcoming_predictions.parquet`` (written by
``src/predict_upcoming.py``) and adds each pitcher's throwing hand and pitch
mix from the cached raw Statcast pitches in ``data/raw/statcast``: usage,
mean and spread of velocity, and whiffs per swing for every pitch type thrown
in the last ``--mix-days`` days, plus his strike-three pitches over the same
window (type, speed, location and whether he got it looking) so the app can
place simulated strikeouts where his real ones went, chases included, and
his two-strike profile (pitch mix, out-of-zone rate, chase and whiff rates).
When the last year has fewer than 40 strikeouts, those use every cached season.

The model predicts strikeouts, not batters faced, so ``pred_bf`` is the
pitcher's average batters faced over his last five starts (``bf_last5``) and
``k_rate`` is ``pred_k / pred_bf``. A simulated start at that K rate over that
many batters averages the model's prediction. ``bf_history`` lists his batters
faced in his last 30 starts that season before the game (the season before too
when he has fewer than 15), so the app can draw short and long starts as often
as he really has them, recentered on ``pred_bf``.

Usage::

    python src/predict_upcoming.py
    python src/export_sim_json.py                 # writes sim/predictions.json
    python src/export_sim_json.py --out my.json --mix-days 180
    python src/export_sim_json.py --season 2026 --k-limit 60 --out sim/season_2026.json

With ``--season``, the export covers every pitcher with at least ``--min-starts``
starts that season instead of the upcoming parquet: the prediction for his latest
start comes from the blend trained on earlier seasons only (out of sample), and
the record carries the real strikeouts as ``actual_k``.

With ``--matchups``, every pitcher with ``--min-starts`` starts in the latest
season is predicted against each of the other 29 teams, as if he started at his
team's park (home) and at the opponent's park (away) the day after the last
cached game (weather and umpire unknown, the opponent's latest lineup). Each
prediction is split into factor groups: ``contribution_f = pred(x) - pred(x with
f at its training mean)``, summed per group in ``FACTOR_GROUPS``, plus a baseline
(every feature at its training mean) and an "Interactions" remainder, so the
parts add up to ``pred_k``. It writes ``sim/matchups_2026.json``:
``{"teams", "parks", "factors", "pitchers", "matchups": {pitcher_id: {opp: {"h": side, "a": side}}}}``
with ``side = {"k": pred_k, "p": [P(3+)..], "c": [group contributions.., interactions], "v": [opponent values]}``.

    python src/export_sim_json.py --matchups --k-limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import FEATURES, FEATURES_PATH, TARGET, build_features
from distribution import PROB_METHOD, THRESHOLDS, Distribution, prob_at_least
from fetch_data import (GAME_COLUMNS, LINEUP_COLUMNS, TEAM_GAME_COLUMNS, UPCOMING_COLUMNS, load_game_logs,
                        load_raw)
from fetch_data import STATCAST_RAW_DIR, STRIKEOUT_EVENTS, SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS
from predict_upcoming import UPCOMING_PATH
from train import make_blend

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "sim" / "predictions.json"
MATCHUPS_OUT = ROOT / "sim" / "matchups_2026.json"
TEAM_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN", 114: "CLE", 115: "COL",
    116: "DET", 117: "HOU", 118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH", 134: "PIT",
    135: "SD", 136: "SEA", 137: "SF", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}
TEAM_NAMES = {
    "ARI": "Arizona Diamondbacks", "ATH": "Athletics", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "CWS": "Chicago White Sox", "DET": "Detroit Tigers", "HOU": "Houston Astros",
    "KC": "Kansas City Royals", "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets", "NYY": "New York Yankees",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}
# Factor groups for the "What moved the prediction" breakdown, in display order.
FACTOR_GROUPS = {
    "home": ("Home/away", ["is_home"]),
    "park": ("Park", ["park_k_factor", "roof_closed"]),
    "weather": ("Weather", ["wind_out_mph", "day_game"]),
    "ump": ("Umpire", ["ump_k_factor"]),
    "opp_k": ("Opponent lineup K rate", ["opp_k_rate_last30", "lineup_k_rate_vs_hand"]),
    "whiffs": ("Lineup whiffs by pitch type", ["lineup_whiff_fb_hi", "lineup_whiff_brk", "lineup_whiff_os",
                                               "matchup_whiff_rate"]),
    "platoon": ("Platoon", ["lineup_platoon_adv_share", "is_lhp"]),
    "skill": ("His strikeout skill", ["season_k_rate", "season_k_per_start", "k_rate_ewm", "k_per_start_ewm",
                                      "k_per_bf_last5", "pitcher_k_rate_vs_lineup"]),
    "stuff": ("Stuff and form", ["fb_velo_last3", "fb_velo_last5", "whiff_rate_last3", "whiff_rate_last5",
                                 "usage_fb_hi_last5", "usage_brk_last5", "usage_os_last5"]),
    "workload": ("Workload", ["bf_last5", "pitches_last5"]),
    "rest": ("Rest/schedule", ["rest_days", "first_start_of_season", "n_prior_starts"]),
}
# Pitcher-level and opponent-level feature values the app uses to describe each factor.
PITCHER_VALUES = ["season_k_rate", "k_rate_ewm", "k_per_bf_last5", "season_k_per_start", "fb_velo_last5",
                  "whiff_rate_last5", "bf_last5", "pitches_last5", "rest_days", "n_prior_starts",
                  "first_start_of_season", "is_lhp"]
MATCHUP_VALUES = ["lineup_k_rate_vs_hand", "opp_k_rate_last30", "matchup_whiff_rate", "lineup_platoon_adv_share",
                  "park_k_factor", "roof_closed", "wind_out_mph", "day_game", "ump_k_factor"]
MIN_USAGE = 0.02  # pitch types thrown less often than this are dropped
BF_HISTORY_N, BF_HISTORY_MIN = 30, 15  # starts kept for the batters-faced draw; below the min, add last season


def start_logs() -> dict[int, pd.DataFrame]:
    """Each pitcher's starts (date, season, batters faced), oldest first."""
    logs = load_game_logs()
    logs = logs[logs["started"] == 1][["pitcher_id", "game_date", "season", "batters_faced"]].copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs["batters_faced"] = pd.to_numeric(logs["batters_faced"], errors="coerce")
    logs = logs.dropna(subset=["batters_faced"]).sort_values("game_date")
    return {int(k): g for k, g in logs.groupby("pitcher_id")}


def bf_history(starts: pd.DataFrame | None, before: pd.Timestamp, season: int) -> list[int]:
    """Batters faced in his last 30 starts of ``season`` before ``before``; the season before too if under 15."""
    if starts is None:
        return []
    prior = starts[starts["game_date"] < before]
    cur = prior[prior["season"] == season]
    if len(cur) < BF_HISTORY_MIN:
        cur = prior[prior["season"] >= season - 1]
    return [int(x) for x in cur["batters_faced"].tail(BF_HISTORY_N)]


def all_pitches(pitcher_ids: set[int]) -> pd.DataFrame:
    """Every cached raw Statcast pitch by these pitchers (only the columns the export needs)."""
    cols = ["pitcher", "game_date", "p_throws", "pitch_name", "release_speed", "description",
            "events", "plate_x", "plate_z", "sz_top", "sz_bot", "strikes", "zone"]
    frames = []
    for path in sorted(STATCAST_RAW_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(path, columns=cols)
        except Exception:  # noqa: BLE001 - an older chunk missing a column
            df = pd.read_parquet(path).reindex(columns=cols)
        frames.append(df[df["pitcher"].isin(pitcher_ids)])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True).sort_values("game_date")


def two_strike_profile(pitches: pd.DataFrame) -> dict | None:
    """What he throws with two strikes and how batters respond: his own numbers."""
    two = pitches[pd.to_numeric(pitches["strikes"], errors="coerce") == 2].dropna(subset=["zone"])
    if len(two) < 100:
        return None
    oz = pd.to_numeric(two["zone"], errors="coerce").between(11, 14)
    swing = two["description"].isin(SWING_DESCRIPTIONS)
    whiff = two["description"].isin(WHIFF_DESCRIPTIONS)
    ks = pitches[pitches["events"].isin(STRIKEOUT_EVENTS)]
    mix = two["pitch_name"].value_counts(normalize=True)
    return {
        "pitches": int(len(two)),
        "mix": {k: round(float(v), 3) for k, v in mix[mix >= 0.02].items()},
        "out_of_zone": round(float(oz.mean()), 3),
        "chase": round(float(swing[oz].mean()), 3),
        "whiff_per_chase": round(float(whiff[oz & swing].mean()), 3) if (oz & swing).any() else None,
        "looking_k_share": round(float((ks["description"] == "called_strike").mean()), 3) if len(ks) else None,
        "strikeouts": int(len(ks)),
    }


def pitch_mix(pitches: pd.DataFrame) -> list[dict]:
    """Usage, velocity mean and SD, and whiffs per swing for each pitch type."""
    pitches = pitches.dropna(subset=["pitch_name"])
    if pitches.empty:
        return []
    swing = pitches["description"].isin(SWING_DESCRIPTIONS)
    whiff = pitches["description"].isin(WHIFF_DESCRIPTIONS)
    g = pitches.assign(swing=swing, whiff=whiff).groupby("pitch_name")
    mix = pd.DataFrame({
        "n": g.size(),
        "velo_mean": g["release_speed"].mean(),
        "velo_sd": g["release_speed"].std(),
        "swings": g["swing"].sum(),
        "whiffs": g["whiff"].sum(),
    })
    mix["usage"] = mix["n"] / mix["n"].sum()
    mix = mix[mix["usage"] >= MIN_USAGE].sort_values("usage", ascending=False)
    mix["usage"] /= mix["usage"].sum()
    return [
        {
            "type": name,
            "usage": round(float(r.usage), 3),
            "velo_mean": round(float(r.velo_mean), 1),
            "velo_sd": round(float(np.nan_to_num(r.velo_sd, nan=1.0)), 2),
            "whiff_rate": round(float(r.whiffs / r.swings), 3) if r.swings else None,
        }
        for name, r in mix.iterrows()
    ]


def k_pitches(pitches: pd.DataFrame, limit: int = 120) -> list[dict]:
    """Strike-three pitches: type, speed, location (catcher's view, feet) and looking or not.

    Height is rescaled from each batter's own zone (sz_bot..sz_top) to the app's
    standard 1.6-3.5 ft zone, so chases stay chases whoever was hitting.
    """
    k = pitches[pitches["events"].isin(STRIKEOUT_EVENTS)].dropna(subset=["plate_x", "plate_z", "pitch_name"])
    if k.empty:
        return []
    k = k.tail(limit)
    bot, top = k["sz_bot"].fillna(1.6), k["sz_top"].fillna(3.5)
    pz = 1.6 + (k["plate_z"] - bot) / (top - bot).where(top > bot, 1.9) * 1.9
    return [
        {"type": r.pitch_name, "velo": round(float(r.release_speed), 1) if pd.notna(r.release_speed) else None,
         "px": round(float(r.plate_x), 2), "pz": round(float(z), 2), "looking": r.description == "called_strike"}
        for r, z in zip(k.itertuples(), pz)
    ]


def season_predictions(season: int, min_starts: int = 5) -> pd.DataFrame:
    """Each pitcher's latest start of ``season``, predicted by the blend fit on earlier seasons."""
    df = pd.read_parquet(FEATURES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    train, test = df[df["season"] < season], df[df["season"] == season].sort_values(["game_date", "game_pk"])
    if test.empty:
        raise RuntimeError(f"No {season} starts in {FEATURES_PATH}.")
    model = make_blend().fit(train[FEATURES], train[TARGET])
    test = test.assign(pred_model=np.clip(model.predict(test[FEATURES]), 0, None))
    # Probabilities: spread fit on the season before, predicted the same out-of-sample way.
    early, prev = train[train["season"] < season - 1], train[train["season"] == season - 1]
    mu_prev = np.clip(make_blend().fit(early[FEATURES], early[TARGET]).predict(prev[FEATURES]), 0, None)
    n = test.groupby("pitcher_id")["game_pk"].transform("size")
    last = test[n >= min_starts].groupby("pitcher_id").tail(1).copy()
    pmf = Distribution(PROB_METHOD).fit(prev[TARGET], mu_prev).pmf(last["pred_model"])
    for t in THRESHOLDS:
        last[f"p_{t}plus"] = prob_at_least(pmf, t)
    return last.rename(columns={TARGET: "actual_strikeouts"}).reset_index(drop=True)


def factor_groups(features: list[str]) -> dict[str, list[str]]:
    """``FACTOR_GROUPS`` columns, with any feature not listed added to the group sharing its name prefix."""
    groups = {k: [f for f in cols if f in features] for k, (_, cols) in FACTOR_GROUPS.items()}
    listed = {f for cols in groups.values() for f in cols}
    for f in features:
        if f not in listed:
            key = next((k for k, cols in groups.items() if any(c.split("_")[0] == f.split("_")[0] for c in cols)),
                       "stuff")
            groups[key].append(f)
    return groups


def contributions(model, X: pd.DataFrame, means: pd.Series) -> tuple[float, pd.DataFrame]:
    """Baseline (every feature at its mean) and each group's summed ``pred(x) - pred(x, f at mean)``."""
    base = float(model.predict(means.to_frame().T[X.columns])[0])
    full = model.predict(X)
    out = {}
    for key, cols in factor_groups(list(X.columns)).items():
        total = np.zeros(len(X))
        for f in cols:
            total += full - model.predict(X.assign(**{f: means[f]}))
        out[key] = total
    return base, pd.DataFrame(out, index=X.index)


def matchup_predictions(min_starts: int = 5) -> tuple[pd.DataFrame, dict]:
    """Every pitcher with ``min_starts`` starts in the latest season against each other team, home and away.

    Each pitcher's start is placed the day after the last cached game, at his
    team's park (``side == "h"``) or the opponent's (``side == "a"``), through
    the same feature builder and blend as ``predict_upcoming``. The builder keeps
    one upcoming start per pitcher, so it runs once per round and side, with each
    pitcher facing a different team each round. Also returns the parks, the
    training means and the baseline used for the factor breakdown.
    """
    raw = load_raw()
    logs = raw["game_logs"].copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    season = int(logs["season"].max())
    day = logs["game_date"].max() + timedelta(days=1)
    cur = logs[(logs["season"] == season) & (logs["started"] == 1)].sort_values(["game_date", "game_pk"])
    n = cur.groupby("pitcher_id")["game_pk"].transform("size")
    pitchers = cur[n >= min_starts].groupby("pitcher_id").tail(1)[["pitcher_id", "pitcher_name", "team_id"]]
    teams = sorted(int(t) for t in TEAM_ABBR)

    # Each team's home park (its most used home venue this season, so a neutral-site
    # series doesn't count) and whether it's a dome.
    games = raw["games"].copy()
    home = logs[(logs["is_home"] == 1) & (logs["season"] == season)][["game_pk", "team_id", "game_date"]]
    home = home.drop_duplicates("game_pk").merge(games[["game_pk", "venue_id", "venue_name", "condition"]], on="game_pk")
    usual = home.groupby("team_id")["venue_id"].agg(lambda v: v.mode().iloc[0])
    home = home[home["venue_id"] == home["team_id"].map(usual)]
    park = home.sort_values("game_date").groupby("team_id").tail(1).set_index("team_id")
    dome = games.groupby("venue_id")["condition"].agg(lambda c: (c == "Dome").mean() > 0.5)

    history = None
    rows = []
    for side in ("h", "a"):
        for rnd in range(len(teams) - 1):
            starts, gms, tgs = [], [], []
            for k, r in enumerate(pitchers.itertuples()):
                ti = teams.index(int(r.team_id))
                opp = teams[(ti + 1 + rnd) % len(teams)]
                pk = 900_000_000 + (side == "a") * 1_000_000 + rnd * 10_000 + k
                host = int(r.team_id) if side == "h" else opp
                v = park.loc[host] if host in park.index else None
                date = day.strftime("%Y-%m-%d")
                starts.append({"game_pk": pk, "game_date": date, "season": season, "pitcher_id": int(r.pitcher_id),
                               "pitcher_name": r.pitcher_name, "team_id": int(r.team_id), "opponent_id": opp,
                               "is_home": int(side == "h"), "game_type": "R", "status": "Scheduled",
                               "actual_strikeouts": None})
                g = {c: None for c in GAME_COLUMNS}
                g.update({"game_pk": pk, "game_date": date, "season": season, "day_night": "night"})
                if v is not None:
                    g.update({"venue_id": v["venue_id"], "venue_name": v["venue_name"],
                              "condition": "Dome" if dome.get(v["venue_id"], False) else None})
                gms.append(g)
                for t, o in ((int(r.team_id), opp), (opp, int(r.team_id))):
                    tgs.append({"game_pk": pk, "game_date": date, "season": season, "team_id": t, "opponent_id": o,
                                "bat_pa": np.nan, "bat_k": np.nan})
            upcoming = {"starts": pd.DataFrame(starts, columns=UPCOMING_COLUMNS),
                        "games": pd.DataFrame(gms, columns=GAME_COLUMNS),
                        "lineups": pd.DataFrame(columns=LINEUP_COLUMNS),
                        "team_games": pd.DataFrame(tgs, columns=TEAM_GAME_COLUMNS)}
            feats = build_features(raw, upcoming=upcoming)
            if history is None:
                history = feats[feats["upcoming"] == 0]
            rows.append(feats[feats["upcoming"] == 1].assign(side=side))
            print(f"  {'home' if side == 'h' else 'away'} round {rnd + 1}/{len(teams) - 1}", file=sys.stderr,
                  flush=True)
    nxt = pd.concat(rows, ignore_index=True)

    model = make_blend().fit(history[FEATURES], history[TARGET])
    nxt["pred_model"] = np.clip(model.predict(nxt[FEATURES]), 0, None)
    means = history[FEATURES].mean()
    baseline, contrib = contributions(model, nxt[FEATURES], means)
    nxt = pd.concat([nxt, contrib.add_prefix("c_")], axis=1)
    latest = history["season"].max()
    early, prev = history[history["season"] < latest], history[history["season"] == latest]
    mu_prev = np.clip(make_blend().fit(early[FEATURES], early[TARGET]).predict(prev[FEATURES]), 0, None)
    pmf = Distribution(PROB_METHOD).fit(prev[TARGET], mu_prev).pmf(nxt["pred_model"])
    for t in THRESHOLDS:
        nxt[f"p_{t}plus"] = prob_at_least(pmf, t)

    # Each park's K factor and roof as the builder computed them for the home rows.
    host = nxt[nxt["side"] == "h"].groupby("team_id")[["park_k_factor", "roof_closed"]].first()
    parks = {TEAM_ABBR[int(t)]: {"name": park.loc[int(t), "venue_name"] if int(t) in park.index else None,
                                  "k": round(float(host.loc[t, "park_k_factor"]), 3)
                                  if t in host.index and pd.notna(host.loc[t, "park_k_factor"]) else None,
                                  "roof": int(host.loc[t, "roof_closed"]) if t in host.index else 0}
             for t in teams}
    meta = {"parks": parks, "means": means, "baseline": baseline}
    return nxt.reset_index(drop=True), meta


def rounded(x, nd: int = 3):
    return round(float(x), nd) if pd.notna(x) else None


def export_matchups(mix_days: int = 365, min_starts: int = 5, k_limit: int = 120) -> dict:
    """The pitcher profiles plus each pitcher's prediction against every other team, home and away."""
    preds, meta = matchup_predictions(min_starts)
    prob_cols = [c for c in preds.columns if c.startswith("p_") and c.endswith("plus")]
    pitches = all_pitches(set(preds["pitcher_id"].astype(int)))
    by_pitcher = dict(tuple(pitches.groupby("pitcher")))
    latest = pd.to_datetime(pitches["game_date"]).max()
    logs = start_logs()
    season = int(preds["season"].max())
    groups = list(FACTOR_GROUPS)
    baseline = round(meta["baseline"], 2)

    def side(m) -> dict:
        k = round(float(m["pred_model"]), 2)
        c = [round(float(m[f"c_{g}"]), 2) for g in groups]
        return {"k": k, "p": [round(float(m[c_]), 3) for c_ in prob_cols],
                "c": c + [round(k - baseline - sum(c), 2)],
                "v": [rounded(m[f]) for f in MATCHUP_VALUES]}

    profiles, matchups = [], {}
    for pid, g in preds.groupby("pitcher_id"):
        r = g.iloc[0]
        career = by_pitcher.get(int(pid), pitches.iloc[0:0])
        own = career[pd.to_datetime(career["game_date"]) >= latest - timedelta(days=mix_days)]
        k_src = own if own["events"].isin(STRIKEOUT_EVENTS).sum() >= 40 else career
        hand = own["p_throws"].mode()
        profiles.append({
            "pitcher": r["pitcher_name"],
            "pitcher_id": int(pid),
            "hand": hand.iloc[0] if len(hand) else "R",
            "team": TEAM_ABBR.get(int(r["team_id"]), str(r["team_id"])),
            "pred_bf": round(float(r["bf_last5"]), 1) if pd.notna(r["bf_last5"]) else 22.0,
            "bf_history": bf_history(logs.get(int(pid)), latest + timedelta(days=1), season),
            "values": [rounded(r[f]) for f in PITCHER_VALUES],
            "pitch_mix": pitch_mix(own),
            "k_pitches": k_pitches(k_src, k_limit),
            "two_strike": two_strike_profile(k_src),
        })
        out = {}
        for _, m in g.iterrows():
            opp = TEAM_ABBR.get(int(m["opponent_id"]), str(m["opponent_id"]))
            out.setdefault(opp, {})[m["side"]] = side(m)
        matchups[str(int(pid))] = out
    profiles.sort(key=lambda p: p["pitcher"])
    factors = {
        "groups": [[k, FACTOR_GROUPS[k][0]] for k in groups] + [["interactions", "Interactions"]],
        "baseline": baseline,
        "thresholds": [c[2:-4] + "+" for c in prob_cols],
        "pitcher_values": PITCHER_VALUES,
        "matchup_values": MATCHUP_VALUES,
        "league": {f: rounded(v, 4) for f, v in meta["means"].items()},
    }
    return {"teams": TEAM_NAMES, "parks": meta["parks"], "factors": factors, "pitchers": profiles,
            "matchups": matchups}


def export(mix_days: int = 365, season: int | None = None, min_starts: int = 5, k_limit: int = 120) -> list[dict]:
    """One JSON-ready record per upcoming start, or per pitcher's latest start of ``season``."""
    if season is not None:
        preds = season_predictions(season, min_starts)
    else:
        if not UPCOMING_PATH.exists():
            raise RuntimeError("No upcoming predictions. Run `python src/predict_upcoming.py` first.")
        preds = pd.read_parquet(UPCOMING_PATH)
    prob_cols = [c for c in preds.columns if c.startswith("p_") and c.endswith("plus")]
    pitches = all_pitches(set(preds["pitcher_id"].astype(int)))
    by_pitcher = dict(tuple(pitches.groupby("pitcher")))
    logs = start_logs()

    out = []
    for r in preds.itertuples():
        career = by_pitcher.get(int(r.pitcher_id), pitches.iloc[0:0])
        # Only pitches from before this start.
        day = pd.Timestamp(r.game_date)
        career = career[pd.to_datetime(career["game_date"]) < day]
        own = career[pd.to_datetime(career["game_date"]) >= day - timedelta(days=mix_days)]
        # Strikeout and two-strike numbers: the last year, or all cached seasons if that's thin.
        k_src = own if own["events"].isin(STRIKEOUT_EVENTS).sum() >= 40 else career
        pred_k = float(r.pred_model)
        pred_bf = float(r.bf_last5) if pd.notna(r.bf_last5) else 22.0
        hand = own["p_throws"].mode()
        record = {
            "pitcher": r.pitcher_name,
            "pitcher_id": int(r.pitcher_id),
            "hand": hand.iloc[0] if len(hand) else "R",
            "team": TEAM_ABBR.get(int(r.team_id), str(r.team_id)),
            "opponent": TEAM_ABBR.get(int(r.opponent_id), str(r.opponent_id)),
            "game_date": pd.Timestamp(r.game_date).strftime("%Y-%m-%d"),
            "pred_k": round(pred_k, 2),
            "pred_bf": round(pred_bf, 1),
            "bf_history": bf_history(logs.get(int(r.pitcher_id)), day, day.year),
            "k_rate": round(pred_k / pred_bf, 3) if pred_bf else None,
            "probs": {c[2:-4] + "+": round(float(getattr(r, c)), 3)
                      for c in prob_cols if pd.notna(getattr(r, c))},
            "pitch_mix": pitch_mix(own),
            "k_pitches": k_pitches(k_src, k_limit),
            "two_strike": two_strike_profile(k_src),
        }
        if pd.notna(getattr(r, "actual_strikeouts", np.nan)):
            record["actual_k"] = int(r.actual_strikeouts)
        out.append(record)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON file to write")
    parser.add_argument("--mix-days", type=int, default=365, help="days of pitches used for the pitch mix")
    parser.add_argument("--season", type=int, help="export every pitcher's latest start of this season instead")
    parser.add_argument("--min-starts", type=int, default=5, help="with --season, starts a pitcher needs")
    parser.add_argument("--k-limit", type=int, default=120, help="strike-three pitches kept per pitcher")
    parser.add_argument("--matchups", action="store_true",
                        help="predict each pitcher against every other team (writes sim/matchups_2026.json)")
    args = parser.parse_args(argv)

    if args.matchups:
        out = MATCHUPS_OUT if args.out == DEFAULT_OUT else args.out
        data = export_matchups(args.mix_days, args.min_starts, args.k_limit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, separators=(",", ":")))
        print(f"Wrote {len(data['pitchers'])} pitchers x their opponents to {out}")
        for p in data["pitchers"][:: max(1, len(data["pitchers"]) // 3)][:3]:
            m = {t: s["h"] for t, s in data["matchups"][str(p["pitcher_id"])].items()}
            lo, hi = min(m, key=lambda t: m[t]["k"]), max(m, key=lambda t: m[t]["k"])
            print(f"  {p['pitcher']} ({p['team']}, home): {m[lo]['k']:.2f} vs {lo} to {m[hi]['k']:.2f} vs {hi}")
        return 0

    records = export(args.mix_days, args.season, args.min_starts, args.k_limit)
    records.sort(key=lambda r: (r["team"], -r["pred_k"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.season is not None:
        args.out.write_text(json.dumps(records, separators=(",", ":")))
    else:
        args.out.write_text(json.dumps(records, indent=2))
    no_mix = [r["pitcher"] for r in records if not r["pitch_mix"]]
    print(f"Wrote {len(records)} pitchers to {args.out}")
    if no_mix:
        print("No cached Statcast pitches (app will use a generic mix) for: " + ", ".join(no_mix))
    return 0


if __name__ == "__main__":
    sys.exit(main())

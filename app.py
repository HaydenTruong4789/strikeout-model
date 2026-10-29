"""Simple web app for the strikeout model.

The first tab predicts upcoming starts: the schedule's probable pitchers for
the next few days, from the linear + gradient-boosting blend refit on every
completed start.
The second tab lets you pick a season, a pitcher and one of their starts to
see what the model predicted before the game, next to what actually
happened. A third tab shows the model's overall test results.

Predictions are honest walk-forward ones: every season from 2022 on is
predicted by the linear + gradient-boosting blend fit only on the seasons
before it. 2021 has no earlier season to
learn from, so it is not offered.

Needs ``data/processed/features.parquet`` (run ``./run_pipeline.sh`` first).

Usage::

    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from build_features import FEATURES, FEATURES_PATH, TARGET  # noqa: E402
from predict_upcoming import predict_upcoming  # noqa: E402
from train import baseline_predict, make_blend  # noqa: E402

METRICS_PATH = ROOT / "metrics" / "latest.json"
FIG_DIR = ROOT / "reports" / "figures"

TEAMS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN", 114: "CLE", 115: "COL",
    116: "DET", 117: "HOU", 118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH", 134: "PIT",
    135: "SD", 136: "SEA", 137: "SF", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Inputs shown under a prediction: (column, label, format).
KEY_INPUTS = [
    ("season_k_per_start", "Season K per start so far", "{:.2f}"),
    ("season_k_rate", "Season K rate so far", "{:.1%}"),
    ("k_per_bf_last5", "K rate, last 5 starts", "{:.1%}"),
    ("bf_last5", "Batters faced, last 5 starts (avg)", "{:.1f}"),
    ("pitches_last5", "Pitches, last 5 starts (avg)", "{:.0f}"),
    ("whiff_rate_last5", "Whiff rate, last 5 starts", "{:.1%}"),
    ("fb_velo_last5", "Fastball velo, last 5 starts", "{:.1f} mph"),
    ("rest_days", "Days of rest", "{:.0f}"),
    ("opp_k_rate_last30", "Opponent K rate, last 30 games", "{:.1%}"),
    ("pitcher_k_rate_vs_lineup", "Pitcher K rate vs this lineup's hands", "{:.1%}"),
    ("lineup_k_rate_vs_hand", "Lineup K rate vs this pitcher's hand", "{:.1%}"),
    ("park_k_factor", "Park K factor", "{:.2f}"),
    ("ump_k_factor", "Umpire K factor", "{:.2f}"),
]


def team(team_id) -> str:
    return TEAMS.get(int(team_id), str(team_id)) if pd.notna(team_id) else "?"


def opponents(df: pd.DataFrame) -> list[str]:
    """Opponent labels: "vs NYY" at home, "@ NYY" on the road."""
    return [("vs " if h == 1 else "@ ") + team(o) for h, o in zip(df["is_home"], df["opponent_id"], strict=True)]


@st.cache_data(show_spinner="Fitting the model season by season...")
def load_predictions() -> pd.DataFrame:
    """Every start from the second season on, with walk-forward model and baseline predictions."""
    df = pd.read_parquet(FEATURES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "pitcher_id"]).reset_index(drop=True)
    parts = []
    for season in sorted(df["season"].unique())[1:]:
        train = df[df["season"] < season]
        test = df[df["season"] == season].copy()
        model = make_blend().fit(train[FEATURES], train[TARGET])
        test["pred_model"] = np.clip(model.predict(test[FEATURES]), 0, None)
        test["pred_baseline"] = baseline_predict(test, float(train[TARGET].mean()))
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


@st.cache_data
def league_medians() -> pd.Series:
    return pd.read_parquet(FEATURES_PATH, columns=[c for c, _, _ in KEY_INPUTS]).median()


def fmt(value, pattern: str) -> str:
    return "n/a" if pd.isna(value) else pattern.format(value)


def game_label(row) -> str:
    where = "vs" if row["is_home"] == 1 else "@"
    return f"{row['game_date']:%b %d} {where} {team(row['opponent_id'])}"


@st.cache_data(ttl=3600, show_spinner="Fetching probable pitchers and building their features...")
def load_upcoming(days: int) -> pd.DataFrame:
    return predict_upcoming(days)


def upcoming_tab(preds: pd.DataFrame) -> None:
    c1, c2 = st.columns([1, 3])
    days = c1.selectbox("Look ahead", [1, 3, 7], index=1, format_func=lambda d: f"{d} day{'s' * (d > 1)}")
    if c2.button("Refresh schedule"):
        load_upcoming.clear()
    try:
        up = load_upcoming(days)
    except Exception as exc:  # noqa: BLE001 - shown to the user instead of a traceback
        st.error(f"Couldn't fetch the schedule: {exc}")
        return
    if up.empty:
        st.info("No probable starters are posted for these days.")
        return
    st.caption(f"Model fit on every start through {up['trained_through'].iloc[0]:%B %d, %Y}. "
               "Lineups not posted yet are taken from each opponent's previous game. "
               "Run `python src/fetch_data.py --incremental` to bring the data up to date.")

    # Typical miss, from the walk-forward predictions of the latest season.
    latest = preds[preds["season"] == preds["season"].max()]
    resid = latest[TARGET] - latest["pred_model"]
    lo, hi = resid.quantile([0.1, 0.9])

    def label(r) -> str:
        return (f"{r.pitcher_name} ({team(r.team_id)}), {r.game_date:%b %d} "
                f"{'vs' if r.is_home == 1 else '@'} {team(r.opponent_id)}")

    options = list(up.sort_values(["game_date", "pred_model"], ascending=[True, False]).index)
    choice = st.selectbox("Pitcher", options, format_func=lambda i: label(up.loc[i]))
    g = up.loc[choice]

    st.subheader(label(g))
    m1, m2, m3 = st.columns(3)
    m1.metric("Model prediction", f"{g['pred_model']:.1f} K")
    m2.metric("Likely range", f"{max(0, round(g['pred_model'] + lo))}-{round(g['pred_model'] + hi)} K",
              help="Middle 80% of the model's misses last season, added to this prediction")
    m3.metric("Naive baseline", f"{g['pred_baseline']:.1f} K", help="Season-to-date K per start before this game")
    if pd.notna(g["actual_strikeouts"]):
        st.success(f"Final: {int(g['actual_strikeouts'])} strikeouts.")
    elif g["status"] not in ("Scheduled", "Pre-Game", "Warmup"):
        st.caption(f"Game status: {g['status']}. The prediction uses only what was known before first pitch.")

    if "p_3plus" in g.index:
        st.markdown("**Chance of at least...**")
        chances = {f"{t}+ K": f"{g[f'p_{t}plus']:.0%}" for t in range(3, 10) if f"p_{t}plus" in g.index}
        st.dataframe(pd.DataFrame([chances]), hide_index=True, width="stretch")

    with st.expander("What the model knows going in", expanded=True):
        med = league_medians()
        rows = [{"Input": label_, "This start": fmt(g[col], pattern), "League median": fmt(med[col], pattern)}
                for col, label_, pattern in KEY_INPUTS if col in g.index]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    past = preds[preds["pitcher_id"] == g["pitcher_id"]].sort_values("game_date").tail(10)
    if not past.empty:
        st.markdown(f"**{g['pitcher_name']}'s last {len(past)} starts**")
        st.dataframe(pd.DataFrame({
            "Date": past["game_date"].dt.strftime("%Y-%m-%d"),
            "Opponent": opponents(past),
            "Predicted": past["pred_model"].round(1),
            "Actual": past[TARGET].astype(int),
            "Batters faced": past["batters_faced"],
        }).iloc[::-1], hide_index=True, width="stretch")

    st.markdown("**All probable starters**")
    st.dataframe(pd.DataFrame({
        "Date": up["game_date"].dt.strftime("%Y-%m-%d"),
        "Pitcher": up["pitcher_name"],
        "Team": up["team_id"].map(team),
        "Opponent": opponents(up),
        "Predicted K": up["pred_model"].round(1),
        "Chance of 6+ K": up["p_6plus"].map("{:.0%}".format) if "p_6plus" in up else None,
        "Baseline K": up["pred_baseline"].round(1),
        "Status": up["status"],
        "Actual K": up["actual_strikeouts"],
    }), hide_index=True, width="stretch")


def lookup_tab(preds: pd.DataFrame) -> None:
    seasons = sorted(preds["season"].unique(), reverse=True)
    c1, c2 = st.columns([1, 3])
    season = c1.selectbox("Season", seasons)
    in_season = preds[preds["season"] == season]

    starts = in_season.groupby(["pitcher_id", "pitcher_name"]).size().reset_index(name="n")
    starts = starts.sort_values("pitcher_name")
    labels = {r.pitcher_id: f"{r.pitcher_name} ({r.n} starts)" for r in starts.itertuples()}
    pitcher_id = c2.selectbox("Pitcher", list(labels), format_func=labels.get)

    games = in_season[in_season["pitcher_id"] == pitcher_id].sort_values("game_date").reset_index(drop=True)
    game_idx = st.selectbox("Start", games.index[::-1], format_func=lambda i: game_label(games.loc[i]))
    g = games.loc[game_idx]

    st.subheader(f"{g['pitcher_name']}, {g['game_date']:%B %d, %Y} "
                 f"{'vs' if g['is_home'] == 1 else '@'} {team(g['opponent_id'])}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model prediction", f"{g['pred_model']:.1f} K")
    m2.metric("Actual", f"{int(g[TARGET])} K", f"{g[TARGET] - g['pred_model']:+.1f} vs prediction",
              delta_color="off")
    m3.metric("Naive baseline", f"{g['pred_baseline']:.1f} K", help="Season-to-date K per start before this game")
    st.caption("The model's typical miss is about 1.8 strikeouts either way; much of a single game "
               "is luck and how long the manager leaves the starter in.")

    with st.expander("What the model knew before this game"):
        med = league_medians()
        rows = [{"Input": label, "This start": fmt(g[col], pattern), "League median": fmt(med[col], pattern)}
                for col, label, pattern in KEY_INPUTS if col in g.index]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown(f"**{g['pitcher_name']}'s {season} season**")
    chart = games.set_index("game_date")[[TARGET, "pred_model"]].rename(
        columns={TARGET: "Actual", "pred_model": "Predicted"})
    st.line_chart(chart, y_label="Strikeouts")
    mae = (games[TARGET] - games["pred_model"]).abs().mean()
    st.caption(f"Average miss this season: {mae:.2f} K over {len(games)} starts.")

    table = pd.DataFrame({
        "Date": games["game_date"].dt.strftime("%Y-%m-%d"),
        "Opponent": opponents(games),
        "Predicted": games["pred_model"].round(1),
        "Actual": games[TARGET].astype(int),
        "Baseline": games["pred_baseline"].round(1),
        "Batters faced": games["batters_faced"],
    })
    st.dataframe(table.iloc[::-1], hide_index=True, width="stretch")


def results_tab(preds: pd.DataFrame) -> None:
    if METRICS_PATH.exists():
        result = json.loads(METRICS_PATH.read_text())
        st.markdown(f"Trained on {result['train_seasons'][0]}-{result['train_seasons'][-1]} "
                    f"({result['n_train']:,} starts), tested on {result['test_season']} "
                    f"({result['n_test']:,} starts).")
        names = {"baseline": "Naive baseline", "linear": "Linear regression", "random_forest": "Random forest"}
        rows = [{"Model": names.get(k, k), "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"],
                 "MAE better than baseline": f"{m['mae_improvement_pct']:+.1f}%"}
                for k, m in result["metrics"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No metrics/latest.json yet. Run `python src/train.py`.")

    st.markdown("**This app's predictions, by season** (each season predicted from the seasons before it)")
    by_season = preds.groupby("season").apply(lambda d: pd.Series({
        "Starts": len(d),
        "Model MAE": (d[TARGET] - d["pred_model"]).abs().mean(),
        "Baseline MAE": (d[TARGET] - d["pred_baseline"]).abs().mean(),
    }), include_groups=False).reset_index()
    by_season["Starts"] = by_season["Starts"].astype(int)
    st.dataframe(by_season.round(3), hide_index=True, width="stretch")

    figures = [("predicted_vs_actual.png", "Predicted vs actual"), ("residuals.png", "Residuals"),
               ("feature_importance.png", "Feature importance")]
    for name, caption in figures:
        if (FIG_DIR / name).exists():
            st.image(str(FIG_DIR / name), caption=caption)


def main() -> None:
    st.set_page_config(page_title="Strikeout model", page_icon="⚾", layout="wide")
    st.title("⚾ Starting pitcher strikeout predictions")
    if not FEATURES_PATH.exists():
        st.error("No feature table yet. Run `./run_pipeline.sh` first.")
        st.stop()
    preds = load_predictions()
    upcoming, lookup, results = st.tabs(["Upcoming starts", "Look up a start", "Model results"])
    with upcoming:
        upcoming_tab(preds)
    with lookup:
        lookup_tab(preds)
    with results:
        results_tab(preds)


main()

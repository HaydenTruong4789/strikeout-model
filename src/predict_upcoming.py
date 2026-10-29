"""Predict strikeouts for upcoming starts (the schedule's probable pitchers).

Fetches the probable starters for the next few days from the MLB schedule,
builds their pre-game features from everything cached in ``data/raw``, and
predicts with the linear + gradient-boosting blend refit on every completed
start (the best option in ``src/compare_models.py``). An opponent with no posted lineup yet
is given its lineup from its previous game.

Each prediction also carries the chance of 3+ through 9+ strikeouts
(``p_3plus`` ... ``p_9plus``, see ``src/distribution.py``).

A game that is already under way or finished today, but not yet in the cache,
is still predicted from pre-game information only, with its actual strikeouts
shown once the game is final.

Run ``python src/fetch_data.py --incremental`` first so the cache is current.

Usage::

    python src/predict_upcoming.py            # today and the next two days
    python src/predict_upcoming.py --days 7
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import FEATURES, PROCESSED_DIR, TARGET, build_features
from distribution import PROB_METHOD, THRESHOLDS, Distribution, prob_at_least
from fetch_data import fetch_upcoming, load_raw, make_session
from train import baseline_predict, make_blend

UPCOMING_PATH = PROCESSED_DIR / "upcoming_predictions.parquet"
ROOT = Path(__file__).resolve().parents[1]


def predict_upcoming(days: int = 3, today: date | None = None) -> pd.DataFrame:
    """Return one row per upcoming probable start with its features and predictions."""
    raw = load_raw()
    if raw["game_logs"].empty:
        raise RuntimeError("No cached game logs. Run `python src/fetch_data.py` first.")
    known = set(raw["game_logs"]["game_pk"].astype(int))
    upcoming = fetch_upcoming(make_session(), today or date.today(), days, known)
    if upcoming["starts"].empty:
        return pd.DataFrame()

    feats = build_features(raw, upcoming=upcoming)
    history = feats[feats["upcoming"] == 0]
    nxt = feats[feats["upcoming"] == 1].copy()
    model = make_blend().fit(history[FEATURES], history[TARGET])
    nxt["pred_model"] = np.clip(model.predict(nxt[FEATURES]), 0, None)
    nxt["pred_baseline"] = baseline_predict(nxt, float(history[TARGET].mean()))

    # Probabilities: fit the distribution to out-of-sample predictions of the latest season.
    latest = history["season"].max()
    early, prev = history[history["season"] < latest], history[history["season"] == latest]
    mu_prev = np.clip(make_blend().fit(early[FEATURES], early[TARGET]).predict(prev[FEATURES]), 0, None)
    pmf = Distribution(PROB_METHOD).fit(prev[TARGET], mu_prev).pmf(nxt["pred_model"])
    for t in THRESHOLDS:
        nxt[f"p_{t}plus"] = prob_at_least(pmf, t)
    nxt["trained_through"] = history["game_date"].max()
    nxt["predicted_at"] = datetime.now().isoformat(timespec="minutes")

    info = upcoming["starts"][["game_pk", "pitcher_id", "game_type", "status", "actual_strikeouts"]]
    nxt = nxt.drop(columns=[TARGET]).merge(info, on=["game_pk", "pitcher_id"], how="left")
    return nxt.sort_values(["game_date", "game_pk", "is_home"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=3, help="days of schedule to look at, starting today")
    args = parser.parse_args(argv)

    preds = predict_upcoming(args.days)
    if preds.empty:
        print("No probable starters posted for the coming days.")
        return 0
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(UPCOMING_PATH, index=False)
    print(f"Model trained on starts through {preds['trained_through'].iloc[0]:%Y-%m-%d}")
    table = preds[["game_date", "pitcher_name", "team_id", "opponent_id", "status", "pred_model",
                   "pred_baseline", "actual_strikeouts"]].copy()
    table["game_date"] = table["game_date"].dt.strftime("%Y-%m-%d")
    print(table.round(2).to_string(index=False))
    print(f"\nWrote {len(preds)} rows to {UPCOMING_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Feature tests on a small hand-made fixture (no network access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from build_features import (
    CANDIDATE_FEATURES,
    FEATURES,
    ROLLING_FEATURES,
    TARGET,
    audit_leakage,
    build_features,
    slot_pa,
)
from fetch_data import PITCH_CLASSES

DATES = ["2024-04-01", "2024-04-06", "2024-04-12", "2024-04-17", "2024-04-23"]
PITCHER_A, PITCHER_B = 1, 2
TEAM_A, TEAM_B = 10, 20


@pytest.fixture
def raw():
    """Two starters facing each other five times, plus a relief outing by A between starts.

    Each team's lineup is four hand-made batters (two left-handed, one right,
    one switch) repeated in slots 1-9 of the batting order.
    """
    logs, sc, tg, bg, lineups, games = [], [], [], [], [], []
    hitters = {TEAM_A: [(11, "L"), (12, "L"), (13, "R"), (14, "S")],
               TEAM_B: [(21, "L"), (22, "R"), (23, "R"), (24, "S")]}
    for i, d in enumerate(DATES):
        gpk = 100 + i
        games.append(dict(game_pk=gpk, game_date=d, season=2024, venue_id=1, day_night="night",
                          hp_umpire_id=500 + (i % 2), temp_f=60.0 + i, condition="Clear",
                          wind_mph=5.0, wind_dir="Out To CF"))
        for pid, team, opp, home, k, bf in (
            (PITCHER_A, TEAM_A, TEAM_B, 1, 5 + i, 22 + i),
            (PITCHER_B, TEAM_B, TEAM_A, 0, 3 + i, 20 + i),
        ):
            logs.append(dict(game_pk=gpk, game_date=d, season=2024, pitcher_id=pid, pitcher_name=f"P{pid}",
                             team_id=team, opponent_id=opp, is_home=home, started=1, outs=18,
                             strikeouts=k, batters_faced=bf, pitches=90 + i))
            classes = {f"n_{c}": n for c, n in zip(PITCH_CLASSES, (10, 20, 10, 30, 20), strict=True)}
            classes.update({f"sw_{c}": 8 for c in PITCH_CLASSES})
            classes.update({f"wh_{c}": 1 + (i + j) % 3 for j, c in enumerate(PITCH_CLASSES)})
            sc.append(dict(game_pk=gpk, pitcher_id=pid, game_date=d, p_throws="R" if pid == PITCHER_A else "L",
                           sc_pitches=90 + i, fb_count=40, fb_velo_sum=40 * (94.0 + 0.5 * i), swings=40,
                           whiffs=8 + i, pa_vs_l=bf // 2, k_vs_l=k // 2, pa_vs_r=bf - bf // 2,
                           k_vs_r=k - k // 2, **classes))
            tg.append(dict(game_pk=gpk, game_date=d, season=2024, team_id=opp, opponent_id=team,
                           bat_pa=38, bat_k=k + 2))
            for slot in range(1, 10):
                bid, hand = hitters[opp][(slot - 1) % 4]
                lineups.append(dict(game_pk=gpk, game_date=d, team_id=opp, batter_id=bid, batting_order=slot))
            for bid, hand in hitters[opp]:
                p_hand = "R" if pid == PITCHER_A else "L"
                as_l = 4 if hand == "L" or (hand == "S" and p_hand == "R") else 0
                row = dict(game_pk=gpk, batter_id=bid, game_date=d, bat_team=str(opp), pa_as_l=as_l,
                           pa_as_r=4 - as_l, pa_vs_lhp=4 if p_hand == "L" else 0, k_vs_lhp=1 if p_hand == "L" else 0,
                           pa_vs_rhp=4 if p_hand == "R" else 0, k_vs_rhp=(1 + i % 2) if p_hand == "R" else 0)
                row.update({f"n_{c}": 3 for c in PITCH_CLASSES})
                row.update({f"sw_{c}": 2 for c in PITCH_CLASSES})
                row.update({f"wh_{c}": (bid + i) % 2 for c in PITCH_CLASSES})
                bg.append(row)
    # Relief appearance by A on 04-09: must not reset rest days or enter start windows.
    logs.append(dict(game_pk=999, game_date="2024-04-09", season=2024, pitcher_id=PITCHER_A, pitcher_name="PA",
                     team_id=TEAM_A, opponent_id=TEAM_B, is_home=1, started=0, outs=3,
                     strikeouts=2, batters_faced=4, pitches=15))
    return {"game_logs": pd.DataFrame(logs), "statcast": pd.DataFrame(sc), "team_games": pd.DataFrame(tg),
            "batter_games": pd.DataFrame(bg), "lineups": pd.DataFrame(lineups), "games": pd.DataFrame(games)}


def _row(features: pd.DataFrame, game_pk: int, pitcher_id: int) -> pd.Series:
    hit = features[(features["game_pk"] == game_pk) & (features["pitcher_id"] == pitcher_id)]
    assert len(hit) == 1
    return hit.iloc[0][FEATURES].astype(float)


def test_current_game_stats_do_not_change_that_games_features(raw):
    before = build_features(raw)

    target_game = 102  # A's third start
    changed = {k: v.copy() for k, v in raw.items()}
    logs, sc, bg = changed["game_logs"], changed["statcast"], changed["batter_games"]
    hit = (logs["game_pk"] == target_game) & (logs["pitcher_id"] == PITCHER_A)
    logs.loc[hit, ["strikeouts", "batters_faced", "pitches"]] = [99, 99, 200]
    sc_hit = (sc["game_pk"] == target_game) & (sc["pitcher_id"] == PITCHER_A)
    sc.loc[sc_hit, "fb_velo_sum"] = sc.loc[sc_hit, "fb_count"] * 110.0
    sc.loc[sc_hit, "whiffs"] = sc.loc[sc_hit, "swings"]
    sc.loc[sc_hit, ["k_vs_l", "k_vs_r"]] = sc.loc[sc_hit, ["pa_vs_l", "pa_vs_r"]].to_numpy()
    b_hit = bg["game_pk"] == target_game
    bg.loc[b_hit, "k_vs_rhp"] = bg.loc[b_hit, "pa_vs_rhp"]
    for c in PITCH_CLASSES:
        bg.loc[b_hit, f"wh_{c}"] = bg.loc[b_hit, f"sw_{c}"]
    after = build_features(changed)

    pd.testing.assert_series_equal(_row(before, target_game, PITCHER_A), _row(after, target_game, PITCHER_A))
    # The change must reach the next start, otherwise this test proves nothing.
    nxt_before, nxt_after = _row(before, 103, PITCHER_A), _row(after, 103, PITCHER_A)
    assert nxt_after["fb_velo_last3"] > nxt_before["fb_velo_last3"]
    assert nxt_after["k_per_bf_last5"] != nxt_before["k_per_bf_last5"]
    assert nxt_after["lineup_k_rate_vs_hand"] > nxt_before["lineup_k_rate_vs_hand"]
    assert nxt_after["pitcher_k_rate_vs_lineup"] > nxt_before["pitcher_k_rate_vs_lineup"]


def test_first_start_has_missing_rolling_features(raw):
    features = build_features(raw)
    firsts = features.sort_values("game_date").groupby("pitcher_id").head(1)
    assert len(firsts) == 2
    assert firsts[ROLLING_FEATURES].isna().all().all()
    # The second start already has a (1-start) window thanks to min_periods=1.
    second = features[(features["pitcher_id"] == PITCHER_A) & (features["game_pk"] == 101)].iloc[0]
    assert second[ROLLING_FEATURES].notna().all()
    assert second["k_per_bf_last5"] == pytest.approx(5 / 22)
    assert second["fb_velo_last3"] == pytest.approx(94.0)


def test_rest_days_come_from_previous_start(raw):
    features = build_features(raw)
    a = features[features["pitcher_id"] == PITCHER_A].sort_values("game_date")
    assert len(a) == len(DATES)  # relief outing excluded
    expected = [np.nan, 5, 6, 5, 6]  # 04-09 relief outing ignored
    np.testing.assert_array_equal(a["rest_days"].to_numpy(), np.array(expected, dtype=float))


def test_season_rate_uses_prior_games_only(raw):
    features = build_features(raw)
    a = features[features["pitcher_id"] == PITCHER_A].sort_values("game_date").reset_index(drop=True)
    # Third start: prior starts had K = 5, 6 and BF = 22, 23.
    assert a.loc[2, "season_k_rate"] == pytest.approx(11 / 45)
    assert a.loc[2, "season_k_per_start"] == pytest.approx(5.5)


def test_audit_passes_clean_features_and_catches_leaky_one(raw):
    assert audit_leakage(raw, FEATURES, n_dates=3)["passed"]
    leaky = audit_leakage(raw, FEATURES + ["batters_faced"], n_dates=3)
    assert not leaky["passed"]
    assert "batters_faced" in leaky["current_game_features"]
    assert "batters_faced" in leaky["features_changed_by_current_game"]


def test_lineup_platoon_and_context_features(raw):
    features = build_features(raw)
    # A (RHP) faces team B: slots cycle L, R, R, S. Switch hitters bat left vs a righty,
    # so lefties and the switch hitter have the platoon edge.
    a = features[(features["pitcher_id"] == PITCHER_A) & (features["game_pk"] == 101)].iloc[0]
    weights = np.array([4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95, 3.85])
    adv = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    assert a["is_lhp"] == 0
    assert a["lineup_platoon_adv_share"] == pytest.approx((weights * adv).sum() / weights.sum())
    assert a["wind_out_mph"] == 5.0 and a["roof_closed"] == 0 and a["day_game"] == 0
    # Only one prior game at the venue: the park factor exists and is near 1 after shrinkage.
    assert 0.8 < a["park_k_factor"] < 1.2
    # First game anywhere has no prior park/umpire history beyond the league default.
    first = features[features["game_pk"] == 100]
    assert first["park_k_factor"].isna().all()
    b = features[(features["pitcher_id"] == PITCHER_B) & (features["game_pk"] == 101)].iloc[0]
    assert b["is_lhp"] == 1


def test_previous_lineup_mode_uses_last_game(raw):
    posted = build_features(raw, lineup_source="posted")
    previous = build_features(raw, lineup_source="previous")
    # The first game has no previous lineup.
    assert previous.loc[previous["game_pk"] == 100, "lineup_platoon_adv_share"].isna().all()
    assert posted.loc[posted["game_pk"] == 100, "lineup_platoon_adv_share"].notna().all()


def upcoming_tables(lineups=None):
    """A sixth meeting of A and B, not played yet, with no lineups posted unless given."""
    day, gpk = "2024-04-28", 200
    starts = [dict(game_pk=gpk, game_date=day, season=2024, pitcher_id=pid, pitcher_name=f"P{pid}", team_id=team,
                   opponent_id=opp, is_home=home, game_type="R", status="Scheduled", actual_strikeouts=None)
              for pid, team, opp, home in ((PITCHER_A, TEAM_A, TEAM_B, 1), (PITCHER_B, TEAM_B, TEAM_A, 0))]
    team_games = [dict(game_pk=gpk, game_date=day, season=2024, team_id=t, opponent_id=o, bat_pa=np.nan, bat_k=np.nan)
                  for t, o in ((TEAM_A, TEAM_B), (TEAM_B, TEAM_A))]
    return {
        "starts": pd.DataFrame(starts),
        "games": pd.DataFrame([dict(game_pk=gpk, game_date=day, season=2024, venue_id=1, day_night="night")]),
        "lineups": pd.DataFrame(lineups or [], columns=["game_pk", "game_date", "team_id", "batter_id",
                                                        "batting_order"]),
        "team_games": pd.DataFrame(team_games),
    }


def test_upcoming_starts_read_history_only(raw):
    base = build_features(raw)
    out = build_features(raw, upcoming=upcoming_tables())
    history = out[out["upcoming"] == 0].drop(columns="upcoming").reset_index(drop=True)
    pd.testing.assert_frame_equal(history, base)

    nxt = out[out["upcoming"] == 1].set_index("pitcher_id")
    assert len(nxt) == 2 and nxt[TARGET].isna().all()
    a = nxt.loc[PITCHER_A]
    assert a["rest_days"] == 5
    assert a["season_k_per_start"] == pytest.approx(np.mean([5, 6, 7, 8, 9]))
    assert a["n_prior_starts"] == 5
    # No posted lineup: the opponent's lineup from its last game is used, so matchup features exist.
    assert nxt[["lineup_k_rate_vs_hand", "pitcher_k_rate_vs_lineup", "park_k_factor"]].notna().all().all()


def test_upcoming_posted_lineup_is_used(raw):
    lefties = [dict(game_pk=200, game_date="2024-04-28", team_id=TEAM_B, batter_id=21, batting_order=s)
               for s in range(1, 10)]
    out = build_features(raw, upcoming=upcoming_tables(lefties))
    a = out[(out["upcoming"] == 1) & (out["pitcher_id"] == PITCHER_A)].iloc[0]
    assert a["lineup_platoon_adv_share"] == pytest.approx(1.0)  # every batter a lefty vs a righty


def test_candidate_features_pass_leakage_audit(raw):
    report = audit_leakage(raw, CANDIDATE_FEATURES, n_dates=3)
    assert report["passed"], report
    out = build_features(raw)
    a = out[out["pitcher_id"] == PITCHER_A].reset_index(drop=True)
    # Pitches per PA over his previous starts: (90 + 91) / (22 + 23) before his third start.
    assert a.loc[2, "pitches_per_pa_last5"] == pytest.approx((90 + 91) / (22 + 23))
    # Team hook needs 5 earlier game-days, so the fixture's five starts never have one.
    assert out["team_starter_bf_last30"].isna().all()


def test_rest_flags_and_previous_pitch_count(raw):
    a = build_features(raw)
    a = a[a["pitcher_id"] == PITCHER_A].sort_values("game_date").reset_index(drop=True)
    for col in ("short_rest", "long_layoff", "il_return", "pitches_prev"):
        assert np.isnan(a.loc[0, col])
    assert (a.loc[1:, ["short_rest", "long_layoff", "il_return"]] == 0).all().all()  # 5-6 days of rest
    assert a.loc[1:, "pitches_prev"].tolist() == [90, 91, 92, 93]  # relief outing ignored


def test_slot_pa_counts_trips_through_the_order():
    pa = slot_pa(np.array([9, 10, 27, 9.5, 0]))
    assert pa[0].tolist() == [1] * 9
    assert pa[1].tolist() == [2] + [1] * 8
    assert pa[2].tolist() == [3] * 9
    assert pa[3, 0] == pytest.approx(1.5) and pa[3, 1] == 1
    assert pa[4].sum() == 0


def test_matchup_expected_k(raw):
    out = build_features(raw)
    a = out[out["pitcher_id"] == PITCHER_A].sort_values("game_date").reset_index(drop=True)
    assert np.isnan(a.loc[0, "matchup_expected_k"])  # no batters-faced history yet
    assert 0 < a.loc[2, "matchup_expected_k"] < a.loc[2, "bf_last5"]
    assert 0 < a.loc[2, "lineup_log5_k_rate"] < 1


def test_park_and_umpire_factors_use_earlier_dates_only(raw):
    """Dropping every game after a date leaves the factors on and before it unchanged."""
    cut = pd.Timestamp(DATES[2])
    early = {k: v[pd.to_datetime(v["game_date"]) <= cut] for k, v in raw.items()}
    cols = ["park_k_factor", "park_k_factor_hr", "ump_k_factor"]
    full = build_features(raw).set_index(["game_pk", "pitcher_id"])
    part = build_features(early).set_index(["game_pk", "pitcher_id"])
    pd.testing.assert_frame_equal(full.loc[part.index, cols], part[cols])
    assert full["park_k_factor_hr"].notna().sum() >= 2

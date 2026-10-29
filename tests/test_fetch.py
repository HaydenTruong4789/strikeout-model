"""Parsing tests for fetch_data on hand-made API payloads (no network access)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from fetch_data import date_chunks, innings_to_outs, parse_boxscore, parse_weather, summarize_statcast


def test_innings_to_outs():
    assert innings_to_outs("6.1") == 19
    assert innings_to_outs("5.0") == 15
    assert innings_to_outs("0.2") == 2
    assert innings_to_outs(None) is None


def test_date_chunks_cover_range_without_gaps():
    chunks = date_chunks(date(2024, 4, 1), date(2024, 4, 30), days=14)
    assert chunks[0] == (date(2024, 4, 1), date(2024, 4, 14))
    assert chunks[-1][1] == date(2024, 4, 30)
    for (_, end), (start, _) in zip(chunks, chunks[1:], strict=False):
        assert (start - end).days == 1


def test_parse_boxscore_marks_starter_and_opponent():
    def side(team_id, starter, reliever, k):
        hitters = {f"ID{team_id * 10 + n}": {"person": {"id": team_id * 10 + n}, "battingOrder": str(n * 100)}
                   for n in range(1, 10)}
        hitters[f"ID{team_id * 10 + 99}"] = {"person": {"id": team_id * 10 + 99}, "battingOrder": "301"}
        return {
            "team": {"id": team_id},
            "teamStats": {"batting": {"strikeOuts": k, "plateAppearances": 38}},
            "pitchers": [starter, reliever],
            "players": {
                **hitters,
                f"ID{starter}": {"person": {"fullName": f"S{starter}"}, "stats": {"pitching": {
                    "gamesStarted": 1, "inningsPitched": "6.1", "strikeOuts": 8,
                    "battersFaced": 25, "numberOfPitches": 98}}},
                f"ID{reliever}": {"person": {"fullName": f"R{reliever}"}, "stats": {"pitching": {
                    "gamesStarted": 0, "inningsPitched": "1.0", "strikeOuts": 1,
                    "battersFaced": 4, "numberOfPitches": 15}}},
            },
        }

    box = {
        "teams": {"home": side(10, 1, 2, 9), "away": side(20, 3, 4, 7)},
        "officials": [{"official": {"id": 77}, "officialType": "First Base"},
                      {"official": {"id": 88}, "officialType": "Home Plate"}],
        "info": [{"label": "Weather", "value": "72 degrees, Partly Cloudy."},
                 {"label": "Wind", "value": "8 mph, Out To CF."}],
    }
    game = {"game_pk": 555, "game_date": "2024-05-01", "season": 2024, "venue_id": 3, "day_night": "day"}
    pitchers, teams, lineups, info = parse_boxscore(box, game)
    p = pd.DataFrame(pitchers).set_index("pitcher_id")
    assert p.loc[1, "started"] == 1 and p.loc[2, "started"] == 0
    assert p.loc[1, "outs"] == 19 and p.loc[1, "is_home"] == 1 and p.loc[1, "opponent_id"] == 20
    assert p.loc[3, "is_home"] == 0 and p.loc[3, "opponent_id"] == 10
    t = pd.DataFrame(teams).set_index("team_id")
    assert t.loc[10, "bat_k"] == 9 and t.loc[20, "opponent_id"] == 10
    lu = pd.DataFrame(lineups)
    assert len(lu) == 18  # substitute (battingOrder 301) excluded
    assert sorted(lu[lu["team_id"] == 10]["batting_order"]) == list(range(1, 10))
    assert info["hp_umpire_id"] == 88 and info["venue_id"] == 3 and info["day_night"] == "day"
    assert info["temp_f"] == 72 and info["wind_mph"] == 8 and info["wind_dir"] == "Out To CF"


def test_parse_weather_handles_dome_and_missing():
    w = parse_weather([{"label": "Weather", "value": "72 degrees, Dome."}, {"label": "Wind", "value": "0 mph, None."}])
    assert w["condition"] == "Dome" and w["wind_mph"] == 0
    assert parse_weather([])["temp_f"] is None


def test_summarize_statcast_counts_swings_whiffs_classes_and_splits():
    pitches = pd.DataFrame({
        "game_pk": [1] * 6,
        "pitcher": [7] * 6,
        "batter": [50, 50, 50, 51, 51, 51],
        "game_date": ["2024-05-01"] * 6,
        "game_type": ["R"] * 6,
        "p_throws": ["R"] * 6,
        "stand": ["L", "L", "L", "R", "R", "R"],
        "inning_topbot": ["Top"] * 6,
        "home_team": ["SEA"] * 6,
        "away_team": ["NYY"] * 6,
        "pitch_type": ["FF", "FF", "SI", "SL", "CH", "FF"],
        "release_speed": [97.0, 96.0, 92.0, 85.0, 87.0, 94.0],
        "description": ["swinging_strike", "foul", "ball", "hit_into_play", "foul_tip", "called_strike"],
        "events": [None, None, "walk", "single", None, "strikeout"],
    })
    pitcher, batter = summarize_statcast(pitches)
    p = pitcher.iloc[0]
    assert p["sc_pitches"] == 6 and p["p_throws"] == "R"
    assert p["fb_count"] == 4
    assert p["fb_velo_sum"] == 97.0 + 96.0 + 92.0 + 94.0
    assert p["swings"] == 4 and p["whiffs"] == 2 and p["called"] == 1
    assert (p["n_fb_hi"], p["n_fb_mid"], p["n_fb_lo"], p["n_brk"], p["n_os"]) == (2, 1, 1, 1, 1)
    assert p["wh_fb_hi"] == 1 and p["sw_fb_hi"] == 2 and p["wh_os"] == 1
    assert (p["pa_vs_l"], p["k_vs_l"], p["pa_vs_r"], p["k_vs_r"]) == (1, 0, 2, 1)
    b = batter.set_index("batter_id")
    assert b.loc[50, "pa_as_l"] == 1 and b.loc[50, "pa_vs_rhp"] == 1 and b.loc[50, "bat_team"] == "NYY"
    assert b.loc[51, "k_vs_rhp"] == 1 and b.loc[51, "pa_vs_rhp"] == 2

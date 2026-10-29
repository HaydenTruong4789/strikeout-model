"""Download and cache MLB pitcher game logs and Statcast pitch summaries.

Two sources are cached under ``data/raw``:

1. **MLB Stats API** (statsapi.mlb.com). For every final regular-season and
   postseason game
   we read the schedule entry and the boxscore and keep:

   * ``game_logs``: one row per pitcher appearance (starter flag, outs,
     strikeouts, batters faced, pitch count, home/away, opponent)
   * ``team_games``: each team's batting strikeouts and plate appearances
   * ``games``: venue, day/night, game-time temperature, sky condition, wind,
     and the home plate umpire
   * ``lineups``: each team's starting nine in batting order

2. **Statcast** pitch-level data via ``pybaseball.statcast``. Raw pitches are
   cached locally (gitignored, large) and reduced to two small summaries that
   the feature builder reads and that get committed to git:

   * ``statcast_agg/pitcher``: one row per pitcher per game: throwing hand,
     fastball velocity, swings and whiffs overall and per pitch class,
     plate appearances and strikeouts against left- and right-handed batters
   * ``statcast_agg/batter``: one row per batter per game: batting side,
     plate appearances and strikeouts against LHP and RHP, and swings and
     whiffs per pitch class

   Pitch classes split fastballs (four-seam, sinker, cutter) by velocity,
   ``fb_lo`` under 93 mph, ``fb_mid`` 93-95.9 and ``fb_hi`` 96+, and group the
   rest into breaking balls (``brk``) and offspeed pitches (``os``).

Data is fetched in date chunks. Each finished chunk is written to its own
parquet file named ``<start>_<end>.parquet``; a chunk that already exists is
skipped, so an interrupted run resumes where it stopped. A chunk with any
failed request is not written, so the next run retries it.

Usage::

    python src/fetch_data.py                       # 2021-2026 regular seasons and postseasons
    python src/fetch_data.py --seasons 2024 2025   # specific seasons
    python src/fetch_data.py --no-postseason       # regular seasons only
    python src/fetch_data.py --incremental         # only games after the last cached date
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
GAME_LOG_DIR = RAW_DIR / "game_logs"          # pitcher appearances (committed)
TEAM_GAME_DIR = RAW_DIR / "team_games"        # team batting K / PA per game (committed)
STATCAST_RAW_DIR = RAW_DIR / "statcast"       # raw pitches (gitignored, large)
GAMES_DIR = RAW_DIR / "games"                 # venue, weather, umpire per game (committed)
LINEUP_DIR = RAW_DIR / "lineups"              # starting nine per team per game (committed)
STATCAST_AGG_DIR = RAW_DIR / "statcast_agg"
PITCHER_AGG_DIR = STATCAST_AGG_DIR / "pitcher"  # pitcher-game summaries (committed)
BATTER_AGG_DIR = STATCAST_AGG_DIR / "batter"    # batter-game summaries (committed)

STATSAPI = "https://statsapi.mlb.com/api/v1"
DEFAULT_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
REGULAR_SEASON = "R"
# Wild card, division series, league championship series, World Series.
POSTSEASON = "F,D,L,W"
GAME_TYPES = {"R", "F", "D", "L", "W"}
CHUNK_DAYS = 14
MAX_WORKERS = 4
REQUEST_PAUSE_S = 0.1

# Statcast `description` values. Foul tips count as whiffs: the batter swung,
# missed enough to be credited a swinging strike, and MLB scores them that way.
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | {
    "foul", "foul_bunt", "bunt_foul_tip", "hit_into_play",
    "hit_into_play_no_out", "hit_into_play_score",
}
# Four-seam fastballs and sinkers (FT is the pre-2017 two-seam code) define the
# velocity features; cutters join them in the fastball pitch classes.
FASTBALL_TYPES = {"FF", "SI", "FT"}
FASTBALL_CLASS_TYPES = FASTBALL_TYPES | {"FC"}
BREAKING_TYPES = {"SL", "ST", "SV", "CU", "KC", "CS"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "SC", "KN", "EP"}
PITCH_CLASSES = ["fb_lo", "fb_mid", "fb_hi", "brk", "os"]
FB_MID_MPH, FB_HI_MPH = 93.0, 96.0
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
# Statcast `events` that end a play without ending the plate appearance.
NON_PA_EVENT_PREFIXES = ("caught_stealing", "pickoff", "stolen_base")
NON_PA_EVENTS = {"wild_pitch", "passed_ball", "other_advance", "runner_double_play",
                 "game_advisory", "truncated_pa", "ejection", "defensive_indiff", "balk"}

log = logging.getLogger("fetch_data")


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    """Return a requests session that retries rate limits and server errors.

    Retries 429 and 5xx responses up to 6 times with exponential backoff
    (1s, 2s, 4s, ...) and honours a ``Retry-After`` header when one is sent.
    """
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=MAX_WORKERS * 2)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = "strikeout-model/1.0 (research project)"
    return session


def get_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    """GET a URL and return parsed JSON, raising on any HTTP or network error."""
    time.sleep(REQUEST_PAUSE_S)
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Season dates and chunking
# --------------------------------------------------------------------------- #
def season_window(session: requests.Session, season: int) -> tuple[date, date]:
    """Return the regular-season start and end dates for a season.

    Falls back to a generous Mar 15 - Oct 5 window if the API call fails; games
    outside the real regular season are filtered by game type anyway.
    """
    try:
        data = get_json(session, f"{STATSAPI}/seasons", {"sportId": 1, "season": season})
        s = data["seasons"][0]
        return (
            date.fromisoformat(s["regularSeasonStartDate"]),
            date.fromisoformat(s["regularSeasonEndDate"]),
        )
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the default window
        log.warning("Season dates for %s unavailable (%s); using Mar 15 - Oct 5", season, exc)
        return date(season, 3, 15), date(season, 10, 5)


def postseason_window(session: requests.Session, season: int) -> tuple[date, date] | None:
    """Return the postseason start and end dates for a season, or None if unavailable."""
    try:
        data = get_json(session, f"{STATSAPI}/seasons", {"sportId": 1, "season": season})
        s = data["seasons"][0]
        return date.fromisoformat(s["postSeasonStartDate"]), date.fromisoformat(s["postSeasonEndDate"])
    except Exception as exc:  # noqa: BLE001 - no postseason dates means no postseason fetch
        log.warning("Postseason dates for %s unavailable (%s); skipping it", season, exc)
        return None


def date_chunks(start: date, end: date, days: int = CHUNK_DAYS) -> list[tuple[date, date]]:
    """Split the inclusive range [start, end] into consecutive chunks of ``days`` days."""
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def chunk_path(directory: Path, start: date, end: date) -> Path:
    """Return the parquet path for a cached chunk."""
    return directory / f"{start.isoformat()}_{end.isoformat()}.parquet"


# --------------------------------------------------------------------------- #
# MLB Stats API: game logs from boxscores
# --------------------------------------------------------------------------- #
def list_final_games(session: requests.Session, start: date, end: date,
                     game_types: str = REGULAR_SEASON) -> list[dict]:
    """List completed games of ``game_types`` between two dates (inclusive).

    Postponed and cancelled games are dropped. A suspended game appears on both
    its original and resumption dates; it is kept once, on the date it ended.
    """
    data = get_json(
        session,
        f"{STATSAPI}/schedule",
        {"sportId": 1, "gameType": game_types, "startDate": start.isoformat(), "endDate": end.isoformat()},
    )
    games: dict[int, dict] = {}
    for day in data.get("dates", []):
        for g in day.get("games", []):
            status = g.get("status", {})
            if status.get("abstractGameState") != "Final":
                continue
            if status.get("detailedState") in {"Postponed", "Cancelled"}:
                continue
            games[g["gamePk"]] = {
                "game_pk": g["gamePk"],
                "game_date": g.get("officialDate", day["date"]),
                "season": int(g.get("season", day["date"][:4])),
                "home_id": g["teams"]["home"]["team"]["id"],
                "away_id": g["teams"]["away"]["team"]["id"],
                "venue_id": g.get("venue", {}).get("id"),
                "venue_name": g.get("venue", {}).get("name"),
                "day_night": g.get("dayNight"),
            }
    return list(games.values())


def innings_to_outs(ip: str | float | None) -> int | None:
    """Convert baseball innings notation ("6.1" = 6 1/3) to outs recorded."""
    if ip in (None, ""):
        return None
    whole, _, frac = str(ip).partition(".")
    return int(whole) * 3 + (int(frac) if frac else 0)


def parse_weather(info: list[dict]) -> dict:
    """Parse the boxscore "Weather" and "Wind" info lines.

    Example values: ``"72 degrees, Partly Cloudy."`` and ``"8 mph, Out To CF."``.
    """
    lines = {i.get("label"): i.get("value", "") for i in info if isinstance(i, dict)}
    weather, wind = lines.get("Weather", ""), lines.get("Wind", "")
    temp = re.search(r"(-?\d+)\s*degrees", weather)
    mph = re.search(r"(\d+)\s*mph", wind)
    return {
        "temp_f": float(temp.group(1)) if temp else None,
        "condition": weather.split(",", 1)[1].strip(" .") if "," in weather else None,
        "wind_mph": float(mph.group(1)) if mph else None,
        "wind_dir": wind.split(",", 1)[1].strip(" .") if "," in wind else None,
    }


def parse_boxscore(box: dict, game: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Turn one boxscore JSON into pitcher, team-batting, lineup, and game-info rows."""
    pitcher_rows, team_rows, lineup_rows = [], [], []
    for side, opp_side in (("home", "away"), ("away", "home")):
        team = box["teams"][side]
        team_id = team["team"]["id"]
        opp_id = box["teams"][opp_side]["team"]["id"]
        batting = team.get("teamStats", {}).get("batting", {})
        team_rows.append({
            "game_pk": game["game_pk"],
            "game_date": game["game_date"],
            "season": game["season"],
            "team_id": team_id,
            "opponent_id": opp_id,
            "bat_pa": batting.get("plateAppearances"),
            "bat_k": batting.get("strikeOuts"),
        })
        for player in team.get("players", {}).values():
            order = str(player.get("battingOrder", ""))
            # "100", "200", ... are the starters; "101" etc. are substitutes.
            if order.isdigit() and int(order) % 100 == 0:
                lineup_rows.append({
                    "game_pk": game["game_pk"],
                    "game_date": game["game_date"],
                    "team_id": team_id,
                    "batter_id": player["person"]["id"],
                    "batting_order": int(order) // 100,
                })
        for pid in team.get("pitchers", []):
            player = team["players"].get(f"ID{pid}", {})
            p = player.get("stats", {}).get("pitching", {})
            if not p:
                continue
            pitcher_rows.append({
                "game_pk": game["game_pk"],
                "game_date": game["game_date"],
                "season": game["season"],
                "pitcher_id": pid,
                "pitcher_name": player.get("person", {}).get("fullName"),
                "team_id": team_id,
                "opponent_id": opp_id,
                "is_home": int(side == "home"),
                "started": int(p.get("gamesStarted", 0) or 0),
                "outs": innings_to_outs(p.get("inningsPitched")),
                "strikeouts": p.get("strikeOuts"),
                "batters_faced": p.get("battersFaced"),
                "pitches": p.get("numberOfPitches", p.get("pitchesThrown")),
            })
    hp_ump = next((o.get("official", {}) for o in box.get("officials", [])
                   if o.get("officialType") == "Home Plate"), {})
    game_row = {
        "game_pk": game["game_pk"],
        "game_date": game["game_date"],
        "season": game["season"],
        "venue_id": game.get("venue_id"),
        "venue_name": game.get("venue_name"),
        "day_night": game.get("day_night"),
        "hp_umpire_id": hp_ump.get("id"),
        "hp_umpire_name": hp_ump.get("fullName"),
        **parse_weather(box.get("info", [])),
    }
    return pitcher_rows, team_rows, lineup_rows, game_row


def fetch_game_log_chunk(session: requests.Session, start: date, end: date,
                         game_types: str = REGULAR_SEASON) -> bool:
    """Fetch and cache game logs, team batting, lineups, and game info for one date chunk.

    Returns True if the chunk was written (or there were no games), False if any
    request failed, in which case nothing is written and a rerun will retry it.
    """
    games = list_final_games(session, start, end, game_types)
    pitcher_rows, team_rows, lineup_rows, game_rows, failures = [], [], [], [], []

    def one(game: dict):
        box = get_json(session, f"{STATSAPI}/game/{game['game_pk']}/boxscore")
        return parse_boxscore(box, game)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(one, g): g for g in games}
        for fut in as_completed(futures):
            try:
                p_rows, t_rows, l_rows, g_row = fut.result()
                pitcher_rows.extend(p_rows)
                team_rows.extend(t_rows)
                lineup_rows.extend(l_rows)
                game_rows.append(g_row)
            except Exception as exc:  # noqa: BLE001 - record and keep going
                failures.append((futures[fut]["game_pk"], str(exc)))

    if failures:
        log.error("Game logs %s..%s: %d of %d games failed (first: %s); chunk not saved, rerun to retry",
                  start, end, len(failures), len(games), failures[0])
        return False

    for directory, rows, columns in (
        (TEAM_GAME_DIR, team_rows, TEAM_GAME_COLUMNS),
        (LINEUP_DIR, lineup_rows, LINEUP_COLUMNS),
        (GAMES_DIR, game_rows, GAME_COLUMNS),
        (GAME_LOG_DIR, pitcher_rows, PITCHER_LOG_COLUMNS),  # written last: marks the chunk done
    ):
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_parquet(chunk_path(directory, start, end), index=False)
    log.info("Game logs %s..%s: %d games, %d pitcher rows", start, end, len(games), len(pitcher_rows))
    return True


PITCHER_LOG_COLUMNS = [
    "game_pk", "game_date", "season", "pitcher_id", "pitcher_name", "team_id", "opponent_id",
    "is_home", "started", "outs", "strikeouts", "batters_faced", "pitches",
]
TEAM_GAME_COLUMNS = ["game_pk", "game_date", "season", "team_id", "opponent_id", "bat_pa", "bat_k"]
LINEUP_COLUMNS = ["game_pk", "game_date", "team_id", "batter_id", "batting_order"]
GAME_COLUMNS = ["game_pk", "game_date", "season", "venue_id", "venue_name", "day_night",
                "hp_umpire_id", "hp_umpire_name", "temp_f", "condition", "wind_mph", "wind_dir"]
CLASS_COLUMNS = [f"{kind}_{c}" for c in PITCH_CLASSES for kind in ("n", "sw", "wh")]
PITCHER_AGG_COLUMNS = [
    "game_pk", "pitcher_id", "game_date", "p_throws", "sc_pitches", "fb_count", "fb_velo_sum",
    "swings", "whiffs", "called", "pa_vs_l", "k_vs_l", "pa_vs_r", "k_vs_r", *CLASS_COLUMNS,
]
BATTER_AGG_COLUMNS = [
    "game_pk", "batter_id", "game_date", "bat_team", "pa_as_l", "pa_as_r",
    "pa_vs_lhp", "k_vs_lhp", "pa_vs_rhp", "k_vs_rhp", *CLASS_COLUMNS,
]


# --------------------------------------------------------------------------- #
# Statcast
# --------------------------------------------------------------------------- #
def pitch_class(pitch_type: pd.Series, speed: pd.Series) -> pd.Series:
    """Label each pitch fb_lo / fb_mid / fb_hi / brk / os (NaN for unknown types)."""
    fb = pitch_type.isin(FASTBALL_CLASS_TYPES)
    out = pd.Series(np.nan, index=pitch_type.index, dtype="object")
    out[fb & (speed < FB_MID_MPH)] = "fb_lo"
    out[fb & (speed >= FB_MID_MPH) & (speed < FB_HI_MPH)] = "fb_mid"
    out[fb & (speed >= FB_HI_MPH)] = "fb_hi"
    out[pitch_type.isin(BREAKING_TYPES)] = "brk"
    out[pitch_type.isin(OFFSPEED_TYPES)] = "os"
    return out


def _tag_pitches(pitches: pd.DataFrame) -> pd.DataFrame:
    """Add swing/whiff/PA/K/pitch-class flags used by both summaries."""
    df = pitches
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(GAME_TYPES)]  # drops spring training and exhibitions
    events = df["events"].fillna("").astype(str)
    is_pa = (events != "") & ~events.str.startswith(NON_PA_EVENT_PREFIXES) & ~events.isin(NON_PA_EVENTS)
    is_fb = df["pitch_type"].isin(FASTBALL_TYPES) & df["release_speed"].notna()
    df = df.assign(
        is_fb=is_fb,
        fb_speed=df["release_speed"].where(is_fb, 0.0),
        is_swing=df["description"].isin(SWING_DESCRIPTIONS),
        is_whiff=df["description"].isin(WHIFF_DESCRIPTIONS),
        is_called=df["description"] == "called_strike",
        is_pa=is_pa,
        is_k=is_pa & events.isin(STRIKEOUT_EVENTS),
        pclass=pitch_class(df["pitch_type"], df["release_speed"]),
        bat_team=np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"]),
    )
    for c in PITCH_CLASSES:
        in_c = df["pclass"] == c
        df[f"n_{c}"] = in_c
        df[f"sw_{c}"] = in_c & df["is_swing"]
        df[f"wh_{c}"] = in_c & df["is_whiff"]
    for hand in ("L", "R"):
        h = hand.lower()
        df[f"pa_vs_{h}"] = df["is_pa"] & (df["stand"] == hand)
        df[f"k_vs_{h}"] = df["is_k"] & (df["stand"] == hand)
        df[f"pa_as_{h}"] = df[f"pa_vs_{h}"]
        df[f"pa_vs_{h}hp"] = df["is_pa"] & (df["p_throws"] == hand)
        df[f"k_vs_{h}hp"] = df["is_k"] & (df["p_throws"] == hand)
    return df


def summarize_statcast(pitches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce raw Statcast pitches to (pitcher-game, batter-game) summaries.

    Pitcher rows: throwing hand, pitches, fastball count and summed velocity
    (so averages can be re-weighted later), swings, whiffs, called strikes, PA and K against
    left- and right-handed batters, and pitches/swings/whiffs per pitch class.
    Batter rows: plate appearances by batting side, PA and K against LHP and
    RHP, and swings/whiffs per pitch class seen.
    """
    if pitches.empty:
        return pd.DataFrame(columns=PITCHER_AGG_COLUMNS), pd.DataFrame(columns=BATTER_AGG_COLUMNS)
    df = _tag_pitches(pitches)
    counts = {c: (c, "sum") for c in CLASS_COLUMNS}

    pitcher = (
        df.groupby(["game_pk", "pitcher"], as_index=False)
        .agg(
            game_date=("game_date", "first"),
            p_throws=("p_throws", "first"),
            sc_pitches=("description", "size"),
            fb_count=("is_fb", "sum"),
            fb_velo_sum=("fb_speed", "sum"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
            called=("is_called", "sum"),
            pa_vs_l=("pa_vs_l", "sum"), k_vs_l=("k_vs_l", "sum"),
            pa_vs_r=("pa_vs_r", "sum"), k_vs_r=("k_vs_r", "sum"),
            **counts,
        )
        .rename(columns={"pitcher": "pitcher_id"})
    )
    batter = (
        df.groupby(["game_pk", "batter"], as_index=False)
        .agg(
            game_date=("game_date", "first"),
            bat_team=("bat_team", "first"),
            pa_as_l=("pa_as_l", "sum"), pa_as_r=("pa_as_r", "sum"),
            pa_vs_lhp=("pa_vs_lhp", "sum"), k_vs_lhp=("k_vs_lhp", "sum"),
            pa_vs_rhp=("pa_vs_rhp", "sum"), k_vs_rhp=("k_vs_rhp", "sum"),
            **counts,
        )
        .rename(columns={"batter": "batter_id"})
    )
    for out in (pitcher, batter):
        out["game_date"] = pd.to_datetime(out["game_date"]).dt.strftime("%Y-%m-%d")
    return pitcher[PITCHER_AGG_COLUMNS], batter[BATTER_AGG_COLUMNS]


def download_statcast(start: date, end: date, attempts: int = 4) -> pd.DataFrame:
    """Download raw Statcast pitches for a date range with retries and backoff."""
    from pybaseball import statcast  # imported lazily so tests never need it

    for attempt in range(1, attempts + 1):
        try:
            df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat(), verbose=False)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001 - Savant errors are not typed
            wait = 2 ** attempt * 5
            log.warning("Statcast %s..%s attempt %d failed (%s); retrying in %ss",
                        start, end, attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Statcast download failed for {start}..{end}")


def fetch_statcast_chunk(start: date, end: date, keep_raw: bool = True) -> bool:
    """Fetch, cache, and summarize Statcast pitches for one date chunk."""
    raw_path = chunk_path(STATCAST_RAW_DIR, start, end)
    try:
        if raw_path.exists():
            pitches = pd.read_parquet(raw_path)
        else:
            pitches = download_statcast(start, end)
            if keep_raw and not pitches.empty:
                STATCAST_RAW_DIR.mkdir(parents=True, exist_ok=True)
                pitches.to_parquet(raw_path, index=False)
    except Exception as exc:  # noqa: BLE001
        log.error("Statcast %s..%s failed: %s; chunk not saved, rerun to retry", start, end, exc)
        return False

    pitcher, batter = summarize_statcast(pitches)
    BATTER_AGG_DIR.mkdir(parents=True, exist_ok=True)
    PITCHER_AGG_DIR.mkdir(parents=True, exist_ok=True)
    batter.to_parquet(chunk_path(BATTER_AGG_DIR, start, end), index=False)
    pitcher.to_parquet(chunk_path(PITCHER_AGG_DIR, start, end), index=False)  # last: marks chunk done
    log.info("Statcast %s..%s: %d pitches -> %d pitcher-games, %d batter-games",
             start, end, len(pitches), len(pitcher), len(batter))
    return True


# --------------------------------------------------------------------------- #
# Upcoming starts: probable pitchers from the schedule
# --------------------------------------------------------------------------- #
UPCOMING_COLUMNS = ["game_pk", "game_date", "season", "pitcher_id", "pitcher_name", "team_id",
                    "opponent_id", "is_home", "game_type", "status", "actual_strikeouts"]


def fetch_upcoming(session: requests.Session, start: date, days: int = 3,
                   known_game_pks: set[int] | frozenset = frozenset()) -> dict[str, pd.DataFrame]:
    """Probable starters for games from ``start`` through ``start + days - 1``.

    Only games not already in the cache are returned, so a game that started
    or finished today is still predicted from pre-game information (and its
    actual strikeouts are attached once the boxscore has them). Each game's
    boxscore supplies the posted lineups, home plate umpire and weather when
    they are out; otherwise those are left for the feature builder to fill.

    Returns tables shaped like the cache: ``starts`` (one row per probable
    starter), ``games``, ``lineups`` and ``team_games`` (stats left empty).
    """
    data = get_json(session, f"{STATSAPI}/schedule", {
        "sportId": 1, "gameType": ",".join(sorted(GAME_TYPES)), "hydrate": "probablePitcher",
        "startDate": start.isoformat(), "endDate": (start + timedelta(days=days - 1)).isoformat(),
    })
    starts, games, lineups, team_games = [], [], [], []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if g["gamePk"] in known_game_pks or status in {"Postponed", "Cancelled", "Suspended"}:
                continue
            sides = {s: g["teams"][s] for s in ("home", "away")}
            probables = {s: t.get("probablePitcher") for s, t in sides.items() if t.get("probablePitcher")}
            if not probables:
                continue
            game = {
                "game_pk": g["gamePk"],
                "game_date": g.get("officialDate", day["date"]),
                "season": int(g.get("season", day["date"][:4])),
                "venue_id": g.get("venue", {}).get("id"),
                "venue_name": g.get("venue", {}).get("name"),
                "day_night": g.get("dayNight"),
            }
            box_pitchers: dict[int, dict] = {}
            game_row = {**{c: None for c in GAME_COLUMNS}, **game}
            try:
                box = get_json(session, f"{STATSAPI}/game/{g['gamePk']}/boxscore")
                p_rows, _, l_rows, game_row = parse_boxscore(box, game)
                box_pitchers = {r["pitcher_id"]: r for r in p_rows}
                lineups.extend(l_rows)
            except Exception as exc:  # noqa: BLE001 - lineups and weather are optional here
                log.warning("Boxscore for upcoming game %s unavailable (%s)", g["gamePk"], exc)
            games.append(game_row)
            for side, opp in (("home", "away"), ("away", "home")):
                team_games.append({"game_pk": g["gamePk"], "game_date": game["game_date"], "season": game["season"],
                                   "team_id": sides[side]["team"]["id"], "opponent_id": sides[opp]["team"]["id"],
                                   "bat_pa": np.nan, "bat_k": np.nan})
                pp = probables.get(side)
                if not pp:
                    continue
                done = g.get("status", {}).get("abstractGameState") == "Final"
                actual = box_pitchers.get(pp["id"], {}).get("strikeouts") if done else None
                starts.append({
                    "game_pk": g["gamePk"], "game_date": game["game_date"], "season": game["season"],
                    "pitcher_id": pp["id"], "pitcher_name": pp.get("fullName"),
                    "team_id": sides[side]["team"]["id"], "opponent_id": sides[opp]["team"]["id"],
                    "is_home": int(side == "home"), "game_type": g.get("gameType"), "status": status,
                    "actual_strikeouts": actual,
                })
    return {
        "starts": pd.DataFrame(starts, columns=UPCOMING_COLUMNS),
        "games": pd.DataFrame(games, columns=GAME_COLUMNS),
        "lineups": pd.DataFrame(lineups, columns=LINEUP_COLUMNS),
        "team_games": pd.DataFrame(team_games, columns=TEAM_GAME_COLUMNS),
    }


# --------------------------------------------------------------------------- #
# Loading cached data
# --------------------------------------------------------------------------- #
def _load_dir(directory: Path, columns: list[str], keys: list[str]) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def load_game_logs() -> pd.DataFrame:
    """Load every cached pitcher appearance (all chunks, de-duplicated)."""
    return _load_dir(GAME_LOG_DIR, PITCHER_LOG_COLUMNS, ["game_pk", "pitcher_id"])


def load_team_games() -> pd.DataFrame:
    """Load every cached team-batting game row."""
    return _load_dir(TEAM_GAME_DIR, TEAM_GAME_COLUMNS, ["game_pk", "team_id"])


def load_statcast_summaries() -> pd.DataFrame:
    """Load every cached Statcast pitcher-game summary."""
    return _load_dir(PITCHER_AGG_DIR, PITCHER_AGG_COLUMNS, ["game_pk", "pitcher_id"])


def load_batter_games() -> pd.DataFrame:
    """Load every cached Statcast batter-game summary."""
    return _load_dir(BATTER_AGG_DIR, BATTER_AGG_COLUMNS, ["game_pk", "batter_id"])


def load_games() -> pd.DataFrame:
    """Load every cached game-info row (venue, weather, umpire)."""
    return _load_dir(GAMES_DIR, GAME_COLUMNS, ["game_pk"])


def load_lineups() -> pd.DataFrame:
    """Load every cached starting-lineup row."""
    return _load_dir(LINEUP_DIR, LINEUP_COLUMNS, ["game_pk", "team_id", "batter_id"])


def load_raw() -> dict[str, pd.DataFrame]:
    """Load every cached table the feature builder needs, keyed by name."""
    return {
        "game_logs": load_game_logs(),
        "statcast": load_statcast_summaries(),
        "team_games": load_team_games(),
        "batter_games": load_batter_games(),
        "games": load_games(),
        "lineups": load_lineups(),
    }


def last_cached_date() -> date | None:
    """Return the latest game date present in the cached game logs, if any."""
    logs = load_game_logs()
    if logs.empty:
        return None
    return date.fromisoformat(str(logs["game_date"].max())[:10])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
GAME_SIDE_DIRS = (GAME_LOG_DIR, TEAM_GAME_DIR, GAMES_DIR, LINEUP_DIR)
STATCAST_SIDE_DIRS = (PITCHER_AGG_DIR, BATTER_AGG_DIR)


def chunk_done(directories: tuple[Path, ...], start: date, end: date) -> bool:
    """A chunk is cached only when every table it produces exists.

    This also upgrades caches made by an earlier version of this script: a
    chunk missing the newer lineup/game-info tables is fetched again, and
    Statcast summaries are rebuilt from the cached raw pitches without
    downloading them again.
    """
    return all(chunk_path(d, start, end).exists() for d in directories)


def fetch_range(session: requests.Session, start: date, end: date, keep_raw: bool,
                game_types: str = REGULAR_SEASON) -> tuple[int, int]:
    """Fetch every missing chunk in [start, end]. Returns (chunks fetched, chunks failed)."""
    fetched = failed = 0
    for c_start, c_end in date_chunks(start, end):
        for directories, fn in (
            (GAME_SIDE_DIRS, lambda s=c_start, e=c_end: fetch_game_log_chunk(session, s, e, game_types)),
            (STATCAST_SIDE_DIRS, lambda s=c_start, e=c_end: fetch_statcast_chunk(s, e, keep_raw)),
        ):
            if chunk_done(directories, c_start, c_end):
                continue
            if fn():
                fetched += 1
            else:
                failed += 1
    return fetched, failed


def resummarize_statcast() -> int:
    """Rebuild every Statcast summary from the cached raw pitches (no download).

    Used after the summary columns change (e.g. called strikes were added).
    Returns the number of chunks rebuilt.
    """
    rebuilt = 0
    for raw_path in sorted(STATCAST_RAW_DIR.glob("*.parquet")):
        pitcher, batter = summarize_statcast(pd.read_parquet(raw_path))
        batter.to_parquet(BATTER_AGG_DIR / raw_path.name, index=False)
        pitcher.to_parquet(PITCHER_AGG_DIR / raw_path.name, index=False)
        rebuilt += 1
    log.info("Rebuilt %d Statcast summary chunks from cached raw pitches", rebuilt)
    return rebuilt


def retry_missing_statcast(keep_raw: bool) -> int:
    """Fetch Statcast for any cached game-log chunk whose Statcast chunk failed earlier.

    Returns the number of chunks that still failed.
    """
    failed = 0
    for path in sorted(GAME_LOG_DIR.glob("*.parquet")):
        start, end = (date.fromisoformat(p) for p in path.stem.split("_"))
        if not chunk_done(STATCAST_SIDE_DIRS, start, end):
            failed += not fetch_statcast_chunk(start, end, keep_raw)
    return failed


def write_github_output(**values) -> None:
    """Expose values to later GitHub Actions steps when running in Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as fh:
            for k, v in values.items():
                fh.write(f"{k}={v}\n")


def season_parts(session: requests.Session, season: int, postseason: bool) -> list[tuple[date, date, str]]:
    """The (start, end, game types) windows to fetch for a season."""
    parts = [(*season_window(session, season), REGULAR_SEASON)]
    window = postseason_window(session, season) if postseason else None
    if window:
        parts.append((*window, POSTSEASON))
    return parts


def run(seasons: list[int], incremental: bool, keep_raw: bool, today: date | None = None,
        postseason: bool = True) -> int:
    """Fetch data and return the number of new pitcher appearances cached.

    Postseason games are fetched in their own date range after each regular
    season, so the regular-season chunks stay exactly as before.
    """
    session = make_session()
    today = today or date.today()
    yesterday = today - timedelta(days=1)
    before = len(load_game_logs())
    failed_total = 0

    if incremental:
        last = last_cached_date()
        if last is None:
            log.error("No cached data yet; run a full fetch first")
            return 0
        for season in range(last.year, today.year + 1):
            for p_start, p_end, types in season_parts(session, season, postseason):
                start = max(p_start, last + timedelta(days=1))
                end = min(p_end, yesterday)
                if start > end:
                    continue
                log.info("Incremental fetch %s..%s (game types %s)", start, end, types)
                _, failed = fetch_range(session, start, end, keep_raw, types)
                failed_total += failed
        failed_total += retry_missing_statcast(keep_raw)
    else:
        for season in seasons:
            for p_start, p_end, types in season_parts(session, season, postseason):
                end = min(p_end, yesterday)
                if p_start > end:
                    continue
                log.info("Season %s: %s..%s (game types %s)", season, p_start, end, types)
                _, failed = fetch_range(session, p_start, end, keep_raw, types)
                failed_total += failed

    new_rows = len(load_game_logs()) - before
    log.info("Done: %d new pitcher appearances, %d failed chunks", new_rows, failed_total)
    write_github_output(new_rows=new_rows, failed_chunks=failed_total)
    write_manifest()
    return new_rows


def write_manifest() -> None:
    """Write a small JSON summary of what is cached (handy for README and CI logs)."""
    logs = load_game_logs()
    manifest = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "pitcher_appearances": int(len(logs)),
        "starts": int(logs["started"].sum()) if not logs.empty else 0,
        "first_date": str(logs["game_date"].min()) if not logs.empty else None,
        "last_date": str(logs["game_date"].max()) if not logs.empty else None,
        "statcast_pitcher_games": int(len(load_statcast_summaries())),
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--incremental", action="store_true",
                        help="fetch only games after the last cached date")
    parser.add_argument("--no-keep-raw", action="store_true",
                        help="don't save raw Statcast pitches (saves ~1 GB; used in CI)")
    parser.add_argument("--no-postseason", action="store_true", help="fetch regular-season games only")
    parser.add_argument("--resummarize", action="store_true",
                        help="rebuild Statcast summaries from cached raw pitches, then exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    if args.resummarize:
        resummarize_statcast()
        return 0

    new_rows = run(args.seasons, args.incremental, keep_raw=not args.no_keep_raw,
                   postseason=not args.no_postseason)
    if args.incremental and new_rows == 0:
        print("No new data since the last cached date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

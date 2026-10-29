"""How strikeouts happen: two-strike pitch choice, chases, and tunneling.

Reads the raw Statcast pitch cache (``data/raw/statcast``) and writes
``reports/chase_patterns.md`` with:

1. Every count: how often pitchers go out of the zone, what they throw, and how
   often batters chase.
2. Two strikes by pitch family and pitch type: chase, whiff, foul, called and
   swinging strike three.
3. How batters respond to two-strike pitches out of the zone: by how far off
   the zone the pitch is, and by the batter's own chase tendency.
4. Tunneling: for a two-strike pitch out of the zone, how far it was from the
   previous pitch at the batter's decision point (about 24 ft in front of the
   plate) versus at the plate, and whether tighter tunnels draw more chases.
5. Pitchers: does a pitcher's average tunnel predict the chases he gets?

Out of the zone = Statcast ``zone`` 11-14. Chase = swing at an out-of-zone pitch.

Usage::

    python src/analyze_chase.py               # 2024 on
    python src/analyze_chase.py --since 2021-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_data import (BREAKING_TYPES, FASTBALL_CLASS_TYPES, OFFSPEED_TYPES, STATCAST_RAW_DIR, STRIKEOUT_EVENTS,
                        SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "chase_patterns.md"
COLS = ["game_pk", "game_date", "game_type", "pitcher", "batter", "at_bat_number", "pitch_number", "balls", "strikes",
        "zone", "description", "events", "pitch_type", "pitch_name", "stand", "p_throws", "plate_x", "plate_z",
        "sz_top", "sz_bot", "release_pos_x", "release_pos_z", "release_extension", "vx0", "vy0", "vz0",
        "ax", "ay", "az", "release_speed"]
BALL_R = 0.121                  # ft
HALF_PLATE = 0.708 + BALL_R     # ball touches the zone edge
PLATE_Y = 17 / 12               # front of the plate, Statcast y (ft)
DECISION_Y = PLATE_Y + 23.8     # roughly where a hitter has to commit (~0.17 s out)


def load(since: str) -> pd.DataFrame:
    frames = []
    for path in sorted(STATCAST_RAW_DIR.glob("*.parquet")):
        if path.stem.split("_")[-1] < since:
            continue
        try:
            frames.append(pd.read_parquet(path, columns=COLS))
        except Exception:  # noqa: BLE001
            frames.append(pd.read_parquet(path).reindex(columns=COLS))
    if not frames:
        raise RuntimeError(f"No raw Statcast chunks in {STATCAST_RAW_DIR} since {since}.")
    return prepare(pd.concat(frames, ignore_index=True))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Flag zone, swings, outcomes and pitch family; add inches off the zone."""
    df = df[df["game_type"].isin(["R", "F", "D", "L", "W"]) | df["game_type"].isna()]
    zone = pd.to_numeric(df["zone"], errors="coerce")
    df = df[zone.notna()].copy()
    df["oz"] = pd.to_numeric(df["zone"]).between(11, 14)
    df["swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["foul"] = df["description"].isin({"foul", "foul_bunt"})
    df["called"] = df["description"] == "called_strike"
    df["in_play"] = df["description"].str.startswith("hit_into_play")
    df["ball"] = ~(df["swing"] | df["called"]) & ~df["description"].str.contains("hit_by_pitch", na=False)
    df["k"] = df["events"].isin(STRIKEOUT_EVENTS)
    df["fam"] = np.select([df["pitch_type"].isin(FASTBALL_CLASS_TYPES), df["pitch_type"].isin(BREAKING_TYPES),
                           df["pitch_type"].isin(OFFSPEED_TYPES)], ["fastball", "breaking", "offspeed"], "other")
    df["count"] = df["balls"].astype("Int64").astype(str) + "-" + df["strikes"].astype("Int64").astype(str)
    # How far outside the zone (inches), using each batter's own top and bottom.
    dx = (df["plate_x"].abs() - HALF_PLATE).clip(lower=0)
    dz = np.maximum(df["sz_bot"] - BALL_R - df["plate_z"], df["plate_z"] - df["sz_top"] - BALL_R).clip(lower=0)
    df["off_in"] = np.hypot(dx, dz) * 12
    df["season"] = pd.to_datetime(df["game_date"]).dt.year
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def pct(x) -> str:
    return "" if pd.isna(x) else f"{100 * x:.1f}%"


def table(df: pd.DataFrame, fmt: dict) -> list[str]:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in [df.index.name or ""] + cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, r in df.iterrows():
        out.append("| " + " | ".join([str(idx)] + [fmt.get(c, str)(r[c]) for c in cols]) + " |")
    return out


def rates(g: pd.DataFrame) -> pd.Series:
    oz = g["oz"]
    return pd.Series({
        "pitches": len(g),
        "out of zone": oz.mean(),
        "chase": g.loc[oz, "swing"].mean(),
        "whiff per chase": g.loc[oz & g["swing"], "whiff"].mean(),
        "zone swing": g.loc[~oz, "swing"].mean(),
        "fastball": (g["fam"] == "fastball").mean(),
        "breaking": (g["fam"] == "breaking").mean(),
        "offspeed": (g["fam"] == "offspeed").mean(),
        "strikeout on pitch": g["k"].mean(),
    })


FMT = {"pitches": lambda v: f"{int(v):,}", "n": lambda v: f"{int(v):,}", "r": lambda v: f"{v:+.3f}"}


def by_count(df: pd.DataFrame) -> list[str]:
    order = [f"{b}-{s}" for s in range(3) for b in range(4)]
    t = df.groupby("count").apply(rates).reindex(order)
    t.index.name = "Count"
    return table(t, {**FMT, **{c: pct for c in t.columns if c != "pitches"}})


def two_strike(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    two = df[df["strikes"] == 2]

    def res(g):
        oz = g["oz"]
        return pd.Series({
            "pitches": len(g), "share of 2-strike pitches": len(g) / len(two), "out of zone": oz.mean(),
            "chase": g.loc[oz, "swing"].mean(), "whiff per chase": g.loc[oz & g["swing"], "whiff"].mean(),
            "foul": g["foul"].mean(), "called K": (g["k"] & g["called"]).mean(),
            "swinging K": (g["k"] & g["whiff"]).mean(), "in play": g["in_play"].mean(), "ball": g["ball"].mean(),
        })
    fam = two.groupby("fam").apply(res).sort_values("pitches", ascending=False)
    fam.index.name = "Family"
    typ = two.groupby("pitch_name").apply(res)
    typ = typ[typ["pitches"] >= 2000].sort_values("pitches", ascending=False)
    typ.index.name = "Pitch"
    f = {**FMT, **{c: pct for c in fam.columns if c != "pitches"}}
    return table(fam, f), table(typ, f)


def batter_response(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    two_oz = df[(df["strikes"] == 2) & df["oz"]].copy()
    bins = [0, 2, 4, 6, 9, 12, np.inf]
    labels = ["0-2 in", "2-4 in", "4-6 in", "6-9 in", "9-12 in", "12+ in"]
    two_oz["off"] = pd.cut(two_oz["off_in"], bins, labels=labels, right=False)

    def res(g):
        return pd.Series({"pitches": len(g), "chase": g["swing"].mean(), "whiff per chase": g.loc[g["swing"], "whiff"].mean(),
                          "foul": g["foul"].mean(), "strikeout": g["k"].mean(), "in play": g["in_play"].mean()})
    off = two_oz.groupby("off", observed=True).apply(res)
    off.index.name = "Distance off the zone"
    # Batter chase tendency: his O-Swing% on 0- and 1-strike pitches in the same season.
    early = df[(df["strikes"] < 2) & df["oz"]].groupby(["batter", "season"])["swing"].agg(["mean", "size"])
    early = early[early["size"] >= 150]["mean"].rename("early_chase")
    two_oz = two_oz.join(early, on=["batter", "season"])
    two_oz = two_oz.dropna(subset=["early_chase"])
    if two_oz["early_chase"].nunique() < 3:
        return table(off, {**FMT, "chase": pct, "whiff per chase": pct, "foul": pct, "strikeout": pct, "in play": pct}), \
            ["Not enough batters with 150+ early-count pitches out of the zone."]
    two_oz["tend"] = pd.qcut(two_oz["early_chase"], 3, labels=["low chasers", "middle", "high chasers"])
    tend = two_oz.groupby("tend", observed=True).apply(res)
    tend.index.name = "Batter (O-Swing% before 2 strikes)"
    f = {**FMT, "chase": pct, "whiff per chase": pct, "foul": pct, "strikeout": pct, "in play": pct}
    return table(off, f), table(tend, f)


def positions_at(df: pd.DataFrame, y_target: float) -> tuple[np.ndarray, np.ndarray]:
    """Ball x, z when it reaches ``y_target`` (Statcast's constant-acceleration fit, measured from y = 50 ft)."""
    vx0, vy0, vz0, ax, ay, az = (df[c].to_numpy(float) for c in ["vx0", "vy0", "vz0", "ax", "ay", "az"])
    # Position at y = 50 from the release point (release is at y = 60.5 - extension).
    y_rel = 60.5 - df["release_extension"].fillna(6.0).to_numpy(float)

    def t_at(y):
        disc = np.sqrt(np.maximum(vy0 ** 2 - 2 * ay * (50 - y), 0))
        return (-vy0 - disc) / ay
    t_r = t_at(y_rel)
    x50 = df["release_pos_x"].to_numpy(float) - vx0 * t_r - 0.5 * ax * t_r ** 2
    z50 = df["release_pos_z"].to_numpy(float) - vz0 * t_r - 0.5 * az * t_r ** 2
    t = t_at(y_target)
    return x50 + vx0 * t + 0.5 * ax * t ** 2, z50 + vz0 * t + 0.5 * az * t ** 2


def tunneling(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    d = df.dropna(subset=["vx0", "vy0", "vz0", "ax", "ay", "az", "release_pos_x", "release_pos_z", "plate_x", "plate_z"]).copy()
    d["tx"], d["tz"] = positions_at(d, DECISION_Y)
    same_pa = (d["game_pk"].eq(d["game_pk"].shift()) & d["at_bat_number"].eq(d["at_bat_number"].shift()))
    for c in ["tx", "tz", "plate_x", "plate_z", "release_pos_x", "release_pos_z", "pitch_type", "fam"]:
        d["prev_" + c] = d[c].shift().where(same_pa)
    pairs = d[same_pa & (d["pitch_type"] != d["prev_pitch_type"])].copy()
    pairs["tunnel_in"] = np.hypot(pairs["tx"] - pairs["prev_tx"], pairs["tz"] - pairs["prev_tz"]) * 12
    pairs["plate_in"] = np.hypot(pairs["plate_x"] - pairs["prev_plate_x"], pairs["plate_z"] - pairs["prev_plate_z"]) * 12
    pairs["release_in"] = np.hypot(pairs["release_pos_x"] - pairs["prev_release_pos_x"],
                                   pairs["release_pos_z"] - pairs["prev_release_pos_z"]) * 12
    two_oz = pairs[(pairs["strikes"] == 2) & pairs["oz"]].copy()
    two_oz["tunnel"] = pd.qcut(two_oz["tunnel_in"], 5, labels=["tightest 20%", "2nd", "3rd", "4th", "widest 20%"])
    two_oz["how_far"] = np.where(two_oz["off_in"] < 4, "just off (under 4 in)", "well off (4 in or more)")

    def res(g):
        return pd.Series({"pitches": len(g), "median tunnel gap": g["tunnel_in"].median(),
                          "median plate gap": g["plate_in"].median(), "chase": g["swing"].mean(),
                          "whiff per chase": g.loc[g["swing"], "whiff"].mean(), "strikeout": g["k"].mean()})
    t = two_oz.groupby(["how_far", "tunnel"], observed=True).apply(res)
    t.index = [f"{a}, {b}" for a, b in t.index]
    t.index.name = "Two-strike pitch out of the zone, tunnel gap to the previous pitch"
    f = {**FMT, "median tunnel gap": lambda v: f"{v:.1f} in", "median plate gap": lambda v: f"{v:.1f} in",
         "chase": pct, "whiff per chase": pct, "strikeout": pct}

    # Logistic-free check: chase rate vs tunnel gap with plate distance held in narrow bins.
    two_oz["off_bin"] = pd.cut(two_oz["off_in"], [0, 2, 4, 6, 9, np.inf], right=False)
    held = two_oz.groupby("off_bin", observed=True).apply(
        lambda g: pd.Series({"n": len(g), "r": np.corrcoef(g["tunnel_in"], g["swing"].astype(float))[0, 1]}))
    held.index = held.index.astype(str); held.index.name = "Inches off the zone"
    held_md = table(held, FMT)

    # Pitchers: average fastball -> breaking/offspeed tunnel gap vs two-strike chase rate he gets.
    fb_off = pairs[(pairs["prev_fam"] == "fastball") & pairs["fam"].isin(["breaking", "offspeed"])]
    per = fb_off.groupby("pitcher").agg(n=("tunnel_in", "size"), tunnel=("tunnel_in", "median"),
                                        release=("release_in", "median"))
    chase2 = d[(d["strikes"] == 2) & d["oz"]].groupby("pitcher")["swing"].agg(["mean", "size"])
    per = per.join(chase2.rename(columns={"mean": "chase2", "size": "n2"}))
    per = per[(per["n"] >= 300) & (per["n2"] >= 200)]
    lines = []
    if len(per) >= 20:
        r1 = np.corrcoef(per["tunnel"], per["chase2"])[0, 1]
        r2 = np.corrcoef(per["release"], per["chase2"])[0, 1]
        slope = np.polyfit(per["tunnel"], per["chase2"], 1)[0]
        lines = [f"- {len(per)} pitchers with 300+ fastball-then-breaking/offspeed pairs and 200+ two-strike pitches out of the zone.",
                 f"- Correlation between his median tunnel gap and the two-strike chase rate he gets: r = {r1:+.3f} "
                 f"({100 * slope:+.2f} points of chase rate per extra inch of gap).",
                 f"- Correlation between his median release-point gap and that chase rate: r = {r2:+.3f}."]
    return table(t, f), held_md, lines


def zero_two(df: pd.DataFrame) -> list[str]:
    z = df[df["count"] == "0-2"]
    per = z.groupby("pitcher").agg(n=("oz", "size"), oz=("oz", "mean"),
                                   brk=("fam", lambda s: (s == "breaking").mean()), k=("k", "mean"))
    per = per[per["n"] >= 100]
    q = per[["oz", "brk", "k"]].quantile([0.1, 0.5, 0.9])
    q.index = ["10th percentile", "median", "90th percentile"]; q.index.name = "Pitchers with 100+ 0-2 pitches"
    q.columns = ["out of zone", "breaking balls", "strikeout on pitch"]
    r = np.corrcoef(per["oz"], per["k"])[0, 1] if len(per) > 20 else np.nan
    return table(q, {c: pct for c in q.columns}) + ["", f"Across {len(per)} pitchers, going out of the zone more often on 0-2 "
                                                     f"and strikeouts per 0-2 pitch correlate at r = {r:+.3f}."]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2024-01-01", help="first chunk end date to include (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    df = load(args.since)
    print(f"{len(df):,} pitches since {args.since}")
    fam_md, typ_md = two_strike(df)
    off_md, tend_md = batter_response(df)
    tun_md, held_md, per_lines = tunneling(df)
    lines = [
        "# How strikeouts happen: two strikes, chases and tunneling", "",
        f"Statcast pitches from {df['game_date'].min()} to {df['game_date'].max()} ({len(df):,} pitches, regular season "
        "and postseason). Out of the zone = Statcast zone 11-14. Chase = a swing at a pitch out of the zone.", "",
        "## 1. Every count", "", *by_count(df), "",
        "## 2. 0-2 counts, pitcher by pitcher", "", *zero_two(df), "",
        "## 3. Two strikes, by pitch family", "", *fam_md, "",
        "## 4. Two strikes, by pitch type (2,000+ pitches)", "", *typ_md, "",
        "## 5. How batters respond to two-strike pitches out of the zone", "", "By how far off the zone:", "", *off_md, "",
        "By the batter's own chase tendency before two strikes (thirds):", "", *tend_md, "",
        "## 6. Tunneling", "",
        "Tunnel gap = distance between this pitch and the previous one in the at-bat when both are about 24 ft in "
        "front of the plate, where a hitter has to commit. Plate gap = their distance at the plate. Only pairs of "
        "different pitch types.", "", *tun_md, "",
        "Correlation between tunnel gap and chasing, with distance off the zone held in narrow bins "
        "(negative = tighter tunnels draw more chases):", "", *held_md, "",
        "### Pitchers", "", *per_lines, "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

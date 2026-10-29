"""Descriptive strikeout patterns: rest, home/away, platoon, pitch type x velocity,
lineups, weather, day/night, parks, umpires, and opponent form.

This is exploratory analysis, separate from the model. By default it uses only
the training seasons (every season before the test season) so the patterns you
look at can't steer the model toward the held-out season. ``--all-seasons``
includes everything.

Rates are strikeouts per batter faced (K/BF) unless noted, because K per start
mixes strikeout skill with how long the starter pitches. Every table shows the
sample size and a 95% interval so small cells are easy to spot.

Writes ``reports/patterns.md`` and charts in ``reports/figures/patterns_*.png``.

Usage::

    python src/analyze_patterns.py
    python src/analyze_patterns.py --all-seasons
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_features import FEATURES, FEATURES_PATH  # noqa: E402
from fetch_data import PITCH_CLASSES, load_raw  # noqa: E402
from train import MIN_TEST_STARTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "patterns.md"
FIG_DIR = ROOT / "reports" / "figures"

CLASS_LABELS = {
    "fb_lo": "Fastball < 93 mph", "fb_mid": "Fastball 93-95.9", "fb_hi": "Fastball 96+",
    "brk": "Breaking ball", "os": "Offspeed",
}
BLUE, ORANGE, RED, NEUTRAL = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def rate_table(df: pd.DataFrame, by, num: str = "strikeouts", den: str = "batters_faced") -> pd.DataFrame:
    """Group rate num/den with counts and a normal-approximation 95% interval."""
    g = df.groupby(by, observed=True).agg(n=(num, "size"), num=(num, "sum"), den=(den, "sum"),
                                          k_per_start=(num, "mean"), bf_per_start=(den, "mean"))
    g["rate"] = g["num"] / g["den"]
    se = np.sqrt(g["rate"] * (1 - g["rate"]) / g["den"])
    g["ci_low"], g["ci_high"] = g["rate"] - 1.96 * se, g["rate"] + 1.96 * se
    return g


def md(df: pd.DataFrame, cols: dict[str, str], pct: set[str] = frozenset(), digits: int = 2) -> str:
    """Render selected columns as a markdown table; ``pct`` columns shown as percentages."""
    out = df.reset_index()
    first = out.columns[0] if len(df.index.names) == 1 else None
    keep = ([first] if first else list(df.index.names)) + list(cols)
    out = out[keep]
    header = "| " + " | ".join([*(df.index.names if not first else [first]), *cols.values()]) + " |"
    sep = "|" + "---|" * (len(keep) - len(cols)) + "---:|" * len(cols)
    lines = [header, sep]
    for _, row in out.iterrows():
        cells = [str(row[c]) for c in keep[: len(keep) - len(cols)]]
        for c in cols:
            v = row[c]
            if isinstance(v, str):
                cells.append(v)
            elif pd.isna(v):
                cells.append("")
            elif c in pct:
                cells.append(f"{100 * v:.1f}%")
            elif c == "n" or float(v).is_integer():
                cells.append(f"{int(v):,}")
            else:
                cells.append(f"{v:.{digits}f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def rate_md(t: pd.DataFrame) -> str:
    """Standard rate-table rendering."""
    t = t.assign(ci=[f"{100 * lo:.1f}-{100 * hi:.1f}%" for lo, hi in zip(t["ci_low"], t["ci_high"], strict=True)])
    return md(t, {"n": "Starts", "rate": "K/BF", "ci": "95% CI", "k_per_start": "K/start",
                  "bf_per_start": "BF/start"}, pct={"rate"})


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Shared recessive axis styling."""
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)


def bar_with_ci(t: pd.DataFrame, title: str, path: Path, xlabel: str = "") -> None:
    """Bar chart of K/BF with 95% interval whiskers and a league-average reference line."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(t))
    ax.bar(x, 100 * t["rate"], color=BLUE, width=0.6)
    ax.errorbar(x, 100 * t["rate"], yerr=[100 * (t["rate"] - t["ci_low"]), 100 * (t["ci_high"] - t["rate"])],
                fmt="none", ecolor=INK_MUTED, lw=1, capsize=3)
    overall = t["num"].sum() / t["den"].sum()
    ax.axhline(100 * overall, color=INK_MUTED, lw=1, ls="--")
    ax.text(len(t) - 0.5, 100 * overall, f" all: {100 * overall:.1f}%", va="bottom", ha="right", color=INK_MUTED)
    ax.set_xticks(x, [str(i) for i in t.index])
    lo = 100 * t["ci_low"].min()
    ax.set_ylim(max(0, lo - 2), 100 * t["ci_high"].max() + 1)
    style(ax, title, xlabel, "Strikeouts per batter faced (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def rest_days(f: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """K/BF and workload by days of rest."""
    bucket = pd.cut(f["rest_days"], [0, 4, 5, 6, 9, 29, 30], labels=["<=4", "5", "6", "7-9", "10-29", "30+"])
    bucket = bucket.cat.add_categories(["first start"]).fillna("first start")
    bucket = bucket.where(f["first_start_of_season"] == 0, "first start")
    t = rate_table(f.assign(rest=bucket), "rest")
    return rate_md(t), t


def platoon(sc: pd.DataFrame) -> str:
    """League K per PA by pitcher hand x batter side."""
    rows = []
    for hand in ("R", "L"):
        s = sc[sc["p_throws"] == hand]
        for side in ("l", "r"):
            pa, k = s[f"pa_vs_{side}"].sum(), s[f"k_vs_{side}"].sum()
            rows.append({"matchup": f"{hand}HP vs {side.upper()}HB", "n": pa, "rate": k / pa if pa else np.nan})
    return md(pd.DataFrame(rows).set_index("matchup"), {"n": "PA", "rate": "K/PA"}, pct={"rate"})


def class_whiffs(sc: pd.DataFrame) -> pd.DataFrame:
    """League usage and whiff rate (whiffs per swing) by pitch class."""
    rows = []
    total = sum(sc[f"n_{c}"].sum() for c in PITCH_CLASSES)
    for c in PITCH_CLASSES:
        sw, wh, n = sc[f"sw_{c}"].sum(), sc[f"wh_{c}"].sum(), sc[f"n_{c}"].sum()
        rows.append({"pitch": CLASS_LABELS[c], "n": n, "usage": n / total, "whiff": wh / sw if sw else np.nan,
                     "swing": sw / n if n else np.nan})
    return pd.DataFrame(rows).set_index("pitch")


def team_class_whiffs(bg: pd.DataFrame, season: int) -> pd.DataFrame:
    """Each batting team's whiff rate per pitch class in one season, minus the league rate (points)."""
    s = bg[bg["season"] == season]
    sw = s.groupby("bat_team")[[f"sw_{c}" for c in PITCH_CLASSES]].sum()
    wh = s.groupby("bat_team")[[f"wh_{c}" for c in PITCH_CLASSES]].sum()
    rate = pd.DataFrame({c: wh[f"wh_{c}"] / sw[f"sw_{c}"] for c in PITCH_CLASSES})
    league = pd.Series({c: wh[f"wh_{c}"].sum() / sw[f"sw_{c}"].sum() for c in PITCH_CLASSES})
    diff = 100 * (rate - league)
    return diff.loc[diff.mean(axis=1).sort_values(ascending=False).index]


def plot_team_heatmap(diff: pd.DataFrame, season: int, path: Path) -> None:
    """Diverging heatmap: red = team whiffs more than league on that pitch class, blue = less."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    lim = max(1.0, float(np.nanmax(np.abs(diff.to_numpy()))))
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(diff) + 1.8))
    im = ax.imshow(diff.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(len(diff.columns)), [CLASS_LABELS[c] for c in diff.columns], rotation=20, ha="right")
    ax.set_yticks(range(len(diff)), diff.index)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            v = diff.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_title(f"Lineup whiff rate vs. league by pitch class, {season} (points)", loc="left", color=INK)
    ax.tick_params(colors=INK_MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Whiff% minus league (more whiffs = red)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_whiffs(t: pd.DataFrame, path: Path) -> None:
    """Whiff rate per swing for each pitch class."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(t))
    ax.bar(x, 100 * t["whiff"], color=BLUE, width=0.6)
    for xi, v in zip(x, t["whiff"], strict=True):
        ax.text(xi, 100 * v, f"{100 * v:.1f}%", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(x, t.index)
    style(ax, "Whiffs per swing by pitch class and fastball velocity", "", "Whiff rate (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_correlations(f: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of every model feature with K and with K/BF."""
    kbf = f["strikeouts"] / f["batters_faced"]
    rows = []
    for col in FEATURES:
        x = f[col]
        ok = x.notna()
        if ok.sum() < 30 or x[ok].nunique() < 2:
            continue
        rows.append({"feature": col, "n": int(ok.sum()),
                     "rho_k": x[ok].rank().corr(f.loc[ok, "strikeouts"].rank()),
                     "rho_kbf": x[ok].rank().corr(kbf[ok].rank())})
    out = pd.DataFrame(rows).set_index("feature")
    return out.reindex(out["rho_k"].abs().sort_values(ascending=False).index)


def quantile_rates(f: pd.DataFrame, col: str, q: int = 5) -> pd.DataFrame:
    """K/BF by quantile bin of a feature."""
    ok = f[col].notna()
    bins = pd.qcut(f.loc[ok, col], q, duplicates="drop")
    labels = bins.cat.categories
    names = [f"Q{i + 1}: {iv.left:.3f} to {iv.right:.3f}" for i, iv in enumerate(labels)]
    bins = bins.cat.rename_categories(names)
    return rate_table(f.loc[ok].assign(bin=bins), "bin")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def analysis_window(f: pd.DataFrame, all_seasons: bool) -> list[int]:
    """Seasons to analyze: all, or every season before the newest full season."""
    counts = f.groupby("season").size()
    seasons = sorted(int(s) for s in counts.index)
    if all_seasons:
        return seasons
    full = counts[counts >= MIN_TEST_STARTS]
    test = int(full.index.max()) if len(full) >= 2 else None
    return [s for s in seasons if s != test]


def build_report(all_seasons: bool = False) -> str:
    """Run every analysis and return the markdown report (also writes charts)."""
    f = pd.read_parquet(FEATURES_PATH)
    seasons = analysis_window(f, all_seasons)
    f = f[f["season"].isin(seasons)].copy()
    raw = load_raw()
    games = raw["games"].copy()
    sc = raw["statcast"].copy()
    bg = raw["batter_games"].copy()
    for df in (sc, bg):
        df["season"] = pd.to_datetime(df["game_date"]).dt.year
    sc, bg = sc[sc["season"].isin(seasons)], bg[bg["season"].isin(seasons)]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    parts = [
        "# Strikeout patterns",
        "",
        f"Starts from seasons {seasons[0]}-{seasons[-1]} ({len(f):,} starts). Rates are strikeouts per batter "
        "faced (K/BF) with 95% intervals; K/start also reflects how deep starters go. These are raw "
        "associations: a pattern here can be explained by something else (for example, better pitchers "
        "may get more rest), which is why the model, not this table, decides what matters.",
        "",
    ]

    text, t = rest_days(f)
    bar_with_ci(t, "Strikeout rate by days of rest", FIG_DIR / "patterns_rest.png", "Days since previous start")
    parts += ["## Days of rest", "", text, "![Rest](figures/patterns_rest.png)", ""]

    t = rate_table(f.assign(where=f["is_home"].map({1.0: "home", 0.0: "away"})), "where")
    parts += ["## Home vs. away", "", rate_md(t), ""]

    t = rate_table(f.assign(time=f["day_game"].map({1.0: "day", 0.0: "night"})), "time")
    parts += ["## Day vs. night", "", rate_md(t), ""]

    if not sc.empty:
        parts += ["## Platoon: pitcher hand vs. batter side", "", platoon(sc), ""]
    if f["lineup_platoon_adv_share"].notna().any():
        parts += ["### Starts by share of the lineup with the platoon advantage", "",
                  rate_md(quantile_rates(f, "lineup_platoon_adv_share")), ""]
        parts += ["### Starts by the opposing lineup's prior K rate vs. this pitcher's hand", "",
                  rate_md(quantile_rates(f, "lineup_k_rate_vs_hand")), ""]

    if not sc.empty:
        t = class_whiffs(sc)
        plot_class_whiffs(t, FIG_DIR / "patterns_pitch_class.png")
        parts += ["## Pitch type and velocity", "",
                  md(t, {"n": "Pitches", "usage": "Usage", "swing": "Swing rate", "whiff": "Whiffs/swing"},
                     pct={"usage", "swing", "whiff"}),
                  "![Pitch classes](figures/patterns_pitch_class.png)", ""]
    if not bg.empty:
        season = seasons[-1]
        diff = team_class_whiffs(bg, season)
        plot_team_heatmap(diff, season, FIG_DIR / "patterns_team_pitch.png")
        top = {c: diff[c].idxmax() for c in PITCH_CLASSES}
        parts += [f"## Which lineups strike out on which pitches ({season})", "",
                  "Each cell is the team's whiffs per swing on that pitch class minus the league rate, in "
                  "percentage points. Teams are sorted by their average across classes.", "",
                  "Most whiff-prone lineup per class: "
                  + "; ".join(f"{CLASS_LABELS[c]}: {t}" for c, t in top.items()) + ".", "",
                  "![Team x pitch](figures/patterns_team_pitch.png)", ""]

    if f["temp_f"].notna().any():
        outdoor = f[f["roof_closed"] == 0]
        temp = pd.cut(outdoor["temp_f"], [-100, 55, 65, 75, 85, 200], labels=["<55", "55-64", "65-74", "75-84", "85+"])
        t = rate_table(outdoor.assign(temp=temp), "temp")
        bar_with_ci(t, "Strikeout rate by game-time temperature (open-air games)",
                    FIG_DIR / "patterns_temperature.png", "Temperature (F)")
        parts += ["## Weather", "", "### Temperature (open-air games)", "", rate_md(t),
                  "![Temperature](figures/patterns_temperature.png)", ""]
        wind = np.select([f["roof_closed"] == 1, f["wind_out_mph"] >= 8, f["wind_out_mph"] <= -8],
                         ["roof closed", "out 8+ mph", "in 8+ mph"], "calm or cross")
        parts += ["### Wind and roof", "", rate_md(rate_table(f.assign(wind=wind), "wind")), ""]

    if not games.empty:
        g = games[games["season"].isin(seasons)].merge(
            raw["team_games"].groupby("game_pk")[["bat_k", "bat_pa"]].sum(), on="game_pk")
        # Parks by venue ID (labelled with the latest name), so a renamed park is one row.
        latest = g.sort_values("game_date").groupby("venue_id")["venue_name"].last()
        g["park"] = g["venue_id"].map(latest)
        for key, label, minimum in (("park", "Park", 5000), ("hp_umpire_name", "Home plate umpire", 3000)):
            t = rate_table(g, key, "bat_k", "bat_pa")
            t = t[t["den"] >= minimum].sort_values("rate")
            league = g["bat_k"].sum() / g["bat_pa"].sum()
            t["vs_league"] = t["rate"] / league
            show = pd.concat([t.tail(5).iloc[::-1], t.head(5)])
            parts += [f"## {label}: highest and lowest strikeout rates", "",
                      f"All batters, both teams; minimum {minimum:,} PA. League K/PA {100 * league:.1f}%.", "",
                      md(show, {"den": "PA", "rate": "K/PA", "vs_league": "x league"}, pct={"rate"}), ""]

    parts += ["## Opponent recent form (team K rate over its last 10 games)", "",
              rate_md(quantile_rates(f, "opp_k_rate_last10")), ""]

    corr = feature_correlations(f)
    parts += ["## Every model feature: rank correlation with strikeouts", "",
              "Spearman correlation with K in the start and with K/BF. Workload features correlate with K "
              "but not K/BF; skill and matchup features correlate with both.", "",
              md(corr, {"n": "Starts", "rho_k": "rho with K", "rho_kbf": "rho with K/BF"}, digits=3), ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-seasons", action="store_true", help="include the held-out test season")
    args = parser.parse_args(argv)
    if not FEATURES_PATH.exists():
        print("No feature table. Run `python src/build_features.py` first.")
        return 1
    report = build_report(args.all_seasons)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"\nWrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

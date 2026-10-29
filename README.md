# Strikeout model

Predicts how many strikeouts an MLB starting pitcher will record in a single game, using only information available before the game starts, and measures how much of the game-to-game variance in strikeouts is predictable at all.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

The whole pipeline, end to end:

```bash
./run_pipeline.sh
```

That runs these five steps, which you can also run one at a time:

| Step | Command | What it does | Writes |
|---|---|---|---|
| 1 | `python src/fetch_data.py` | Downloads 2021-2026 regular-season and postseason game logs (MLB Stats API) and Statcast pitches (pybaseball), cached so reruns skip what's already there | `data/raw/` |
| 2 | `python src/build_features.py` | Builds one row per start with pre-game features, then runs the leakage audit | `data/processed/features.parquet` |
| 3 | `python src/train.py` | Fits the baseline, linear regression and tuned random forest on a time split as benchmarks, plus the leaky-feature sanity check (predictions use the linear + gradient boosting blend; see Models and evaluation) | `models/`, `metrics/latest.json` |
| 4 | `python src/evaluate.py` | Comparison table, plots, feature importance, noise analysis, what each feature group adds, and written interpretation | `reports/results.md`, `reports/figures/` |
| 5 | `python src/analyze_patterns.py` | Strikeout patterns by rest, home/away, platoon, pitch type and velocity, team lineups, weather, park, umpire and opponent form | `reports/patterns.md`, `reports/figures/patterns_*.png` |

The first fetch makes about 12,000 boxscore requests and downloads roughly 3.5 million Statcast pitches, so expect it to take a few hours. It saves each two-week chunk as soon as it finishes, so you can stop it and rerun at any time and it picks up where it left off. Requests that hit rate limits or server errors are retried with exponential backoff; a chunk with any request that still fails is not saved, and the next run retries it.

If you fetched with an earlier version of this project, just run `python src/fetch_data.py` again. It fetches boxscores again for the lineup, weather and umpire tables and rebuilds the Statcast summaries from your cached raw pitches without downloading them again. You can then delete the old `.parquet` files sitting directly in `data/raw/statcast_agg/`; the current ones live in its `pitcher/` and `batter/` subfolders.

Useful options:

```bash
python src/fetch_data.py --seasons 2024 2025   # just some seasons
python src/fetch_data.py --incremental         # only games after the last cached date
python src/fetch_data.py --no-postseason       # regular seasons only
python src/predict_upcoming.py --days 3        # predict the next three days' probable starters
python src/train.py --test-season 2025         # pin the held-out season
python src/evaluate.py --gate                  # exit 1 if the RF is worse than the baseline
pytest -q && ruff check .                      # tests and lint (no network needed)
```

## Upcoming starts

```bash
python src/fetch_data.py --incremental   # bring the cache up to yesterday
python src/predict_upcoming.py           # today and the next two days
```

This reads the MLB schedule's probable pitchers, builds each one's pre-game features from the cache, and predicts with a 50/50 blend of the linear regression and Poisson gradient boosting (`make_blend` in `src/train.py`, the best option in `python src/compare_models.py`), refit on every completed start. Posted lineups, the umpire and the weather are used when the boxscore has them; an opponent with no posted lineup gets its lineup from its previous game. A game under way or finished today is still predicted from pre-game information only. Results go to `data/processed/upcoming_predictions.parquet`.

Postseason games (wild card through World Series) are part of the data for every season, as history for each pitcher and as training rows; `--no-postseason` leaves them out.

## App

A small web app to browse predictions, after the pipeline has run:

```bash
streamlit run app.py
```

The first tab, **Upcoming starts**, shows the predictions above for any probable starter in the next few days, with a likely range and the pitcher's recent starts. In the second tab, pick a season, a pitcher and a start to see the model's pre-game prediction next to the actual strikeouts, what the model knew going in, and the pitcher's whole season. Each season from 2022 on is predicted by the same blend fit only on earlier seasons, so every number shown is out of sample. A third tab shows the test results and figures.

## Simulator

`sim/` is a static page that reads the JSON files next to it. Serve the folder and open it in a browser:

```bash
python3 -m http.server 8765 --directory sim
```

Then go to http://localhost:8765. Opening `index.html` straight from disk will not work, because browsers block the JSON fetches from `file://`.

## Project layout

```
src/fetch_data.py       download + cache game logs and Statcast summaries
src/build_features.py   pre-game feature table + leakage audit
src/train.py            time split, tuning, models, leaky-feature demo
src/evaluate.py         metrics, plots, feature importance, interpretation, quality gate
src/analyze_patterns.py descriptive strikeout patterns report
tests/                  feature, leakage and parsing tests on hand-made fixtures
.github/workflows/pipeline.yml   CI and weekly retraining
data/raw/               cached source data
data/processed/         feature table and test predictions (rebuilt each run)
```

## Data

* **Boxscores** from statsapi.mlb.com, one request per final regular-season game, give four tables: pitcher appearances (starter flag, outs, strikeouts, batters faced, pitch count, home/away, opponent), each team's batting strikeouts and plate appearances, each team's starting nine in batting order, and game info (venue, day/night, game-time temperature, sky condition, wind, home plate umpire).
* **Statcast** comes from `pybaseball.statcast` by date range. Raw pitches are reduced to two summaries. Per pitcher per game: throwing hand, fastball count and summed velocity (four-seamers and sinkers), swings, whiffs, PA and strikeouts against left- and right-handed batters, and pitches, swings and whiffs per pitch class. Per batter per game: batting side, PA and strikeouts against left- and right-handed pitchers, and swings and whiffs per pitch class.
* **Pitch classes** split fastballs (four-seam, sinker, cutter) by velocity, under 93 mph, 93-95.9 and 96+, and group the rest into breaking balls (sliders, sweepers, slurves, curves, knuckle curves) and offspeed (changeups, splitters, forkballs, screwballs, knuckleballs). A whiff is a swinging strike, a blocked swinging strike, a foul tip, or a missed bunt; a swing is any of those plus fouls and balls in play.
* Only games where the pitcher started are modeled. The target is that game's strikeouts.

## Features

Every feature for a start is computed only from games before that start. Pitcher rolling features use `groupby(pitcher).shift(1)` before `.rolling(...)`. League averages and the opponent rate are built from daily totals shifted one date, so even the first game of a doubleheader never feeds the second.

| Feature | Definition |
|---|---|
| `fb_velo_last3`, `fb_velo_last5` | Mean fastball velocity over the last 3 / 5 starts |
| `whiff_rate_last3`, `whiff_rate_last5` | Whiffs / swings summed over the last 3 / 5 starts |
| `k_per_bf_last5` | Strikeouts / batters faced over the last 5 starts |
| `bf_last5`, `pitches_last5` | Mean batters faced and pitch count over the last 5 starts (expected workload) |
| `k_rate_ewm`, `k_per_start_ewm` | K rate and K per start over all prior starts, recency-weighted (a start 5 starts ago counts half as much as the latest; added after `src/compare_windows.py --emphasis` showed a small gain) |
| `rest_days` | Days since the previous start (relief outings ignored), capped at 30 |
| `first_start_of_season` | 1 for a pitcher's first start of a season |
| `is_home` | 1 at home |
| `season_k_rate` | Season-to-date K / BF from prior starts |
| `season_k_per_start` | Season-to-date K per start from prior starts (also the naive baseline) |
| `opp_k_rate_last30` | Opponent's batting K / PA over its previous 30 game-days (needs 5) |
| `n_prior_starts` | Starts before this one in the data |

**Matchup features** use the opponent's posted starting lineup, which is announced a few hours before first pitch. Each batter's numbers come only from games before this one, shrunk toward the league average so a batter with 30 PAs doesn't look extreme. Lineup averages weight each slot by its typical plate appearances (the leadoff hitter bats most).

| Feature | Definition |
|---|---|
| `is_lhp` | Pitcher throws left-handed |
| `pitcher_k_rate_vs_lineup` | The pitcher's prior K rate against each batter's side (switch hitters bat opposite the pitcher), averaged over the lineup |
| `lineup_k_rate_vs_hand` | The lineup's prior K rate against pitchers of this pitcher's hand |
| `lineup_platoon_adv_share` | Share of the lineup with the platoon advantage (opposite hand, or a switch hitter) |
| `usage_fb_hi_last5`, `usage_brk_last5`, `usage_os_last5` | The pitcher's share of 96+ mph fastballs, breaking balls and offspeed pitches over the last 5 starts |
| `lineup_whiff_fb_hi`, `lineup_whiff_brk`, `lineup_whiff_os` | The lineup's prior whiffs per swing on each of those pitch classes |
| `matchup_whiff_rate` | Expected whiff rate for this pitcher's mix against this lineup: the pitcher's usage of each of the five classes times the lineup's whiff rate on it |

**Context features**

| Feature | Definition |
|---|---|
| `day_game` | 1 for a day game |
| `wind_out_mph` | Wind speed, positive blowing out, negative blowing in, 0 across or under a roof |
| `roof_closed` | Dome or roof closed |
| `park_k_factor` | The venue's prior K/PA (both teams) relative to the league, shrunk |
| `ump_k_factor` | The home plate umpire's prior K/PA relative to the league, shrunk |

Game-time temperature (`temp_f`) and the opponent's K rate over its last 10 game-days (`opp_k_rate_last10`) are still built but no longer used: `src/compare_features.py` showed no loss without them in either test season. Colder games do have more strikeouts, about 0.5 points of K% per 10 F within the same month, but the model gets nothing extra from it. The lineup's K rate covers the opponent.

`src/compare_features.py` tests changes one at a time with a paired bootstrap over game dates and writes `reports/feature_tests.md`. Rest flags, a home-vs-road park factor, a per-batter matchup feature, SwStr%, CSW%, pitches per PA, team hook, own whiffs by pitch and velo vs normal were all within noise and are built but unused. Park and umpire factors use only games on earlier dates and are keyed by venue and umpire ID. A renamed park (Minute Maid to Daikin) is one venue, and the Rays' 2025 season at Steinbrenner Field has its own venue ID.

`build_features(raw, lineup_source="previous")` swaps the posted lineup for the opponent's lineup from its previous game, for predictions made a day before lineups are out.

### Early starts with incomplete windows

Rolling windows run over a pitcher's last N starts regardless of season, with `min_periods=1`. A pitcher's second start in the data already has a one-start window, and the first April start uses the end of last season. Only a pitcher's first start in the data has no rolling values; those stay missing and the models fill them with training-set medians plus missing-value indicator columns. `n_prior_starts` tells the model how much history stands behind each value.

`season_k_rate` and `season_k_per_start` use the current season's prior starts. On a pitcher's first start of a season they fall back to that pitcher's most recent earlier season, then to the league average of all starts before that date.

Pitchers whose careers began before 2021 have no rolling values at their first 2021 start, because the data starts in 2021.

## Leakage audit

`audit_leakage()` in `src/build_features.py` runs three checks and prints a PASS/FAIL report every time features are built:

1. No feature is a current-game stat (strikeouts, batters faced, outs, pitch count, velocity, swings, whiffs, ...).
2. Every rolling feature is missing at each pitcher's first start, which can only happen if the window is shifted.
3. For a few random dates, every game stat on that date is set to absurd values (99 K, 99 batters faced, 110 mph, every swing a whiff, every batter plate appearance a strikeout, opponent 99 K), features are rebuilt, and those rows must be unchanged. Lineups, venue, weather and the umpire are left alone because they are known before the game.

`train.py` then adds a known leaky feature, the current game's batters faced, refits, reports how much it inflates accuracy, and runs the audit on that feature set to show it is caught.

## Models and evaluation

* **Split:** time-based only. The test season is the newest season with at least 300 starts and training uses every earlier season, which is train 2021-2025 and test 2026 for the 2021-2026 data. No shuffling anywhere.
* **Tuning:** random forest grid over `n_estimators`, `max_depth` and `min_samples_leaf`, scored by MAE on expanding-window season folds inside training (fit on seasons before S, validate on S, for the last two training seasons).
* **Models:** naive baseline (season-to-date K per start), linear regression, random forest. Seeds are fixed at 42.
* **Benchmark model file:** after evaluation, the random forest is refit on all seasons and saved to `models/model.joblib`. It is the model the weekly quality gate checks, but nothing that makes predictions loads it.
* **Model behind every prediction:** the upcoming-starts predictions, the app and the simulator all use a 50/50 blend of the linear regression and Poisson gradient boosting (`make_blend` in `src/train.py`). It won the comparison in `python src/compare_models.py` (see [`reports/model_comparison.md`](reports/model_comparison.md)) and is refit from the cached data each time those run, so it is never saved to disk.

`evaluate.py` writes MAE, RMSE, R² and percent improvement over the baseline for all three models, permutation feature importance for the random forest, a predicted vs. actual scatter, residual distributions, and a written interpretation. It also shows test MAE for nested feature sets (core only, core plus matchup, core plus context, everything) so you can see which group of features actually helps.

## Patterns report

`python src/analyze_patterns.py` writes `reports/patterns.md` with charts. It covers strikeout rate by days of rest, home vs. away, day vs. night, platoon matchups (pitcher hand vs. batter side), lineups by platoon share and prior K rate, whiff rate by pitch class and fastball velocity, a heatmap of which team lineups whiff more or less than the league on each pitch class, temperature, wind and roofs, the highest- and lowest-strikeout parks and umpires, opponent recent form, and every model feature's rank correlation with strikeouts.

It uses only the training seasons by default, so looking at patterns can't steer the model toward the test season; `--all-seasons` includes everything. Rates are per batter faced with 95% intervals, because strikeouts per start also depend on how deep the starter goes. These are raw associations, so read them alongside the feature-group comparison in `reports/results.md`.

### How much is noise?

Each plate appearance is close to a coin flip at the pitcher's true strikeout rate. Even a model that knew that rate and exactly how many batters the starter would face would still be off by binomial noise, `BF * p * (1 - p)` per start. `evaluate.py` averages that over the test season, compares it with the actual variance of strikeouts, and reports the implied R² ceiling and how much of the explainable variance each model captures. The gap between a model's error and the floor is everything else a pre-game model cannot see: early hooks and pitch limits, the umpire's zone, injuries and blowouts, and matchup randomness.

## Results

Results are generated by a run on real data and written to [`reports/results.md`](reports/results.md) and `metrics/latest.json`. This README does not quote numbers until that first run is done.

## CI and weekly retraining

`.github/workflows/pipeline.yml` has two jobs:

* **On push and pull request to `main`:** `ruff check .` and `pytest`. The tests use a hand-made fixture and never call an API.
* **Weekly (Mondays, April through September) and on manual trigger:** fetch only games after the last cached date, and stop cleanly if there are none. Otherwise commit the new data, rebuild features, retrain, evaluate, and commit `metrics/latest.json`, `models/model.joblib` and `reports/`. The job fails before committing the model if the random forest's test MAE is worse than the baseline's. A manual run can force a retrain with no new data.

### Where the cache lives, and why

The cache is committed to the repository as small per-game summaries:

* `data/raw/game_logs/`, `data/raw/team_games/` and `data/raw/statcast_agg/` hold one parquet file per fetched date chunk, well within normal git limits.
* So are `data/raw/games/` and `data/raw/lineups/`. The batter summaries are the largest table, around 220,000 rows for five seasons, still a few megabytes as parquet.
* Raw Statcast pitches (`data/raw/statcast/`, about 1 GB) are gitignored and kept only on your machine, so reruns can re-summarize without downloading again. The weekly job runs with `--no-keep-raw` and never stores them.

Committing the summaries makes them persist between workflow runs with no extra setup. The GitHub Actions cache would also work, but it evicts entries unused for seven days, and a missed week would force a full multi-hour re-download. Git LFS would work for the raw pitches but adds storage costs and nothing the features need.

**Before the first scheduled run:** run `python src/fetch_data.py` locally once, then commit `data/raw/` (everything except the gitignored raw Statcast folder) and push. The weekly job only fetches incrementally, so it needs that starting point. In Settings > Actions > General, set Workflow permissions to "Read and write" so the job can commit.

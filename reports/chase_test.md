# Chase rate (O-Swing%) feature tests

Baseline: the live blend with its current features. Each test adds one feature (or group). Fit on seasons before the test season, scored on it. Diff = MAE with the change minus baseline MAE (negative is better), with a 95% paired-bootstrap interval over game dates. Kept only if better in both seasons with both intervals below zero. Under 0.005 and not in both seasons is noise.

| Test | MAE 2025 base → new | Diff 2025 [95% CI] | MAE 2026 base → new | Diff 2026 [95% CI] | Verdict |
|---|---|---|---|---|---|
| + pitcher chase rate, last 5 starts | 1.7542 → 1.7554 | +0.0012 [-0.0009, +0.0032] | 1.7282 → 1.7285 | +0.0002 [-0.0016, +0.0019] | noise |
| + pitcher chase rate, career (shrunk) | 1.7542 → 1.7544 | +0.0002 [-0.0014, +0.0017] | 1.7282 → 1.7278 | -0.0004 [-0.0022, +0.0012] | noise |
| + pitcher two-strike chase rate | 1.7542 → 1.7555 | +0.0014 [-0.0001, +0.0029] | 1.7282 → 1.7261 | -0.0021 [-0.0038, -0.0004] | noise |
| + lineup chase rate | 1.7542 → 1.7532 | -0.0010 [-0.0025, +0.0007] | 1.7282 → 1.7274 | -0.0008 [-0.0026, +0.0009] | noise |
| + pitcher x lineup chase (log5) | 1.7542 → 1.7541 | -0.0000 [-0.0016, +0.0015] | 1.7282 → 1.7273 | -0.0009 [-0.0027, +0.0009] | noise |
| + all chase features | 1.7542 → 1.7535 | -0.0006 [-0.0027, +0.0013] | 1.7282 → 1.7281 | -0.0002 [-0.0025, +0.0021] | noise |

## Coverage and correlation with strikeouts

| Feature | Coverage | Corr with K |
|---|---|---|
| p_chase_last5 | 0.946 | +0.138 |
| p_chase_career | 0.999 | +0.134 |
| p_chase_2k_career | 0.999 | +0.077 |
| lineup_chase_rate | 0.999 | +0.003 |
| matchup_chase_rate | 0.999 | +0.101 |

## Leakage check (rebuilt with data cut off before each date)

- 2026-03-25 p_chase_last5: ok
- 2026-03-25 p_chase_career: ok
- 2026-03-25 p_chase_2k_career: ok
- 2026-03-25 lineup_chase_rate: ok
- 2026-03-25 matchup_chase_rate: ok
- 2026-05-15 p_chase_last5: ok
- 2026-05-15 p_chase_career: ok
- 2026-05-15 p_chase_2k_career: ok
- 2026-05-15 lineup_chase_rate: ok
- 2026-05-15 matchup_chase_rate: ok
- 2026-08-13 p_chase_last5: ok
- 2026-08-13 p_chase_career: ok
- 2026-08-13 p_chase_2k_career: ok
- 2026-08-13 lineup_chase_rate: ok
- 2026-08-13 matchup_chase_rate: ok


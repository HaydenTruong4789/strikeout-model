# Recent form and fine pitch-matchup feature tests

Baseline: the live blend with its current features. Each test adds one feature (or group). Fit on seasons before the test season, scored on it. Diff = MAE with the change minus baseline MAE (negative is better), with a 95% paired-bootstrap interval over game dates. Kept only if better in both seasons with both intervals below zero. Under 0.005 and not in both seasons is noise.

| Test | MAE 2025 base → new | Diff 2025 [95% CI] | MAE 2026 base → new | Diff 2026 [95% CI] | Verdict |
|---|---|---|---|---|---|
| + last start K surprise | 1.7542 → 1.7540 | -0.0002 [-0.0020, +0.0017] | 1.7282 → 1.7271 | -0.0012 [-0.0029, +0.0004] | noise |
| + bad starts in last 5 | 1.7542 → 1.7535 | -0.0007 [-0.0020, +0.0006] | 1.7282 → 1.7265 | -0.0017 [-0.0039, +0.0003] | noise |
| + good starts in last 5 | 1.7542 → 1.7554 | +0.0013 [-0.0003, +0.0027] | 1.7282 → 1.7264 | -0.0018 [-0.0033, -0.0004] | noise |
| + last start short (BF <= 15) | 1.7542 → 1.7544 | +0.0003 [-0.0016, +0.0022] | 1.7282 → 1.7252 | -0.0030 [-0.0052, -0.0008] | noise |
| + fine pitch-type x velo whiff matchup | 1.7542 → 1.7530 | -0.0011 [-0.0046, +0.0024] | 1.7282 → 1.7273 | -0.0009 [-0.0045, +0.0024] | noise |
| + fine pitch-type x velo chase matchup | 1.7542 → 1.7553 | +0.0011 [-0.0008, +0.0029] | 1.7282 → 1.7265 | -0.0018 [-0.0040, +0.0003] | noise |
| + both fine matchups | 1.7542 → 1.7545 | +0.0003 [-0.0027, +0.0034] | 1.7282 → 1.7283 | +0.0001 [-0.0031, +0.0032] | noise |

## Bounce-back and momentum: residuals the model leaves

Residual = actual K minus the live blend's out-of-sample prediction (fit on earlier seasons), for 2024-2026 starts. Positive means the pitcher beat the model. A bad start: K/BF more than 0.08 below his recency-weighted K rate, 4+ runs while he pitched (from Statcast; game logs have no earned runs), or under 4 innings. A good start: K/BF 0.08 above, or 7+ K with 0-1 runs.

Share of all starts: bad 45.9%, good 24.5%.

| Previous starts | Group | n | Mean residual [95% CI] |
|---|---|---|---|
| All starts | all | 14826 | -0.036 [-0.071, -0.000] |
| Last start bad | yes | 6479 | -0.052 [-0.104, -0.001] |
| Last start bad | no | 8049 | -0.029 [-0.078, +0.020] |
| Last start good | yes | 3591 | -0.004 [-0.079, +0.070] |
| Last start good | no | 10937 | -0.051 [-0.092, -0.011] |
| Bad starts in last 5 | 0 | 984 | -0.086 [-0.227, +0.055] |
| Bad starts in last 5 | 1 | 3482 | -0.015 [-0.088, +0.058] |
| Bad starts in last 5 | 2 | 4859 | -0.070 [-0.132, -0.008] |
| Bad starts in last 5 | 3+ | 5203 | -0.019 [-0.077, +0.040] |
| Good starts in last 5 | 0 | 3803 | -0.115 [-0.182, -0.048] |
| Good starts in last 5 | 1 | 5692 | -0.039 [-0.096, +0.017] |
| Good starts in last 5 | 2 | 3571 | +0.037 [-0.036, +0.111] |
| Good starts in last 5 | 3+ | 1462 | -0.033 [-0.153, +0.087] |
| Last 3 starts all bad | yes | 1310 | -0.146 [-0.254, -0.037] |
| Last 3 starts all bad | no | 12740 | -0.024 [-0.063, +0.014] |
| Last 3 starts all good | yes | 247 | +0.054 [-0.225, +0.333] |
| Last 3 starts all good | no | 13803 | -0.037 [-0.074, -0.001] |
| Last start short (BF <= 15) | yes | 1033 | -0.218 [-0.339, -0.098] |
| Last start short (BF <= 15) | no | 13495 | -0.026 [-0.063, +0.011] |

## Coverage and correlation with strikeouts

| Feature | Coverage | Corr with K |
|---|---|---|
| last_start_k_surprise | 0.942 | +0.029 |
| bad_starts_last5 | 0.968 | -0.090 |
| good_starts_last5 | 0.968 | +0.173 |
| last_start_short | 0.968 | -0.164 |
| matchup_whiff_fine | 0.933 | +0.355 |
| matchup_chase_fine | 0.933 | +0.131 |

## Leakage check (rebuilt with data cut off before each date)

- 2026-03-25 last_start_k_surprise: ok
- 2026-03-25 bad_starts_last5: ok
- 2026-03-25 good_starts_last5: ok
- 2026-03-25 last_start_short: ok
- 2026-03-25 matchup_whiff_fine: ok
- 2026-03-25 matchup_chase_fine: ok
- 2026-05-15 last_start_k_surprise: ok
- 2026-05-15 bad_starts_last5: ok
- 2026-05-15 good_starts_last5: ok
- 2026-05-15 last_start_short: ok
- 2026-05-15 matchup_whiff_fine: ok
- 2026-05-15 matchup_chase_fine: ok
- 2026-08-13 last_start_k_surprise: ok
- 2026-08-13 bad_starts_last5: ok
- 2026-08-13 good_starts_last5: ok
- 2026-08-13 last_start_short: ok
- 2026-08-13 matchup_whiff_fine: ok
- 2026-08-13 matchup_chase_fine: ok


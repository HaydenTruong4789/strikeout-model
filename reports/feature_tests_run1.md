# Model tests: one change at a time

Each test is fit on all seasons before the test season and scored on it. Diff is the change's MAE minus the baseline's (negative is better), with a 95% paired-bootstrap interval over game dates. Kept only if better in both seasons with both intervals below zero; a drop is kept if not significantly worse in either. Differences under 0.005 that don't hold in both seasons are noise.

```
Leakage audit, candidate features (15 features)
  [PASS] no current-game stats used as features
  [PASS] rolling features shifted (NaN at first start)
  [PASS] features unchanged when current game is altered (3 dates)
  => PASSED
```

## Results

| Test | Target | 2025 base -> new | 2026 base -> new | 2025 diff [95% CI] | 2026 diff [95% CI] | Verdict |
|---|---|---|---|---|---|---|
| 1a. rest flags replace rest_days (blend) | K | 1.7550 -> 1.7542 | 1.7270 -> 1.7259 | -0.0008 [-0.0032, +0.0018] | -0.0011 [-0.0031, +0.0011] | noise |
| 1b. rest flags replace rest_days (two-stage) | K | 1.7560 -> 1.7553 | 1.7363 -> 1.7352 | -0.0007 [-0.0025, +0.0013] | -0.0011 [-0.0038, +0.0014] | noise |
| 1c. rest flags, batters-faced model | BF | 2.7511 -> 2.7503 | 2.8668 -> 2.8504 | -0.0008 [-0.0106, +0.0099] | -0.0164 [-0.0306, -0.0017] | not consistent |
| 2a. leave short-rest starts out of training | K | 1.7550 -> 1.7571 | 1.7270 -> 1.7232 | +0.0021 [-0.0010, +0.0050] | -0.0038 [-0.0073, -0.0004] | noise |
| 2b. leave starts under 9 BF out of training | K | 1.7550 -> 1.7620 | 1.7270 -> 1.7529 | +0.0070 [+0.0024, +0.0120] | +0.0259 [+0.0204, +0.0314] | worse |
| 3. home-vs-road park factor replaces current | K | 1.7550 -> 1.7540 | 1.7270 -> 1.7271 | -0.0010 [-0.0043, +0.0021] | +0.0001 [-0.0029, +0.0030] | noise |
| 4. drop temperature | K | 1.7550 -> 1.7547 | 1.7270 -> 1.7276 | -0.0004 [-0.0022, +0.0015] | +0.0006 [-0.0011, +0.0022] | drop (no significant loss) |
| 5. blend vs linear alone | K | 1.7567 -> 1.7550 | 1.7349 -> 1.7270 | -0.0017 [-0.0068, +0.0032] | -0.0079 [-0.0124, -0.0031] | not consistent |
| 6. + per-batter matchup expected K | K | 1.7550 -> 1.7576 | 1.7270 -> 1.7278 | +0.0026 [-0.0011, +0.0065] | +0.0008 [-0.0030, +0.0046] | noise |
| 7a. batters-faced model + workload | BF | 2.7511 -> 2.7337 | 2.8668 -> 2.8476 | -0.0173 [-0.0264, -0.0083] | -0.0192 [-0.0291, -0.0089] | keep |
| 7b. two-stage with workload BF model | K | 1.7560 -> 1.7559 | 1.7363 -> 1.7332 | -0.0001 [-0.0024, +0.0024] | -0.0031 [-0.0056, -0.0006] | noise |
| 7c. two-stage (workload BF) vs one-stage blend | K | 1.7550 -> 1.7559 | 1.7270 -> 1.7332 | +0.0009 [-0.0047, +0.0064] | +0.0062 [+0.0007, +0.0112] | worse |
| 9. + swinging strike rate | K | 1.7550 -> 1.7539 | 1.7270 -> 1.7275 | -0.0011 [-0.0028, +0.0007] | +0.0005 [-0.0012, +0.0022] | noise |
| 9. + CSW rate | K | 1.7550 -> 1.7550 | 1.7270 -> 1.7261 | -0.0000 [-0.0019, +0.0019] | -0.0009 [-0.0026, +0.0008] | noise |
| 9. + pitches per PA | K | 1.7550 -> 1.7540 | 1.7270 -> 1.7265 | -0.0010 [-0.0026, +0.0007] | -0.0006 [-0.0018, +0.0008] | noise |
| 9. + team hook | K | 1.7550 -> 1.7555 | 1.7270 -> 1.7266 | +0.0004 [-0.0016, +0.0027] | -0.0004 [-0.0022, +0.0012] | noise |
| 9. + own whiffs by pitch | K | 1.7550 -> 1.7546 | 1.7270 -> 1.7270 | -0.0004 [-0.0026, +0.0017] | +0.0000 [-0.0018, +0.0019] | noise |
| 9. + velo vs normal | K | 1.7550 -> 1.7540 | 1.7270 -> 1.7255 | -0.0010 [-0.0039, +0.0020] | -0.0015 [-0.0043, +0.0013] | noise |
| 9. + previous pitch count | K | 1.7550 -> 1.7551 | 1.7270 -> 1.7254 | +0.0000 [-0.0026, +0.0028] | -0.0016 [-0.0041, +0.0009] | noise |
| 9. drop the 5 low-signal features | K | 1.7550 -> 1.7586 | 1.7270 -> 1.7278 | +0.0036 [+0.0012, +0.0059] | +0.0008 [-0.0018, +0.0033] | keep the features |
| 9. drop opponent last-10 form (keep lineup rate) | K | 1.7550 -> 1.7547 | 1.7270 -> 1.7267 | -0.0003 [-0.0023, +0.0016] | -0.0003 [-0.0016, +0.0010] | drop (no significant loss) |

Passed the rule: 4. drop temperature, 7a. batters-faced model + workload, 9. drop opponent last-10 form (keep lineup rate).

## Probabilities (blend means, distribution fit on the previous season)

| Method, season | log loss | Brier 4.5 | calib err 4.5 | Brier 5.5 | calib err 5.5 | Brier 6.5 | calib err 6.5 | alpha |
|---|---|---|---|---|---|---|---|---|
| poisson 2025 | 2.1871 | 0.2156 | 0.0181 | 0.1982 | 0.0223 | 0.1544 | 0.0205 |  |
| negbin 2025 | 2.1871 | 0.2156 | 0.0181 | 0.1982 | 0.0223 | 0.1544 | 0.0205 | 0.0000 |
| empirical 2025 | 2.2000 | 0.2157 | 0.0190 | 0.1982 | 0.0222 | 0.1543 | 0.0192 |  |
| poisson 2026 | 2.1545 | 0.2102 | 0.0201 | 0.1929 | 0.0229 | 0.1511 | 0.0188 |  |
| negbin 2026 | 2.1547 | 0.2102 | 0.0198 | 0.1929 | 0.0231 | 0.1512 | 0.0191 | 0.0046 |
| empirical 2026 | 2.1700 | 0.2112 | 0.0345 | 0.1932 | 0.0296 | 0.1512 | 0.0226 |  |

## Parks by venue ID

The model's park factor has always been keyed by venue ID; only the patterns report grouped by name.

Venue IDs with more than one name: 4: Guaranteed Rate Field, Rate Field; 22: Dodger Stadium, UNIQLO Field at Dodger Stadium; 2392: Minute Maid Park, Daikin Park; 2735: BB&T Ballpark, Muncy Bank Ballpark, Journey Bank Ballpark

Rays home games by season and venue:

| Season | Venue | Games |
|---|---|---|
| 2024 | Tropicana Field | 81 |
| 2025 | George M. Steinbrenner Field | 81 |
| 2026 | Tropicana Field | 86 |

Highest and lowest home-vs-road park factors, 2026 (mean over games):

| Venue | Current factor | Home vs road |
|---|---|---|
| T-Mobile Park | 1.116 | 1.115 |
| American Family Field | 1.091 | 1.087 |
| Daikin Park | 1.020 | 1.063 |
| Petco Park | 1.014 | 1.063 |
| Tropicana Field | 1.073 | 1.053 |
| Coors Field | 0.890 | 0.882 |
| Kauffman Stadium | 0.883 | 0.911 |
| PNC Park | 0.966 | 0.920 |
| Fenway Park | 0.977 | 0.935 |
| Busch Stadium | 0.893 | 0.943 |

Correlation between the two factors across games: 0.83.

## Temperature within month (open-air games, training seasons)

K/PA change per 10 F, all games: -0.43 points [-0.54, -0.31]; within the same season and month: -0.48 points [-0.61, -0.33].

| Month | <65 | 65-74 | 75-84 | 85+ |
|---|---|---|---|---|
| 4 | 23.3% (62,715) | 23.0% (20,675) | 22.5% (11,002) | 20.3% (2,104) |
| 5 | 22.6% (39,093) | 22.5% (35,357) | 22.4% (23,134) | 21.9% (6,113) |
| 6 | 23.1% (11,103) | 22.7% (34,432) | 22.4% (35,120) | 21.4% (17,048) |
| 7 | 24.3% (4,302) | 22.5% (22,607) | 22.4% (44,038) | 21.7% (21,821) |
| 8 | 24.5% (3,221) | 23.5% (26,953) | 22.3% (52,910) | 21.6% (19,386) |
| 9 | 24.2% (19,784) | 23.3% (48,876) | 22.5% (37,460) | 21.6% (9,194) |

Cells show K/PA (PA); cells under 2,000 PA are blank. Months 3 and 10 are folded into 4 and 9.


# How strikeouts happen: two strikes, chases and tunneling

Statcast pitches from 2024-03-20 to 2026-10-19 (2,170,127 pitches, regular season and postseason). Out of the zone = Statcast zone 11-14. Chase = a swing at a pitch out of the zone.

## 1. Every count

| Count | pitches | out of zone | chase | whiff per chase | zone swing | fastball | breaking | offspeed | strikeout on pitch |
|---|---|---|---|---|---|---|---|---|---|
| 0-0 | 557,782 | 46.4% | 16.6% | 49.5% | 45.5% | 60.4% | 30.6% | 8.9% | 0.0% |
| 1-0 | 210,746 | 45.1% | 22.7% | 48.7% | 59.5% | 59.9% | 26.4% | 13.6% | 0.0% |
| 2-0 | 70,781 | 41.2% | 20.4% | 44.0% | 55.6% | 70.7% | 19.6% | 9.5% | 0.0% |
| 3-0 | 22,002 | 37.5% | 3.1% | 35.7% | 11.6% | 94.1% | 3.8% | 1.8% | 0.0% |
| 0-1 | 281,853 | 55.4% | 29.1% | 47.9% | 74.2% | 50.2% | 31.3% | 18.4% | 0.0% |
| 1-1 | 217,769 | 50.4% | 32.1% | 45.7% | 77.5% | 50.6% | 30.6% | 18.8% | 0.0% |
| 2-1 | 111,798 | 43.9% | 32.4% | 43.5% | 77.7% | 59.3% | 26.6% | 14.0% | 0.0% |
| 3-1 | 46,393 | 38.9% | 26.2% | 38.3% | 70.7% | 77.2% | 15.6% | 7.0% | 0.0% |
| 0-2 | 147,744 | 68.4% | 33.8% | 42.3% | 86.6% | 43.0% | 39.4% | 17.5% | 17.7% |
| 1-2 | 212,900 | 62.9% | 39.0% | 41.1% | 88.5% | 43.9% | 37.8% | 18.2% | 19.1% |
| 2-2 | 180,882 | 54.0% | 43.1% | 38.9% | 89.1% | 48.5% | 34.2% | 17.2% | 19.9% |
| 3-2 | 109,477 | 42.0% | 44.8% | 35.0% | 88.8% | 62.8% | 26.2% | 11.0% | 19.9% |

## 2. 0-2 counts, pitcher by pitcher

| Pitchers with 100+ 0-2 pitches | out of zone | breaking balls | strikeout on pitch |
|---|---|---|---|
| 10th percentile | 59.5% | 15.0% | 12.9% |
| median | 68.6% | 40.2% | 17.6% |
| 90th percentile | 77.6% | 63.2% | 23.3% |

Across 508 pitchers, going out of the zone more often on 0-2 and strikeouts per 0-2 pitch correlate at r = -0.330.

## 3. Two strikes, by pitch family

| Family | pitches | share of 2-strike pitches | out of zone | chase | whiff per chase | foul | called K | swinging K | in play | ball |
|---|---|---|---|---|---|---|---|---|---|---|
| fastball | 313,498 | 48.2% | 50.0% | 36.9% | 31.6% | 27.1% | 6.0% | 12.2% | 22.8% | 31.6% |
| breaking | 229,349 | 35.2% | 64.3% | 40.3% | 48.3% | 19.9% | 3.5% | 17.2% | 20.9% | 38.2% |
| offspeed | 107,734 | 16.5% | 68.8% | 42.9% | 39.3% | 19.1% | 2.2% | 16.6% | 22.5% | 39.4% |
| other | 422 | 0.1% | 65.9% | 31.7% | 6.8% | 17.1% | 3.1% | 2.4% | 32.0% | 44.5% |

## 4. Two strikes, by pitch type (2,000+ pitches)

| Pitch | pitches | share of 2-strike pitches | out of zone | chase | whiff per chase | foul | called K | swinging K | in play | ball |
|---|---|---|---|---|---|---|---|---|---|---|
| 4-Seam Fastball | 207,716 | 31.9% | 51.1% | 37.4% | 32.5% | 28.3% | 5.1% | 13.2% | 20.8% | 32.2% |
| Slider | 101,949 | 15.7% | 62.9% | 40.6% | 50.3% | 20.1% | 3.2% | 17.9% | 21.2% | 37.3% |
| Changeup | 77,613 | 11.9% | 68.8% | 43.0% | 37.1% | 19.1% | 2.2% | 15.9% | 23.3% | 39.3% |
| Sinker | 70,669 | 10.9% | 44.9% | 33.7% | 24.6% | 24.2% | 9.6% | 8.1% | 28.0% | 29.5% |
| Sweeper | 61,189 | 9.4% | 64.2% | 39.9% | 45.8% | 20.1% | 3.7% | 16.4% | 21.0% | 38.2% |
| Curveball | 47,373 | 7.3% | 66.1% | 39.9% | 46.3% | 19.6% | 3.8% | 16.3% | 20.3% | 39.5% |
| Cutter | 35,113 | 5.4% | 53.5% | 39.2% | 37.1% | 25.2% | 4.0% | 14.3% | 23.7% | 32.5% |
| Split-Finger | 28,617 | 4.4% | 68.8% | 42.5% | 45.3% | 19.1% | 2.0% | 18.5% | 20.4% | 39.6% |
| Knuckle Curve | 14,843 | 2.3% | 68.0% | 41.9% | 52.1% | 18.8% | 3.3% | 18.5% | 19.6% | 39.5% |
| Slurve | 3,768 | 0.6% | 66.8% | 39.7% | 47.3% | 19.0% | 4.2% | 16.1% | 20.4% | 39.6% |

## 5. How batters respond to two-strike pitches out of the zone

By how far off the zone:

| Distance off the zone | pitches | chase | whiff per chase | foul | strikeout | in play |
|---|---|---|---|---|---|---|
| 0-2 in | 72,446 | 69.4% | 25.6% | 28.4% | 22.1% | 23.3% |
| 2-4 in | 63,576 | 58.5% | 32.6% | 22.6% | 19.9% | 16.9% |
| 4-6 in | 54,988 | 46.6% | 41.7% | 16.4% | 19.5% | 10.8% |
| 6-9 in | 65,022 | 32.6% | 55.9% | 9.0% | 18.2% | 5.4% |
| 9-12 in | 45,461 | 19.5% | 74.0% | 3.5% | 14.4% | 1.6% |
| 12+ in | 76,965 | 7.7% | 91.3% | 0.4% | 7.0% | 0.2% |

By the batter's own chase tendency before two strikes (thirds):

| Batter (O-Swing% before 2 strikes) | pitches | chase | whiff per chase | foul | strikeout | in play |
|---|---|---|---|---|---|---|
| low chasers | 118,918 | 35.0% | 38.1% | 12.6% | 14.4% | 9.1% |
| middle | 118,713 | 39.6% | 39.5% | 13.7% | 16.6% | 10.2% |
| high chasers | 118,278 | 43.6% | 40.5% | 14.9% | 18.5% | 11.1% |

## 6. Tunneling

Tunnel gap = distance between this pitch and the previous one in the at-bat when both are about 24 ft in front of the plate, where a hitter has to commit. Plate gap = their distance at the plate. Only pairs of different pitch types.

| Two-strike pitch out of the zone, tunnel gap to the previous pitch | pitches | median tunnel gap | median plate gap | chase | whiff per chase | strikeout |
|---|---|---|---|---|---|---|
| just off (under 4 in), tightest 20% | 24,243 | 4.1 in | 13.2 in | 71.2% | 27.4% | 21.9% |
| just off (under 4 in), 2nd | 21,583 | 7.5 in | 17.5 in | 66.5% | 27.9% | 21.3% |
| just off (under 4 in), 3rd | 19,024 | 10.4 in | 21.4 in | 63.1% | 27.9% | 20.3% |
| just off (under 4 in), 4th | 15,724 | 13.6 in | 25.5 in | 58.7% | 27.1% | 18.7% |
| just off (under 4 in), widest 20% | 11,287 | 18.5 in | 31.2 in | 53.3% | 26.3% | 17.1% |
| well off (4 in or more), tightest 20% | 27,314 | 4.2 in | 15.7 in | 37.1% | 53.2% | 19.8% |
| well off (4 in or more), 2nd | 29,973 | 7.6 in | 20.8 in | 31.3% | 54.0% | 16.9% |
| well off (4 in or more), 3rd | 32,532 | 10.5 in | 25.4 in | 26.5% | 54.8% | 14.5% |
| well off (4 in or more), 4th | 35,832 | 13.7 in | 30.4 in | 21.1% | 56.2% | 11.9% |
| well off (4 in or more), widest 20% | 40,270 | 19.2 in | 38.7 in | 13.4% | 55.7% | 7.5% |

Correlation between tunnel gap and chasing, with distance off the zone held in narrow bins (negative = tighter tunnels draw more chases):

| Inches off the zone | n | r |
|---|---|---|
| [0.0, 2.0) | 48,914 | -0.107 |
| [2.0, 4.0) | 42,947 | -0.135 |
| [4.0, 6.0) | 37,336 | -0.132 |
| [6.0, 9.0) | 44,330 | -0.128 |
| [9.0, inf) | 84,255 | -0.142 |

### Pitchers

- 456 pitchers with 300+ fastball-then-breaking/offspeed pairs and 200+ two-strike pitches out of the zone.
- Correlation between his median tunnel gap and the two-strike chase rate he gets: r = -0.308 (-1.60 points of chase rate per extra inch of gap).
- Correlation between his median release-point gap and that chase rate: r = -0.098.


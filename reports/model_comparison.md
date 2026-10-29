# Model comparison

Fit on all seasons before the test season; lower MAE is better.

## 2025

| Model | MAE | RMSE | R² |
|---|---|---|---|
| linear (current) | 1.7567 | 2.2162 | 0.2052 |
| ridge | 1.7563 | 2.2160 | 0.2053 |
| lasso | 1.7564 | 2.2162 | 0.2052 |
| poisson regression | 1.7595 | 2.2174 | 0.2043 |
| random forest | 1.7680 | 2.2302 | 0.1951 |
| gradient boosting (squared_error) | 1.7644 | 2.2225 | 0.2006 |
| gradient boosting (absolute_error) | 1.7687 | 2.2318 | 0.1939 |
| gradient boosting (poisson) | 1.7651 | 2.2234 | 0.2000 |
| two-stage BF x K rate | 1.7560 | 2.2149 | 0.2061 |
| blend linear + boosting | 1.7550 | 2.2122 | 0.2081 |
| linear, recency half-life 3 | 1.7575 | 2.2173 | 0.2044 |
| linear, recency half-life 8 | 1.7575 | 2.2165 | 0.2050 |
| naive baseline | 1.9132 | 2.4193 | 0.0528 |

## 2026

| Model | MAE | RMSE | R² |
|---|---|---|---|
| linear (current) | 1.7349 | 2.1706 | 0.2444 |
| ridge | 1.7350 | 2.1707 | 0.2443 |
| lasso | 1.7349 | 2.1706 | 0.2443 |
| poisson regression | 1.7418 | 2.1778 | 0.2394 |
| random forest | 1.7353 | 2.1726 | 0.2430 |
| gradient boosting (squared_error) | 1.7352 | 2.1680 | 0.2462 |
| gradient boosting (absolute_error) | 1.7281 | 2.1684 | 0.2459 |
| gradient boosting (poisson) | 1.7322 | 2.1648 | 0.2484 |
| two-stage BF x K rate | 1.7363 | 2.1699 | 0.2449 |
| blend linear + boosting | 1.7270 | 2.1602 | 0.2516 |
| linear, recency half-life 3 | 1.7344 | 2.1705 | 0.2445 |
| linear, recency half-life 8 | 1.7353 | 2.1712 | 0.2440 |
| naive baseline | 1.8772 | 2.3499 | 0.1144 |

## MAE by season, best first

| Model | 2025 | 2026 | mean MAE |
|---|---|---|---|
| blend linear + boosting | 1.7550 | 1.7270 | 1.7410 |
| ridge | 1.7563 | 1.7350 | 1.7456 |
| lasso | 1.7564 | 1.7349 | 1.7456 |
| linear (current) | 1.7567 | 1.7349 | 1.7458 |
| linear, recency half-life 3 | 1.7575 | 1.7344 | 1.7460 |
| two-stage BF x K rate | 1.7560 | 1.7363 | 1.7462 |
| linear, recency half-life 8 | 1.7575 | 1.7353 | 1.7464 |
| gradient boosting (absolute_error) | 1.7687 | 1.7281 | 1.7484 |
| gradient boosting (poisson) | 1.7651 | 1.7322 | 1.7487 |
| gradient boosting (squared_error) | 1.7644 | 1.7352 | 1.7498 |
| poisson regression | 1.7595 | 1.7418 | 1.7507 |
| random forest | 1.7680 | 1.7353 | 1.7517 |
| naive baseline | 1.9132 | 1.8772 | 1.8952 |

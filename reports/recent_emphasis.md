# Training window comparison

Linear regression refit every 5 days on starts before that day; lower MAE is better.

## 2025 (4,954 starts)

| Training data | MAE | RMSE | MAE Apr-May | MAE Jun-Sep |
|---|---|---|---|---|
| all history | 1.759 | 2.218 | 1.780 | 1.747 |
| + last-5 K per start | 1.759 | 2.218 | 1.780 | 1.747 |
| + recency-weighted K stats | 1.754 | 2.214 | 1.777 | 1.742 |
| weight last 5 starts x3 | 1.758 | 2.223 | 1.783 | 1.745 |
| weight by recency (1-year half-life) | 1.758 | 2.217 | 1.786 | 1.743 |
| season fit, all earlier seasons | 1.761 | 2.220 | 1.780 | 1.751 |
| naive baseline | 1.913 | 2.419 | 2.026 | 1.852 |

## 2026 (4,928 starts)

| Training data | MAE | RMSE | MAE Apr-May | MAE Jun-Sep |
|---|---|---|---|---|
| all history | 1.734 | 2.170 | 1.725 | 1.739 |
| + last-5 K per start | 1.733 | 2.169 | 1.724 | 1.738 |
| + recency-weighted K stats | 1.732 | 2.167 | 1.722 | 1.738 |
| weight last 5 starts x3 | 1.729 | 2.168 | 1.719 | 1.735 |
| weight by recency (1-year half-life) | 1.733 | 2.169 | 1.724 | 1.738 |
| season fit, all earlier seasons | 1.737 | 2.174 | 1.727 | 1.742 |
| naive baseline | 1.877 | 2.350 | 1.922 | 1.852 |

# Training window comparison

Linear regression refit every 5 days on starts before that day; lower MAE is better.

## 2025 (4,954 starts)

| Training data | MAE | RMSE | MAE Apr-May | MAE Jun-Sep |
|---|---|---|---|---|
| all history | 1.759 | 2.218 | 1.780 | 1.747 |
| last 12 months | 1.762 | 2.222 | 1.790 | 1.748 |
| last 6 months | 1.784 | 2.251 | 1.850 | 1.749 |
| last 3 months | 1.928 | 2.774 | 2.262 | 1.748 |
| last ~1 season of starts | 1.763 | 2.222 | 1.790 | 1.748 |
| train on each pitcher's last 10 starts | 1.767 | 2.242 | 1.798 | 1.751 |
| season stats -> last 10 starts | 1.760 | 2.220 | 1.779 | 1.750 |
| season fit, all earlier seasons | 1.761 | 2.220 | 1.780 | 1.751 |
| naive baseline | 1.913 | 2.419 | 2.026 | 1.852 |

## 2026 (4,926 starts)

| Training data | MAE | RMSE | MAE Apr-May | MAE Jun-Sep |
|---|---|---|---|---|
| all history | 1.734 | 2.170 | 1.725 | 1.739 |
| last 12 months | 1.738 | 2.174 | 1.727 | 1.745 |
| last 6 months | 1.770 | 2.216 | 1.818 | 1.743 |
| last 3 months | 1.758 | 2.201 | 1.779 | 1.747 |
| last ~1 season of starts | 1.738 | 2.174 | 1.727 | 1.745 |
| train on each pitcher's last 10 starts | 1.732 | 2.178 | 1.727 | 1.735 |
| season stats -> last 10 starts | 1.738 | 2.174 | 1.728 | 1.744 |
| season fit, all earlier seasons | 1.737 | 2.174 | 1.727 | 1.742 |
| naive baseline | 1.877 | 2.350 | 1.922 | 1.852 |

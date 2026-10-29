# Training window comparison

Linear regression refit every 5 days on starts before that day; lower MAE is better.

## 2025 (4,954 starts)

| Training data | MAE | RMSE | MAE Apr-May | MAE Jun-Sep |
|---|---|---|---|---|
| all history | 1.759 | 2.218 | 1.780 | 1.747 |
| season stats -> last 5 starts | 1.766 | 2.224 | 1.782 | 1.757 |
| season stats -> last 6 starts | 1.763 | 2.222 | 1.781 | 1.753 |
| season stats -> last 7 starts | 1.764 | 2.223 | 1.782 | 1.754 |
| season stats -> last 8 starts | 1.762 | 2.222 | 1.780 | 1.752 |
| season stats -> last 9 starts | 1.761 | 2.220 | 1.780 | 1.751 |
| season stats -> last 10 starts | 1.760 | 2.220 | 1.779 | 1.750 |
| train on each pitcher's last 5 starts | 1.778 | 2.259 | 1.811 | 1.760 |
| train on each pitcher's last 6 starts | 1.776 | 2.256 | 1.807 | 1.759 |
| train on each pitcher's last 7 starts | 1.774 | 2.253 | 1.806 | 1.757 |
| train on each pitcher's last 8 starts | 1.771 | 2.248 | 1.803 | 1.753 |
| train on each pitcher's last 9 starts | 1.768 | 2.245 | 1.800 | 1.750 |
| train on each pitcher's last 10 starts | 1.767 | 2.242 | 1.798 | 1.751 |
| season fit, all earlier seasons | 1.761 | 2.220 | 1.780 | 1.751 |
| naive baseline | 1.913 | 2.419 | 2.026 | 1.852 |

## 2026 (4,928 starts)

| Training data | MAE | RMSE | MAE Apr-May | MAE Jun-Sep |
|---|---|---|---|---|
| all history | 1.734 | 2.170 | 1.725 | 1.739 |
| season stats -> last 5 starts | 1.740 | 2.177 | 1.732 | 1.744 |
| season stats -> last 6 starts | 1.739 | 2.176 | 1.734 | 1.742 |
| season stats -> last 7 starts | 1.739 | 2.176 | 1.731 | 1.744 |
| season stats -> last 8 starts | 1.739 | 2.175 | 1.730 | 1.744 |
| season stats -> last 9 starts | 1.738 | 2.174 | 1.728 | 1.744 |
| season stats -> last 10 starts | 1.738 | 2.173 | 1.728 | 1.744 |
| train on each pitcher's last 5 starts | 1.741 | 2.193 | 1.732 | 1.746 |
| train on each pitcher's last 6 starts | 1.738 | 2.188 | 1.730 | 1.742 |
| train on each pitcher's last 7 starts | 1.735 | 2.183 | 1.729 | 1.739 |
| train on each pitcher's last 8 starts | 1.734 | 2.181 | 1.729 | 1.737 |
| train on each pitcher's last 9 starts | 1.733 | 2.179 | 1.728 | 1.736 |
| train on each pitcher's last 10 starts | 1.732 | 2.178 | 1.727 | 1.735 |
| season fit, all earlier seasons | 1.737 | 2.174 | 1.727 | 1.742 |
| naive baseline | 1.877 | 2.350 | 1.922 | 1.852 |

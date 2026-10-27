# LSTM, validation results (2023-2024)

Seeds per model: 5. Test set not used.

## Grid (mean RMSE ratio vs. 'no change' over the six stock/horizon pairs, lower is better)

|               |   mean rmse_ratio |
|:--------------|------------------:|
| (40, 32, 0.2) |            1.0028 |
| (40, 16, 0.2) |            1.0044 |
| (10, 16, 0.2) |            1.0051 |
| (20, 32, 0.2) |            1.0071 |
| (20, 16, 0.2) |            1.0082 |
| (10, 32, 0.2) |            1.0087 |

Chosen config: `{"lookback": 40, "hidden_size": 32, "dropout": 0.2}`

## Chosen config, all 16 inputs

| stock   |   h |   rmse |   rmse_zero |   rmse_train_mean |   rmse_ratio |   dm_p_vs_zero |    mae |   direction_acc |   majority_dir |   pred_std |   member_rmse_std |   mean_best_epoch |
|:--------|----:|-------:|------------:|------------------:|-------------:|---------------:|-------:|----------------:|---------------:|-----------:|------------------:|------------------:|
| EQNR    |   3 | 0.0279 |      0.0278 |            0.0279 |       1.0019 |         0.8506 | 0.0218 |          0.4807 |         0.5092 |     0.0045 |            0      |               6   |
| EQNR    |   5 | 0.0352 |      0.0351 |            0.0353 |       1.0033 |         0.8434 | 0.0277 |          0.5031 |         0.5255 |     0.0063 |            0.0002 |               5.8 |
| EQNR    |  10 | 0.0465 |      0.0478 |            0.0485 |       0.9721 |         0.3402 | 0.0374 |          0.5479 |         0.5621 |     0.0138 |            0.004  |               9.2 |
| AKRBP   |   3 | 0.0283 |      0.028  |            0.0283 |       1.0082 |         0.2465 | 0.0221 |          0.501  |         0.501  |     0.0015 |            0.0001 |               1.6 |
| AKRBP   |   5 | 0.0353 |      0.0349 |            0.0355 |       1.0139 |         0.2164 | 0.0272 |          0.5132 |         0.5153 |     0.0021 |            0.0001 |               1.8 |
| AKRBP   |  10 | 0.0485 |      0.0467 |            0.0485 |       1.0398 |         0.0383 | 0.0373 |          0.4949 |         0.5051 |     0.0044 |            0.0003 |               2.4 |

## Input ablation (RMSE ratio vs. 'no change')

|               |   own stock |   + other stock |   + Oslo Børs |   + global (all 16) |
|:--------------|------------:|----------------:|--------------:|--------------------:|
| ('AKRBP', 3)  |      1.0086 |          1.0101 |        1.0059 |              1.0082 |
| ('AKRBP', 5)  |      1.024  |          1.0143 |        1.012  |              1.0139 |
| ('AKRBP', 10) |      1.028  |          1.027  |        1.0426 |              1.0398 |
| ('EQNR', 3)   |      1.0021 |          0.9985 |        1.0087 |              1.0019 |
| ('EQNR', 5)   |      1.004  |          1.0022 |        1.0175 |              1.0033 |
| ('EQNR', 10)  |      0.998  |          1.0081 |        1.0148 |              0.9721 |

Column guide: rmse_ratio < 1 beats predicting zero return. rmse_train_mean is the
RMSE of always predicting the average training return. dm_p_vs_zero is the
Diebold-Mariano p-value for equal MSE (Newey-West, lag h-1). majority_dir is the
direction accuracy of always guessing the more common sign on val. pred_std near 0
means the net predicts almost a constant. member_rmse_std is the spread over seeds.

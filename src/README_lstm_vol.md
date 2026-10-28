# LSTM volatility forecast (post-hoc extension)

The return LSTM in `src/README_lstm.md` did not beat "predict zero" for the h-day log
return of EQNR and AKRBP. The interval work on branch `prediksjonsintervall`
(`src/intervals.py`) found that the size of the moves is partly predictable: GARCH
and 20-day volatility give well-calibrated intervals. This extension asks whether the
same LSTM can forecast future volatility, and whether it beats simple volatility
baselines.

**This is a post-hoc extension.** The question was asked after the return results,
including the test set, were known. The return model is frozen and is not changed here.

## Run

```
pip install -r requirements.txt
python src/run_lstm_vol.py            # grid on val, then the final model on val
python src/run_lstm_vol.py --final    # final model only, from reports/lstm_vol/config.json
python src/run_lstm_vol.py --quick    # 2 seeds, one config (lookback 20, hidden 16), writes to reports/lstm_vol/quick/ (gitignored)
python src/run_lstm_vol.py --test     # ONCE, at the very end: the same final model on test
```

`--final` and `--test` cannot be combined. `--test` writes `metrics_test.json` and adds
test rows to the prediction files; `metrics_val.json` and `summary.md` are left as they are.

## Target

For a sample on day $t$ and horizon $h \in \{3, 5, 10\}$, with $r_t$ the daily
dividend-adjusted log return (`<STOCK>_ret`):

$$
\mathrm{RV}_{t,h} = \sqrt{\sum_{k=1}^{h} r_{t+k}^2},
\qquad z_{t,h} = \ln \mathrm{RV}_{t,h}
$$

Only returns after day $t$ enter the target; $r_t$ itself does not. Inputs $X$, sample
dates, the split and the input scaler come from `lstm_data()` in `datasets.py`, so they
are identical to the return models. Only $y$ is replaced by $z$. The code checks that
no target is NaN and that $t + h$ for every train sample comes before the first val
row (and the same for val against test), as for the return targets.

The model is `LSTMRegressor` from `src/lstm_model.py`, unchanged. It standardises the
target internally.

## Baselines

All forecasts are log volatility over $h$ days, so every model is on the same scale.

| Name | Forecast made on day $t$ |
|---|---|
| `zero` | mean of $z$ over the train samples. `LSTMRegressor` standardises the target with the train mean and std, so this is the forecast 0 in the model's own units, the counterpart of the zero-return forecast for the return model |
| `vol20` | $\ln\big(\sqrt{h}\,\hat\sigma_t\big)$, with $\hat\sigma_t$ the sample std of $r_{t-19}, \dots, r_t$ |
| `garch` | GARCH(1,1) with Student-t errors (`arch`) on $100\,r$. Parameters are fitted only on data before the first val date, the same rule as `intervals.py`. The $h$-day variance is the sum of the 1- to $h$-step variance forecasts made at $t$, using returns up to and including $t$; the forecast is $\ln(\sqrt{\text{var}}/100)$ |

## Grid and final model

Grid on val only: lookback 20 or 40, hidden size 16 or 32, dropout 0.2,
`BASE_FEATURES`, 5 seeds. One config is chosen for all six stock/horizon pairs: the
one with the lowest mean `ratio_vs_garch`. It is saved in `reports/lstm_vol/config.json`.
All other training settings are the `LSTMRegressor` defaults (see `src/README_lstm.md`).

## Metrics

Computed per stock and horizon, for the LSTM and for each baseline.

| Field | Meaning |
|---|---|
| `mse_log` | mean squared error of the log-volatility forecast (primary metric) |
| `qlike` | $\frac{1}{n}\sum \big(\mathrm{RV}^2/s^2 - \ln(\mathrm{RV}^2/s^2) - 1\big)$ with $s = e^{\text{forecast}}$. 0 for a perfect forecast; lower is better |
| `corr` | Pearson correlation between forecast and $z$ (`null` for `zero`) |
| `mean_forecast` | mean of the forecast over the split |
| `bias_sq`, `mse_rest` | `mse_log` split in two: `bias_sq` $= (\bar f - \bar z)^2$ is the error from a wrong average level, `mse_rest` $=$ `mse_log` $-$ `bias_sq` the error from not following the variation around it |

`results[STOCK]["h<h>"]["mean_z"]` holds the mean of $z$ on the split. `summary.md`
also shows `level_share_of_gain` $= (\text{bias\_sq}_\text{zero} - \text{bias\_sq}_\text{lstm}) / (\text{mse}_\text{zero} - \text{mse}_\text{lstm})$:
the share of the LSTM's MSE gain over `zero` that comes from getting the level right.

LSTM only:

| Field | Meaning |
|---|---|
| `ratio_vs_zero`, `ratio_vs_vol20`, `ratio_vs_garch` | `mse_log` of the LSTM divided by that of the baseline. Below 1 means the LSTM is better |
| `dm_p_vs_zero`, `dm_p_vs_vol20`, `dm_p_vs_garch` | two-sided Diebold-Mariano p-value on the difference in squared log errors. The targets overlap, so the variance uses Newey-West weights up to lag $h-1$ |
| `qlike_ratio_vs_zero` | `qlike` of the LSTM divided by that of `zero` |
| `seed_ratio_vs_zero_mean`, `seed_ratio_vs_zero_std` | mean and std over the seeds of each single model's `ratio_vs_zero` (from `predict_members`) |
| `mean_best_epoch` | mean early-stopping epoch over the seeds |

## Files

| File | Content |
|---|---|
| `src/run_lstm_vol.py` | target, baselines, metrics, grid, final step |
| `tests/test_lstm_vol.py` | target alignment, `vol20` window, GARCH fitted before val only, sample rows, `zero` = train mean of $z$, Diebold-Mariano, QLIKE |
| `reports/lstm_vol/grid_val.csv` | every grid config on every stock/horizon (LSTM metrics) |
| `reports/lstm_vol/config.json` | the chosen config |
| `reports/lstm_vol/metrics_val.json` | `config`, `n_seeds`, versions, the GARCH parameters, `split`, and `results[STOCK]["h<h>"][model]` with the fields above |
| `reports/lstm_vol/summary.md` | tables built from `grid_val.csv` and `metrics_val.json`, including "LSTM vs zero forecast (val)"; no interpretation |
| `reports/lstm_vol/eqnr_h5_val.png` | actual $z$ and the `lstm`, `garch` and `vol20` forecasts, EQNR h=5, val |

## Prediction files

`reports/predictions/lstmvol_<STOCK>_h<h>.csv`, one row per sample date:

| Column | Content |
|---|---|
| `Date` | sample date $t$ (forecast made at the close) |
| `logvol_pred` | LSTM forecast of $z_{t,h}$ |
| `sigma_pred` | `exp(logvol_pred)`, the forecast volatility over $h$ days as a log-return std |

The final step writes val rows only. `--test` rewrites the files with val and test rows.
With `--quick` the files go to `reports/lstm_vol/quick/predictions/`.

## Tests

```
python -m pytest -q tests/test_lstm_vol.py
```

Synthetic data only, no network, no training; a few seconds.

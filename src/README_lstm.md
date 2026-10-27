# LSTM model (Sigurd)

LSTM regressor for the h-day log return, trained on the shared dataset from
`data/processed/` so it can be compared directly with the random forest:
same target, same sample dates, same split, same metrics (`evaluate()` in `datasets.py`).

## Files

| File | Purpose |
|---|---|
| `src/lstm_model.py` | `LSTMRegressor` with `fit(X, y)` and `predict(X)`, same interface as the RF |
| `src/run_lstm.py` | Grid search and input ablation on the validation set, final model (`--final`), test run behind `--test` |
| `tests/test_lstm.py`, `tests/conftest.py` | Unit tests on synthetic data |
| `reports/lstm/` | Results: `summary.md`, `grid_val.csv`, `ablation_val.csv`, `config.json`, `metrics.json`, `val_predictions.csv` (`test_metrics.json` after `--test`) |
| `reports/predictions/lstm_<STOCK>_h<h>.csv` | Predictions per stock/horizon, input to `src/intervals.py` |

## Run

```
pip install -r requirements.txt
python src/run_lstm.py --quick    # check that it runs (about 2 min)
python src/run_lstm.py            # grid + input ablation + final model on val (about 15 to 30 min on a laptop CPU)
python src/run_lstm.py --final    # only the final model on val, from reports/lstm/config.json
python src/run_lstm.py --test     # ONCE, at the very end, after RF and LSTM configs are frozen
```

No GPU is needed. Results are reproducible for a given seed on the same machine
and PyTorch version, but small differences between machines are normal.

### Modes

| Mode | What it does | Writes |
|---|---|---|
| (default) | Grid on val, choose config, input ablation, then the final step | `grid_val.csv`, `config.json`, `ablation_val.csv`, plus everything the final step writes |
| `--final` | Final step only. Reads `reports/lstm/config.json` (`lookback`, `hidden_size`, `dropout`). No grid, no ablation. Also reads `grid_val.csv` and `ablation_val.csv` from an earlier full run, to rebuild `summary.md` | `metrics.json`, `val_predictions.csv`, val rows in `reports/predictions/lstm_<STOCK>_h<h>.csv`, regenerated `summary.md` |
| `--test` | Same code as the final step (same `final_step()`), but scores val and test. Reads `config.json`. Run once, at the very end | `test_metrics.json`, val + test rows in `reports/predictions/lstm_<STOCK>_h<h>.csv` |
| `--quick` | Modifier: 2 seeds, lookback 20, hidden 32 (unless set on the command line). Everything, including `config.json` and `predictions/`, goes to `reports/lstm/quick/` (gitignored) | same files as the mode it is combined with, under `reports/lstm/quick/` |

`--final` and `--test` cannot be combined. The default run ends with the same final
step as `--final`.

### Command-line flags

| Flag | Default | Meaning |
|---|---|---|
| `--quick` | off | see above |
| `--final` | off | final model on val, no grid, no ablation |
| `--test` | off | final model on val + test |
| `--lookbacks` | `10 20 40` | grid: lookback values (days) |
| `--hidden-sizes` | `16 32` | grid: hidden units |
| `--dropout` | `0.2` | grid: dropout. `--final` and `--test` use the dropout in `config.json` |
| `--num-layers` | `1` | LSTM layers |
| `--lr` | `0.001` | Adam learning rate |
| `--weight-decay` | `0.0001` | Adam weight decay |
| `--batch-size` | `64` | mini-batch size |
| `--max-epochs` | `200` | upper limit on epochs |
| `--patience` | `20` | early stopping patience (epochs) |
| `--es-fraction` | `0.15` | share of the training period used for early stopping |
| `--embargo` | `10` | days dropped between fit part and early-stopping part |
| `--no-refit-full` | off | skip retraining on all of train after early stopping |
| `--n-seeds` | 5 (2 with `--quick`) | models averaged per stock/horizon |
| `--seed` | `0` | base random seed |

### The final model

There is one final model definition, used by both the val run and `--test`:

- the config in `reports/lstm/config.json` (`lookback`, `hidden_size`, `dropout`),
- the inputs in `BASE_FEATURES` order (`FINAL_FEATURES` in `run_lstm.py`),
- the prediction is the mean over `n_seeds` models.

The input ablation is a separate study. It builds its own input sets (own stock, then
adding other stock, Oslo Børs factors, global markets) with its own column order,
so it does not feed `val_predictions.csv`, `metrics.json` or the prediction files.

## Tests

```
pip install -r requirements.txt
python -m pytest -q
```

The tests use synthetic data only and need no network. They take about 25 s.
They cover: array shapes, window alignment and no look-ahead, label alignment in the
dataset build, no leak between train/val/test, scaler fitted on train only, the net
overfitting a tiny batch, learning a planted signal, same seed gives same
predictions, the Diebold-Mariano test, and the `ValueError` guards in `fit()`
(`n_seeds < 1`, X/y length mismatch, empty or constant `y`, too few samples).

## Use in `train.py`

```python
import sys; sys.path[:0] = ["src", "data/processed"]
from datasets import load, lstm_data, evaluate
from lstm_model import LSTMRegressor

df = load()
d = lstm_data(df, "EQNR", h=5, lookback=20)
X_train, y_train, _ = d["train"]
X_val, y_val, _ = d["val"]

model = LSTMRegressor(hidden_size=32, dropout=0.2).fit(X_train, y_train)
print(evaluate(y_val, model.predict(X_val)))
```

`fit()` only needs the training data. Early stopping uses the last 15 % of the
training period internally, so the validation set is never used to fit weights.

## Model

For each sample the network reads the last $T$ trading days
$x_1, \dots, x_T \in \mathbb{R}^{16}$ (scaled inputs, $T$ = lookback) and updates
its cell state $c_t$ and hidden state $h_t$:

$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) && \text{forget gate} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) && \text{input gate} \\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g) && \text{candidate values} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) && \text{output gate} \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

The forecast uses only the last hidden state (many-to-one):

$$
\hat{y}_{t,h} = w^\top \operatorname{dropout}(h_T) + b,
\qquad y_{t,h} = \ln\frac{P_{t+h}}{P_t}
$$

The additive cell update $c_t = f_t \odot c_{t-1} + \dots$ is why an LSTM can keep
information over many steps: the gradient through $c_t$ is scaled by $f_t$, not
repeatedly multiplied by the same weight matrix as in a plain RNN, which reduces
vanishing gradients.

Training minimises the mean squared error on the standardised target
$\tilde{y} = (y - \bar{y}_{\text{train}}) / s_{\text{train}}$ with Adam
(learning rate $10^{-3}$, weight decay $10^{-4}$, batch 64, gradient norm clipped at 1).

Size with 16 inputs and $H = 32$ hidden units:
$4\,(H(D+H) + 2H) + H + 1 = 4\,(32 \cdot 48 + 64) + 33 = 6\,433$ weights.
PyTorch has two bias vectors per gate, hence $2H$.

## Design choices

| Choice | Reason |
|---|---|
| One layer, 16 or 32 hidden units | About 1 670 training windows and a very noisy target. Larger nets fit noise. |
| Regression on log returns, not prices | Prices drift (non-stationary). A model on price levels looks good only because $P_{t+h} \approx P_t$. |
| Target standardised | Raw returns are around 0.01 to 0.05, which gives very small MSE gradients. Undone in `predict()`. |
| Early stopping on the last 15 % of train, 10-day embargo | Validation stays independent of training. The embargo stops overlapping h-day targets from leaking into the stopping set. |
| Retrain on all of train for the found number of epochs | The 15 % held out for early stopping is the most recent data, too valuable to drop. |
| Mean over 5 random seeds | Small nets on noisy data vary a lot between initialisations. Averaging reduces this variance. |
| One config for all six stock/horizon pairs | Choosing a separate config for each pair would fit the validation set six times over. |
| Inputs clipped at ±5 std (in `lstm_data`) | Real extreme days (March 2020) are kept, but cannot dominate the gradients. |
| Shuffled mini-batches | Each window is an independent sample (state is reset per window), so order inside training does not matter. |

## Evaluation

All results in `reports/lstm/summary.md` are on the validation set (2023 to 2024).
The test set (2025 to 2026) is not used until `--test`.

- `rmse_ratio` = RMSE(LSTM) / RMSE(predict zero). Below 1 beats "no change".
- `rmse_train_mean` = RMSE of always predicting the average training return.
- `dm_p_vs_zero` = Diebold-Mariano test for equal MSE against "no change". The
  h-day targets overlap, so errors are autocorrelated up to lag $h-1$; the variance
  uses Newey-West weights.
- `majority_dir` = direction accuracy from always guessing the most common sign.
  Note: `evaluate()` counts a prediction of exactly 0 as "down", so the zero
  baseline's direction accuracy is the share of down periods.
- The input ablation adds input groups one at a time: own stock, other stock,
  Oslo Børs factors, global markets. It is a separate study (see "The final model").

`reports/lstm/val_predictions.csv` holds the val predictions of the final model
(columns `Date, stock, h, y_true, y_pred`).

## `metrics.json`

Written by the final step (`--final` or the default run) to `reports/lstm/metrics.json`.
`--test` writes `test_metrics.json` with the same layout (see below).

| Key | Content |
|---|---|
| `config` | `lookback`, `hidden_size`, `dropout` of the final model |
| `training_settings` | `num_layers`, `lr`, `weight_decay`, `batch_size`, `max_epochs`, `patience`, `es_fraction`, `embargo`, `n_seeds`, `seed`, `refit_full` |
| `n_seeds` | number of models averaged |
| `torch_version` | PyTorch version used |
| `split` | `"val"` (`"test"` in `test_metrics.json`) |
| `results` | `results[STOCK]["h<h>"]`, one entry per stock and horizon, fields below |

Fields per stock/horizon:

| Field | Meaning |
|---|---|
| `rmse`, `mae` | error of the model on the split |
| `direction_acc` | share of correct up/down calls |
| `direction_balanced_acc` | balanced accuracy of the up/down calls |
| `confusion_matrix` | `[[TN, FP], [FN, TP]]`: rows = actual [down, up], columns = predicted [down, up]. Up means `y > 0` (actual) or `pred > 0` (predicted), so 0 counts as down |
| `rmse_zero` | RMSE of always predicting zero return |
| `rmse_train_mean` | RMSE of always predicting the average training return |
| `rmse_ratio` | `rmse / rmse_zero` |
| `dm_p_vs_zero` | Diebold-Mariano p-value against "no change" (`null` if undefined) |
| `majority_dir` | direction accuracy of always guessing the most common sign |
| `momentum_dir` | baseline with `direction_acc`, `direction_balanced_acc`, `confusion_matrix` (same layout). Predicted direction = sign of `log P[t] - log P[t-h]`, past data only (adjusted close); a change of exactly 0 counts as down |

`test_metrics.json` has the same keys, with `split = "test"` and `results` on the test
set. It also has a `val` key with the val results of the same run, same layout as `results`.

## Prediction files for `src/intervals.py`

`reports/predictions/lstm_<STOCK>_h<h>.csv` has the columns `Date,pred`, one row per
sample date. `--final` (and the default run) writes val rows only. `--test` overwrites
the files with val + test rows, because the conformal step needs both. With `--quick`
the files go to `reports/lstm/quick/predictions/` instead.

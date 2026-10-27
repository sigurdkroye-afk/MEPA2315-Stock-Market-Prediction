# LSTM model (Sigurd)

LSTM regressor for the h-day log return, trained on the shared dataset from
`data/processed/` so it can be compared directly with the random forest:
same target, same sample dates, same split, same metrics (`evaluate()` in `datasets.py`).

## Files

| File | Purpose |
|---|---|
| `src/lstm_model.py` | `LSTMRegressor` with `fit(X, y)` and `predict(X)`, same interface as the RF |
| `src/run_lstm.py` | Grid search and input ablation on the validation set, test run behind `--test` |
| `reports/lstm/` | Results: `summary.md`, CSV tables, val predictions, chosen config |

## Run

```
pip install pandas numpy scikit-learn torch tabulate
python src/run_lstm.py --quick    # check that it runs (about 2 min)
python src/run_lstm.py            # full validation run (about 15 to 30 min on a laptop CPU)
python src/run_lstm.py --test     # ONCE, at the end, after RF and LSTM configs are frozen
```

No GPU is needed. Results are reproducible for a given seed on the same machine
and PyTorch version, but small differences between machines are normal.

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
  Oslo Børs factors, global markets.

`reports/lstm/val_predictions.csv` holds the val predictions of the final model,
which can be used for conformal prediction intervals.

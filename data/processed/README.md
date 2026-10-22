# Shared data pipeline, MEPA2315 stock project

One cleaned dataset and one set of loaders for both models, so RF and LSTM are
trained and tested on exactly the same rows, targets and split.

## Files

```
data/raw/panel_long.csv        Titlon daily data for EQNR and AKRBP (input)
data/raw/market.csv            Brent, VIX, S&P 500, USD/NOK (input, lagged one trading day)
data/processed/dataset.csv     output of prepare_data.py, do not edit by hand
prepare_data.py                cleaning, targets and split
datasets.py                    loaders for RF and LSTM, shared metrics and baselines
```

## Run

```
pip install pandas numpy scikit-learn
python prepare_data.py
```

Do not open and save the CSV files in Excel. With Norwegian settings Excel
rewrites the separators and the data breaks.

## What prepare_data.py does

1. **Removes closed days.** Titlon includes 30 holidays from December 2020
   onward (Christmas Eve, May 17, Ascension Day and so on) with Close = 0 and
   Volume = 0 but a filled-in price. Only days when both stocks traded are kept.
2. **Recomputes returns** from `AdjustedPrice` (dividend and split adjusted)
   after step 1, so the full move across a holiday lands on the next trading day.
   Market changes are recomputed from levels the same way.
3. **Drops columns with gaps or errors.** `NumberOfTrades` (missing 2016 to 2020),
   `HML` (missing from June 2026), and `NumberOfShares` and `mktcap`
   (capped at 2 147 483 647 for Equinor, the int32 maximum, in the Titlon data).
4. **Transforms features.** Log returns, intraday range log(High/Low), and volume
   relative to its own 20-day average (raw volume drifts as the share count grows).
5. **Builds targets.** `y_<STOCK>_h<h>` = log(AdjustedPrice[t+h] / AdjustedPrice[t])
   for h = 3, 5, 10. Direct forecast from information at the close of day t.
6. **Labels the split** by date (see below) with a 10-day gap before val and test.

Outliers are kept. Extreme days such as March 2020 are real market events, and
removing them would make the models underestimate risk. They are dampened in
the LSTM input only, by clipping scaled inputs at five standard deviations.
RF splits on thresholds and is not affected by extreme values.

## Split

| part  | dates                         |
|-------|-------------------------------|
| train | Feb 2016 to mid Dec 2022      |
| val   | Jan 2023 to mid Dec 2024      |
| test  | Jan 2025 to Sep 2026          |

The last 10 trading days before val and before test are labeled `gap` and never
used as samples, because their 10-day targets would use prices from the next
period. Change the dates only in `prepare_data.py` and rerun it.

Use `val` for tuning and model choice. Run on `test` once, at the end.

## Use

```python
from datasets import load, rf_data, lstm_data, evaluate, baselines

df = load()

# Random forest
d = rf_data(df, "EQNR", h=5)
X_train, y_train, dates_train = d["train"]

# LSTM, X has shape (samples, lookback, features)
d = lstm_data(df, "EQNR", h=5, lookback=20)
X_train, y_train, dates_train = d["train"]

# Same metrics and the same naive forecasts for every model
X_val, y_val, _ = d["val"]
for name, pred in baselines(y_train, len(y_val)).items():
    print(name, evaluate(y_val, pred))
```

Both loaders skip the first 60 rows (history only), so RF and LSTM use
identical sample dates in every split. The LSTM scaler is fitted on train only.

To turn a return forecast into a price forecast, use the actual close:
`price_t+h = <STOCK>_close[t] * exp(y_hat)`.

## Not included yet

The project proposal lists the Oslo Børs energy sector index and indices from
other sectors. They are not in the raw files yet. Add them to `data/raw/`, join
them in `prepare_data.py` like `market.csv`, and add their returns to
`BASE_FEATURES` in `datasets.py`.

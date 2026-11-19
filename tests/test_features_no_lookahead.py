"""No look-ahead in the input variables.

For a handful of dates t spread over train, val and test, every raw value dated
after t (prices, volume, high/low, Titlon market columns and market.csv) is
multiplied by random noise, and the dataset is rebuilt with prepare_data.build().
The inputs for day t must not change:

  * the 37 tabular features on row t (rf_data, used by trees and lasso)
  * the 16 LSTM features on row t, and the raw LSTM window ending at t
  * the scaled LSTM window from lstm_data() ending at t

The targets y_h on row t must change, which shows the perturbation reaches the data.

Unlike the other tests this one reads the raw files under data/processed/data/raw
(read only) and is skipped if they are missing.
"""
import numpy as np
import pandas as pd
import pytest

import prepare_data
from datasets import BASE_FEATURES, PARTS, STOCKS, WARMUP, lstm_data, rf_data

HORIZONS = prepare_data.HORIZONS
TARGETS = [f"y_{s}_h{h}" for s in STOCKS for h in HORIZONS]
LOOKBACK = WARMUP + 1                      # the longest window lstm_data allows
# Fractions of the sample rows of each part where t is placed.
DATE_PICKS = [("train", 0.3), ("train", 0.8), ("val", 0.5), ("test", 0.2), ("test", 0.7)]

pytestmark = pytest.mark.skipif(
    not (prepare_data.RAW_STOCKS.exists() and prepare_data.RAW_MARKET.exists()),
    reason="raw data not available")


@pytest.fixture(scope="module")
def raw():
    return {"stocks": pd.read_csv(prepare_data.RAW_STOCKS, parse_dates=["Date"]),
            "market": pd.read_csv(prepare_data.RAW_MARKET, parse_dates=["Date"])}


@pytest.fixture(scope="module")
def original():
    return prepare_data.build()


def pick_dates(df):
    """Five sample dates t: rows that rf_data/lstm_data use for every stock and horizon."""
    samples = rf_data(df, STOCKS[0], max(HORIZONS))
    out = []
    for part, frac in DATE_PICKS:
        dates = samples[part][2]
        out.append(dates[int(frac * (len(dates) - 1))])
    return out


def perturb_after(frame, t, rng):
    """Copy with every numeric value dated after t multiplied by a random factor.
    The factor is positive, so closed days (Close or Volume <= 0) stay closed."""
    out = frame.copy()
    after = out["Date"] > t
    cols = out.select_dtypes("number").columns
    out[cols] = out[cols].astype(float)     # Volume etc. are int; same values before t
    noise = np.exp(rng.normal(0.0, 0.2, size=(after.sum(), len(cols))))
    out.loc[after, cols] = out.loc[after, cols].to_numpy() * noise
    return out


def rebuild_perturbed(raw, t, tmp_path, monkeypatch, seed):
    rng = np.random.default_rng(seed)
    stocks, market = tmp_path / "panel_long.csv", tmp_path / "market.csv"
    perturb_after(raw["stocks"], t, rng).to_csv(stocks, index=False)
    perturb_after(raw["market"], t, rng).to_csv(market, index=False)
    monkeypatch.setattr(prepare_data, "RAW_STOCKS", stocks)
    monkeypatch.setattr(prepare_data, "RAW_MARKET", market)
    return prepare_data.build()


def part_of(df, t):
    return df.at[t, "split"]


def lstm_window(d, t):
    part = next(p for p in PARTS if t in d[p][2])
    i = d[part][2].get_loc(t)
    return d[part][0][i]


@pytest.fixture(scope="module")
def cases(raw, original, tmp_path_factory):
    """(t, perturbed dataset) for each picked date. Built once for all tests."""
    mp = pytest.MonkeyPatch()
    try:
        out = []
        for k, t in enumerate(pick_dates(original)):
            tmp = tmp_path_factory.mktemp(f"raw_{k}")
            out.append((t, rebuild_perturbed(raw, t, tmp, mp, seed=k)))
        return out
    finally:
        mp.undo()


def test_dates_cover_train_val_test(original):
    parts = [part_of(original, t) for t in pick_dates(original)]
    assert parts == [p for p, _ in DATE_PICKS]


def test_rows_up_to_t_unchanged(original, cases):
    for t, new in cases:
        assert new.loc[:t].index.equals(original.loc[:t].index), t
        assert new.at[t, "split"] == original.at[t, "split"]


def test_targets_on_row_t_change(original, cases):
    """The test has teeth: every y_h on row t uses prices after t and must move."""
    for t, new in cases:
        for col in TARGETS:
            assert new.at[t, col] != original.at[t, col], f"{col} unchanged at {t.date()}"


@pytest.mark.parametrize("stock", STOCKS)
@pytest.mark.parametrize("h", HORIZONS)
def test_tabular_features_on_row_t_unchanged(original, cases, stock, h):
    """All 37 rf_data columns on row t (used by trees and lasso)."""
    for t, new in cases:
        part = part_of(original, t)
        X0, X1 = rf_data(original, stock, h)[part][0], rf_data(new, stock, h)[part][0]
        assert X0.shape[1] == 37
        assert list(X1.columns) == list(X0.columns)
        leaks = [c for c in X0.columns if not X1.at[t, c] == X0.at[t, c]]
        assert not leaks, f"look-ahead at {t.date()} ({part}) in: {leaks}"


def test_lstm_features_and_raw_window_unchanged(original, cases):
    """The 16 LSTM columns on row t, and on the LOOKBACK rows ending at t."""
    assert len(BASE_FEATURES) == 16
    for t, new in cases:
        i = original.index.get_loc(t)
        w0 = original[BASE_FEATURES].iloc[i - LOOKBACK + 1:i + 1]
        w1 = new[BASE_FEATURES].iloc[i - LOOKBACK + 1:i + 1]
        assert w1.index.equals(w0.index)
        leaks = [c for c in BASE_FEATURES if not (w1[c].to_numpy() == w0[c].to_numpy()).all()]
        assert not leaks, f"look-ahead in LSTM window ending {t.date()} in: {leaks}"


def test_scaled_lstm_window_unchanged_outside_train(original, cases):
    """lstm_data() builds the window the LSTMs see. For t in val or test every
    training row is before t, so the scaler and the scaled window must be identical."""
    for t, new in cases:
        if part_of(original, t) == "train":
            continue
        for stock in STOCKS:
            for h in HORIZONS:
                d0 = lstm_data(original, stock, h, lookback=LOOKBACK)
                d1 = lstm_data(new, stock, h, lookback=LOOKBACK)
                w0, w1 = lstm_window(d0, t), lstm_window(d1, t)
                leaks = [c for j, c in enumerate(BASE_FEATURES) if not (w1[:, j] == w0[:, j]).all()]
                assert not leaks, f"look-ahead in scaled window ending {t.date()} in: {leaks}"


def test_scaled_lstm_window_in_train_depends_on_future_only_via_scaler(original, cases):
    """For t in train the StandardScaler is fitted on all training rows, also those
    after t, so the scaled window can change. Check that this is the only route:
    the new window equals the original raw window put through the new scaler."""
    for t, new in cases:
        if part_of(original, t) != "train":
            continue
        d1 = lstm_data(new, STOCKS[0], max(HORIZONS), lookback=LOOKBACK)
        i = original.index.get_loc(t)
        raw0 = original[BASE_FEATURES].iloc[i - LOOKBACK + 1:i + 1]
        expected = np.clip(d1["scaler"].transform(raw0).astype(np.float32), -5.0, 5.0)
        np.testing.assert_array_equal(lstm_window(d1, t), expected)

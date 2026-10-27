"""Tests for the LSTM pipeline. Tiny models and synthetic data, no real dataset."""
import numpy as np
import pandas as pd
import pytest
import torch

import datasets
import prepare_data
from datasets import BASE_FEATURES, STOCKS, lstm_data
from lstm_model import LSTMRegressor, _Net
from run_lstm import dm_test

LOOKBACK = 20
HORIZONS = [3, 5, 10]


def tiny_model(**kw):
    args = dict(hidden_size=4, dropout=0.0, lr=1e-2, batch_size=32, max_epochs=3,
                patience=2, n_seeds=2, seed=0)
    args.update(kw)
    return LSTMRegressor(**args)


def signal_data(n, seed):
    """Windows where y depends on the last-step features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, LOOKBACK, 16)).astype(np.float32)
    y = (0.02 * X[:, -1, 0] - 0.01 * X[:, -1, 1] + rng.normal(0, 0.002, n)).astype(np.float32)
    return X, y


# a. shapes
def test_shapes(df):
    d = lstm_data(df, "EQNR", 5, lookback=LOOKBACK)
    X, y, dates = d["train"]
    assert X.shape == (len(y), LOOKBACK, 16)
    assert len(dates) == len(y) > 0
    m = tiny_model().fit(X, y)
    Xv = d["val"][0]
    assert m.predict(Xv).shape == (len(Xv),)
    assert m.predict_members(Xv).shape == (2, len(Xv))


# b. window alignment
@pytest.mark.parametrize("stock", STOCKS)
@pytest.mark.parametrize("h", HORIZONS)
def test_window_alignment(df, stock, h):
    d = lstm_data(df, stock, h, lookback=LOOKBACK, clip=5.0)
    sc = d["scaler"]
    F = np.clip((df[BASE_FEATURES].to_numpy() - sc.mean_) / sc.scale_, -5.0, 5.0)
    for part in datasets.PARTS:
        X, y, dates = d[part]
        for i in (0, len(y) // 2, len(y) - 1):
            pos = df.index.get_loc(dates[i])
            np.testing.assert_allclose(X[i, -1], F[pos], atol=1e-5)
            np.testing.assert_allclose(X[i, 0], F[pos - (LOOKBACK - 1)], atol=1e-5)
            assert y[i] == pytest.approx(df.loc[dates[i], f"y_{stock}_h{h}"], abs=1e-6)


# c. label alignment in prepare_data.build()
def write_raw(tmp_path, n=160, closed_pos=40):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-04", periods=n, name="Date")
    rows = []
    for s in STOCKS:
        adj = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        d = pd.DataFrame({
            "Date": dates, "Ticker": s, "Close": adj * 1.01,
            "Volume": rng.integers(1000, 5000, n).astype(float),
            "AdjustedPrice": adj, "High": adj * 1.02, "Low": adj * 0.98,
            "lnDeltaOSEBX": rng.normal(0, 0.01, n), "lnDeltaOBX": rng.normal(0, 0.01, n),
            "NOWA_DayLnrate": 0.01, "SMB": rng.normal(0, 0.01, n),
            "LIQ": rng.normal(0, 0.01, n), "MOM": rng.normal(0, 0.01, n),
        })
        d.loc[closed_pos, ["Close", "Volume"]] = 0.0     # closed day, price filled in
        rows.append(d)
    panel = pd.concat(rows)
    market = pd.DataFrame({
        "Date": dates,
        "brent_usd": 80 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
        "vix": 20 + rng.normal(0, 1, n),
        "sp500": 4000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "usdnok": 10 * np.exp(np.cumsum(rng.normal(0, 0.005, n))),
    })
    p_stocks, p_market = tmp_path / "panel_long.csv", tmp_path / "market.csv"
    panel.to_csv(p_stocks, index=False)
    market.to_csv(p_market, index=False)
    return p_stocks, p_market, dates, panel


def test_build_label_alignment(tmp_path, monkeypatch):
    closed_pos = 40
    p_stocks, p_market, dates, panel = write_raw(tmp_path, closed_pos=closed_pos)
    monkeypatch.setattr(prepare_data, "RAW_STOCKS", p_stocks)
    monkeypatch.setattr(prepare_data, "RAW_MARKET", p_market)
    monkeypatch.setattr(prepare_data, "TRAIN_END", str(dates[85].date()))
    monkeypatch.setattr(prepare_data, "VAL_END", str(dates[125].date()))

    out = prepare_data.build()
    assert dates[closed_pos] not in out.index
    prepare_data.check(out)

    for s in STOCKS:
        p = panel[(panel["Ticker"] == s) & (panel["Close"] > 0)].set_index("Date")["AdjustedPrice"]
        logp = np.log(p)
        for h in HORIZONS:
            expected = (logp.shift(-h) - logp).reindex(out.index)
            np.testing.assert_allclose(out[f"y_{s}_h{h}"].to_numpy(), expected.to_numpy(),
                                       atol=1e-9, equal_nan=True)


# d. split: no overlap, targets do not reach into the next part
@pytest.mark.parametrize("h", HORIZONS)
def test_split_no_leak(df, h):
    d = lstm_data(df, "EQNR", h, lookback=LOOKBACK)
    dates = {p: d[p][2] for p in datasets.PARTS}
    assert not set(dates["train"]) & set(dates["val"])
    assert not set(dates["train"]) & set(dates["test"])
    assert not set(dates["val"]) & set(dates["test"])
    for part, nxt in (("train", "val"), ("val", "test")):
        pos = df.index.get_indexer(dates[part])
        target_dates = df.index[pos + h]
        assert target_dates.max() < dates[nxt].min()


# e. scaler fitted on train only
def test_scaler_train_only(df):
    d = lstm_data(df, "EQNR", 5, lookback=LOOKBACK)
    train_mean = df.loc[df["split"] == "train", BASE_FEATURES].mean().to_numpy()
    np.testing.assert_allclose(d["scaler"].mean_, train_mean, rtol=1e-9, atol=1e-12)

    df2 = df.copy()
    df2.loc[df2["split"] != "train", BASE_FEATURES] *= 100
    d2 = lstm_data(df2, "EQNR", 5, lookback=LOOKBACK)
    np.testing.assert_array_equal(d["scaler"].mean_, d2["scaler"].mean_)
    np.testing.assert_array_equal(d["scaler"].scale_, d2["scaler"].scale_)
    np.testing.assert_array_equal(d["train"][0], d2["train"][0])


# f. the network can overfit a tiny set
def test_net_overfits_small_batch():
    torch.manual_seed(0)
    X = torch.randn(16, LOOKBACK, 16)
    y = torch.randn(16)
    net = _Net(16, 16, 1, 0.0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    first = loss_fn(net(X), y).item()
    for _ in range(500):
        opt.zero_grad()
        loss = loss_fn(net(X), y)
        loss.backward()
        opt.step()
    assert loss_fn(net(X), y).item() < 0.05 * first


# g. learns a signal
def test_learns_signal():
    X_tr, y_tr = signal_data(300, 1)
    X_te, y_te = signal_data(150, 2)
    m = tiny_model(hidden_size=8, max_epochs=40, patience=40).fit(X_tr, y_tr)
    pred = m.predict(X_te)
    assert not np.isnan(pred).any()
    rmse = np.sqrt(np.mean((pred - y_te) ** 2))
    rmse_mean = np.sqrt(np.mean((y_tr.mean() - y_te) ** 2))
    assert rmse < 0.7 * rmse_mean


# h. determinism
def test_same_seed_same_predictions():
    X, y = signal_data(120, 3)
    a = tiny_model(seed=5).fit(X, y).predict(X)
    b = tiny_model(seed=5).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


# i. Diebold-Mariano
def test_dm_test():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.05, 200)
    pred = y + rng.normal(0, 0.005, 200)
    assert np.isnan(dm_test(y, pred, pred.copy(), 5))
    assert dm_test(y, pred, np.zeros(200), 5) < 0.01


# ValueError guards in fit()
def test_fit_guards():
    X, y = signal_data(100, 4)
    with pytest.raises(ValueError):
        tiny_model(n_seeds=0).fit(X, y)
    with pytest.raises(ValueError):
        tiny_model().fit(X, y[:-1])
    with pytest.raises(ValueError):
        tiny_model().fit(X, np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError):
        tiny_model().fit(X[:0], y[:0])
    with pytest.raises(ValueError):
        tiny_model().fit(X[:5], y[:5])                  # fit part empty after embargo
    with pytest.raises(ValueError):
        tiny_model(es_fraction=0.0).fit(X, y)           # empty early-stopping part

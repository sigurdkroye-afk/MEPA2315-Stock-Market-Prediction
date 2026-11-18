"""Tests for src/run_lstm_vol.py. Synthetic data only, no training."""
import numpy as np
import pandas as pd
import pytest

from run_lstm_vol import (dm_test, garch_logvol, level_ref, log_rv, qlike, vol20, vol_data, vol_metrics,
                          zero_level)


def _ret(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.02, n), index=pd.bdate_range("2021-01-01", periods=n))


# a. target alignment
@pytest.mark.parametrize("h", [3, 5, 10])
def test_log_rv_uses_only_future_returns(h):
    ret = _ret()
    z = log_rv(ret, h)
    t = 7
    expected = np.log(np.sqrt(np.sum(ret.iloc[t + 1:t + h + 1].to_numpy() ** 2)))
    assert z.iloc[t] == pytest.approx(expected)

    changed = ret.copy()
    changed.iloc[t] *= 50                     # ret[t] itself must not matter
    assert log_rv(changed, h).iloc[t] == pytest.approx(z.iloc[t])
    assert z.iloc[-h:].isna().all()           # no target without h future days


# b. vol20 uses only ret[t-19..t]
def test_vol20_uses_only_past_20_days():
    ret = _ret(60)
    t, h = 40, 5
    v = vol20(ret, h)
    window = ret.iloc[t - 19:t + 1].to_numpy()
    assert v.iloc[t] == pytest.approx(np.log(np.sqrt(h) * window.std(ddof=1)))

    for k in (t - 20, t + 1):                 # just outside the window on both sides
        changed = ret.copy()
        changed.iloc[k] *= 50
        assert vol20(changed, h).iloc[t] == pytest.approx(v.iloc[t])


# c. GARCH parameters come only from rows before the first val date
def test_garch_fitted_before_val_only(df):
    rng = np.random.default_rng(1)
    for s in ("EQNR", "AKRBP"):
        df[f"{s}_ret"] = rng.normal(0, 0.02, len(df))
    _, res = garch_logvol(df, "EQNR")

    # Everything from the first val date on: val, the gap before test, and test.
    later = np.arange(len(df)) >= np.flatnonzero(df["split"] == "val")[0]
    changed = df.copy()
    changed.loc[later, "EQNR_ret"] *= 3
    _, res2 = garch_logvol(changed, "EQNR")
    np.testing.assert_allclose(res.params.to_numpy(), res2.params.to_numpy())


def test_vol_data_rows_match_return_models(df):
    rng = np.random.default_rng(2)
    df["EQNR_ret"] = rng.normal(0, 0.02, len(df))
    d = vol_data(df, "EQNR", 5, lookback=20)
    X, z, dates = d["val"]
    assert X.shape == (len(dates), 20, 16)
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z, log_rv(df["EQNR_ret"], 5).loc[dates].to_numpy())


# The zero forecast = train mean of z, untouched by val and test data
def test_zero_forecast_is_train_mean(df):
    rng = np.random.default_rng(4)
    df["EQNR_ret"] = rng.normal(0, 0.02, len(df))
    d = vol_data(df, "EQNR", 5, lookback=20)
    train_dates = d["train"][2]
    assert zero_level(d) == pytest.approx(log_rv(df["EQNR_ret"], 5).loc[train_dates].mean())

    changed = df.copy()
    changed.loc[changed["split"].isin(["val", "test"]), "EQNR_ret"] *= 5
    assert zero_level(vol_data(changed, "EQNR", 5, lookback=20)) == pytest.approx(zero_level(d))

    loss = (d["val"][1] - zero_level(d)) ** 2
    assert np.isnan(dm_test(loss, loss, 5))


# d. Diebold-Mariano
def test_dm_test():
    rng = np.random.default_rng(3)
    loss = rng.chisquare(1, 500)
    assert np.isnan(dm_test(loss, loss, 5))
    assert dm_test(0.2 * loss, loss, 5) < 0.01


def test_mse_decomposition():
    rng = np.random.default_rng(5)
    z = rng.normal(-3.5, 0.4, 200)
    m = vol_metrics(z, np.full(200, -3.2))      # constant forecast: only the level differs
    assert m["bias_sq"] == pytest.approx((-3.2 - z.mean()) ** 2)
    assert m["mse_rest"] == pytest.approx(z.var())
    m = vol_metrics(z, z + rng.normal(0, 0.1, 200))
    assert m["bias_sq"] + m["mse_rest"] == pytest.approx(m["mse_log"])


# e. QLIKE
def test_qlike_zero_only_for_perfect_forecast():
    z = np.log(np.array([0.01, 0.02, 0.05]))
    assert qlike(z, z) == pytest.approx(0.0)
    assert qlike(z, z + 0.1) > 0
    assert qlike(z, z - 0.1) > 0


# f. level-adjusted reference: mean of the last 250 z known at t (windows ending <= t)
@pytest.mark.parametrize("h", [3, 5, 10])
def test_level_ref_uses_only_z_known_at_t(h):
    ret = _ret(700, seed=6)
    t = 600
    ref = level_ref(ret, h)
    z = log_rv(ret, h)
    assert ref.iloc[t] == pytest.approx(z.iloc[t - h - 249:t - h + 1].mean())

    # The newest z in the window, z[t-h], uses ret[t-h+1..t]: ret[t] matters ...
    changed = ret.copy()
    changed.iloc[t] *= 50
    assert level_ref(changed, h).iloc[t] != pytest.approx(ref.iloc[t])
    # ... but nothing after t does
    changed = ret.copy()
    changed.iloc[t + 1:] *= 50
    assert level_ref(changed, h).iloc[t] == pytest.approx(ref.iloc[t])
    # and nothing before the oldest window, ret[t-h-248], either
    changed = ret.copy()
    changed.iloc[:t - h - 248] *= 50
    assert level_ref(changed, h).iloc[t] == pytest.approx(ref.iloc[t])

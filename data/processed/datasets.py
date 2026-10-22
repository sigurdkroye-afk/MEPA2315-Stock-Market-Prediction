"""
datasets.py

Shared loaders for the RF and LSTM experiments. Both loaders use the same
rows, targets and split, so results can be compared directly.

    from datasets import load, rf_data, lstm_data, evaluate

    df = load()
    rf = rf_data(df, "EQNR", h=5)          # pandas X, y per split
    nn = lstm_data(df, "EQNR", h=5)        # numpy (samples, lookback, features), y per split

Each loader returns {"train": (X, y, dates), "val": (...), "test": (...)}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent / "data" / "processed" / "dataset.csv"
STOCKS = ["EQNR", "AKRBP"]
PARTS = ("train", "val", "test")

# Daily inputs known at the close of day t. Price levels are left out on purpose
# because they drift over time; use returns instead. The interest rate `nowa` is
# also left out: it averages 0.7 % p.a. in train but 4.4 to 4.6 % in val and test,
# entirely outside the range the models are trained on.
BASE_FEATURES = [
    "EQNR_ret", "EQNR_relvol", "EQNR_range",
    "AKRBP_ret", "AKRBP_relvol", "AKRBP_range",
    "osebx_ret", "obx_ret", "brent_ret", "sp500_ret", "usdnok_ret",
    "vix", "vix_chg", "smb", "liq", "mom",
]

# The first WARMUP rows are history only and never used as samples. This makes
# RF and LSTM use identical sample rows. LSTM lookback must be at most WARMUP + 1.
WARMUP = 60


def load(path: Path = DATA) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def target(stock: str, h: int) -> str:
    return f"y_{stock}_h{h}"


def _sample_mask(df: pd.DataFrame, y: pd.Series, part: str) -> np.ndarray:
    mask = (df["split"] == part).to_numpy() & y.notna().to_numpy()
    mask[:WARMUP] = False
    return mask


def rf_data(df: pd.DataFrame, stock: str, h: int) -> dict:
    """Tabular features for random forest. No scaling needed."""
    X = df[BASE_FEATURES].copy()
    for s in STOCKS:
        r = df[f"{s}_ret"]
        for lag in (1, 2, 3, 4):
            X[f"{s}_ret_lag{lag}"] = r.shift(lag)
        for w in (5, 10, 20):
            X[f"{s}_mom{w}"] = r.rolling(w).sum()    # momentum
            X[f"{s}_vol{w}"] = r.rolling(w).std()    # volatility
    X["brent_mom5"] = df["brent_ret"].rolling(5).sum()

    y = df[target(stock, h)]
    out = {}
    for part in PARTS:
        m = _sample_mask(df, y, part)
        assert not X[m].isna().any().any()
        out[part] = (X[m], y[m], df.index[m])
    return out


def lstm_data(df: pd.DataFrame, stock: str, h: int, lookback: int = 20,
              features: list = BASE_FEATURES, clip: float | None = 5.0) -> dict:
    """Sequences for LSTM. Each sample holds the last `lookback` days up to and
    including day t. The scaler is fitted on training rows only.

    Outliers are kept (they are real market events), but scaled inputs are
    clipped to +/- `clip` standard deviations so single extreme days do not
    dominate training. Targets are never clipped. Set clip=None to turn off."""
    if lookback > WARMUP + 1:
        raise ValueError(f"lookback must be at most {WARMUP + 1}")

    split = df["split"].to_numpy()
    scaler = StandardScaler().fit(df.loc[split == "train", features])
    F = scaler.transform(df[features]).astype(np.float32)
    if clip is not None:
        F = np.clip(F, -clip, clip)
    y = df[target(stock, h)]

    out = {"scaler": scaler}
    for part in PARTS:
        idx = np.flatnonzero(_sample_mask(df, y, part))
        X = np.stack([F[i - lookback + 1:i + 1] for i in idx])
        out[part] = (X, y.to_numpy(np.float32)[idx], df.index[idx])
    return out


def evaluate(y_true, y_pred) -> dict:
    """Same metrics for every model. Direction is 'up' when the predicted return is above 0."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "direction_acc": float(np.mean((y_pred > 0) == (y_true > 0))),
    }


def baselines(y_train, n: int) -> dict:
    """Naive forecasts every model must beat. 'zero' predicts no change,
    'train_mean' predicts the average training return (in practice: always up)."""
    return {
        "zero": np.zeros(n),
        "train_mean": np.full(n, float(np.mean(y_train))),
    }

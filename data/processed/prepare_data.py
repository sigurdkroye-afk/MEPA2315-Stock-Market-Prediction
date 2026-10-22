"""
prepare_data.py

Builds the shared dataset for the MEPA2315 stock project (Equinor and Aker BP).
Everyone in the group trains on the output of this script, so RF and LSTM
start from exactly the same rows, targets and split.

Input
    data/raw/panel_long.csv   Titlon daily data, both stocks stacked
    data/raw/market.csv       Brent, VIX, S&P 500, USD/NOK (already lagged one trading day)
Output
    data/processed/dataset.csv   one row per trading day: split label, features, targets

Run
    python prepare_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_STOCKS = ROOT / "data" / "raw" / "panel_long.csv"
RAW_MARKET = ROOT / "data" / "raw" / "market.csv"
OUT = ROOT / "data" / "processed" / "dataset.csv"

STOCKS = ["EQNR", "AKRBP"]
HORIZONS = [3, 5, 10]

# Chronological split. Change the dates here only and rerun the script.
TRAIN_END = "2022-12-31"   # train: first day to TRAIN_END
VAL_END = "2024-12-31"     # val:   TRAIN_END to VAL_END, test: after VAL_END

# Trading days left out at the end of train and val, so that an h-day target
# never uses prices from the next period. Equal to the longest horizon.
EMBARGO = max(HORIZONS)


def load_stocks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])

    # Titlon includes some days when Oslo Børs was closed (Christmas Eve, May 17,
    # Ascension Day, ...) with Close = 0 and Volume = 0 but a filled-in price.
    # These rows create fake returns and are removed.
    closed = (df["Close"] <= 0) | (df["Volume"] <= 0)
    df = df.loc[~closed]

    parts = []
    for s in STOCKS:
        d = df[df["Ticker"] == s].set_index("Date").sort_index()
        parts.append(pd.DataFrame({
            f"{s}_close": d["Close"],               # actual traded price, for price level forecasts
            f"{s}_adjclose": d["AdjustedPrice"],    # adjusted for dividends and splits, used for returns
            f"{s}_logvol": np.log(d["Volume"]),
            f"{s}_range": np.log(d["High"] / d["Low"]),   # intraday range
        }))

    # Market wide Titlon columns are identical for both stocks, so take them once.
    d = df[df["Ticker"] == STOCKS[0]].set_index("Date").sort_index()
    parts.append(pd.DataFrame({
        "osebx_ret": d["lnDeltaOSEBX"],
        "obx_ret": d["lnDeltaOBX"],
        "nowa": d["NOWA_DayLnrate"],
        "smb": d["SMB"],
        "liq": d["LIQ"],
        "mom": d["MOM"],
    }))

    # Inner join keeps only days when both stocks traded.
    return pd.concat(parts, axis=1, join="inner")


def load_market(path: Path) -> pd.DataFrame:
    # Values on row t are the previous trading day's closes, so they were known
    # when Oslo Børs closed on day t. Keep levels and recompute changes below.
    m = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return m[["brent_usd", "vix", "sp500", "usdnok"]]


def build() -> pd.DataFrame:
    df = load_stocks(RAW_STOCKS).join(load_market(RAW_MARKET), how="left")

    # Returns are recomputed after closed days are removed, so the move across
    # a holiday is kept in full on the next trading day.
    for s in STOCKS:
        df[f"{s}_ret"] = np.log(df[f"{s}_adjclose"]).diff()
    df["brent_ret"] = np.log(df["brent_usd"]).diff()
    df["sp500_ret"] = np.log(df["sp500"]).diff()
    df["usdnok_ret"] = np.log(df["usdnok"]).diff()
    df["vix_chg"] = df["vix"].diff()

    # Volume relative to its own last 20 days. Raw volume drifts over time
    # (Aker BP has about three times as many shares in 2026 as in 2016).
    for s in STOCKS:
        df[f"{s}_relvol"] = df[f"{s}_logvol"] - df[f"{s}_logvol"].rolling(20).mean()

    # Targets: log return from close t to close t+h (direct forecast).
    for s in STOCKS:
        logp = np.log(df[f"{s}_adjclose"])
        for h in HORIZONS:
            df[f"y_{s}_h{h}"] = logp.shift(-h) - logp

    # Drop rows where a feature is missing (the first 20 trading days).
    feature_cols = [c for c in df.columns if not c.startswith("y_")]
    df = df.dropna(subset=feature_cols)

    # Split label with embargo.
    dates = df.index
    split = np.where(dates <= TRAIN_END, "train",
                     np.where(dates <= VAL_END, "val", "test")).astype(object)
    for end in (TRAIN_END, VAL_END):
        first_after = dates.searchsorted(pd.Timestamp(end), side="right")
        split[first_after - EMBARGO:first_after] = "gap"
    df.insert(0, "split", split)

    order = (["split"]
             + [f"{s}_{c}" for s in STOCKS for c in ("close", "adjclose", "ret", "logvol", "relvol", "range")]
             + ["osebx_ret", "obx_ret", "brent_usd", "brent_ret", "sp500", "sp500_ret",
                "usdnok", "usdnok_ret", "vix", "vix_chg", "nowa", "smb", "liq", "mom"]
             + [f"y_{s}_h{h}" for s in STOCKS for h in HORIZONS])
    return df[order]


def check(df: pd.DataFrame) -> None:
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert (df[[f"{s}_close" for s in STOCKS]] > 0).all().all(), "closed days left"
    feats = [c for c in df.columns if c != "split" and not c.startswith("y_")]
    assert not df[feats].isna().any().any(), "missing feature values"
    # No training target may reach into val, and no val target into test.
    pos = np.arange(len(df))
    for part, end in (("train", TRAIN_END), ("val", VAL_END)):
        last = pos[df["split"].to_numpy() == part].max()
        assert df.index[last + EMBARGO] <= pd.Timestamp(end), f"{part} target leaks"


if __name__ == "__main__":
    data = build()
    check(data)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index_label="Date", float_format="%.10g")

    print(f"Saved {OUT.relative_to(ROOT)}: {len(data)} trading days, {data.shape[1]} columns")
    for part in ("train", "val", "test"):
        d = data.index[data["split"] == part]
        print(f"  {part:5s} {len(d):5d} rows  {d.min().date()} to {d.max().date()}")
    print(f"  gap   {(data['split'] == 'gap').sum():5d} rows  ({EMBARGO} before val and {EMBARGO} before test)")

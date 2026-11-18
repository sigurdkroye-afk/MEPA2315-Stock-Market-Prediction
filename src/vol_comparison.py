"""
vol_comparison.py

Builds reports/lstm_vol/comparison.csv from files that already exist. No model is
trained: the LSTM forecasts are read from reports/predictions/lstmvol_<STOCK>_h<h>.csv
(val and test rows), and the references are computed with the same functions as
run_lstm_vol.py (zero, level, vol20, garch). Same DM test and QLIKE definition.

    python src/vol_comparison.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_lstm_vol import (BASES, HORIZONS, OUT, PRED, STOCKS, dm_test, garch_logvol,
                          level_ref, load, log_rv, qlike, vol20, vol_data, zero_level)

COLUMNS = ["part", "stock", "h", "benchmark", "mse_lstm", "mse_ref", "mse_ratio", "dm_p_mse",
           "qlike_lstm", "qlike_ref", "qlike_ratio", "dm_p_qlike"]


def qlike_loss(z, f):
    x = np.exp(2 * (z - f))
    return x - np.log(x) - 1


def main():
    cfg = json.loads((OUT / "config.json").read_text())
    df = load()
    rows = []
    for stock in STOCKS:
        garch, _ = garch_logvol(df, stock)
        ret = df[f"{stock}_ret"]
        for h in HORIZONS:
            d = vol_data(df, stock, h, cfg["lookback"])     # only for sample dates and train mean
            pred = pd.read_csv(PRED / f"lstmvol_{stock}_h{h}.csv", parse_dates=["Date"]).set_index("Date")
            refs = {"zero": None, "level": level_ref(ret, h), "vol20": vol20(ret, h), "garch": garch[h]}
            for part in ("val", "test"):
                dates = pd.DatetimeIndex(d[part][2])
                z = log_rv(ret, h).loc[dates].to_numpy()
                f = pred["logvol_pred"].reindex(dates).to_numpy()
                assert np.isfinite(f).all(), f"{stock} h={h} {part}: missing LSTM predictions"
                for base in BASES:
                    r = (np.full(len(z), zero_level(d)) if base == "zero"
                         else refs[base].loc[dates].to_numpy())
                    assert np.isfinite(r).all(), f"{stock} h={h} {part}: {base} has NaN"
                    mse_l, mse_r = np.mean((f - z) ** 2), np.mean((r - z) ** 2)
                    q_l, q_r = qlike(z, f), qlike(z, r)
                    rows.append({"part": part, "stock": stock, "h": h, "benchmark": base,
                                 "mse_lstm": mse_l, "mse_ref": mse_r, "mse_ratio": mse_l / mse_r,
                                 "dm_p_mse": dm_test((f - z) ** 2, (r - z) ** 2, h),
                                 "qlike_lstm": q_l, "qlike_ref": q_r, "qlike_ratio": q_l / q_r,
                                 "dm_p_qlike": dm_test(qlike_loss(z, f), qlike_loss(z, r), h)})
    out = pd.DataFrame(rows, columns=COLUMNS)
    out.to_csv(OUT / "comparison.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

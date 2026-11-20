"""
lstm_vol_interval.py  (V4: interval scaled by the volatility LSTM)

Protocol, fixed before any test number was computed:
  * Volatility forecast: the frozen volatility LSTM (lookback 20, 16 hidden units),
    read from reports/predictions/lstmvol_<STOCK>_h<h>.csv (sigma_pred). No retraining.
  * Interval: [q_0.1 * sigma, q_0.9 * sigma], where q are the 10th and 90th percentiles of
    y / sigma on the calibration period (same construction as `skalert_intervall` in
    intervals.py). LSTM forecasts exist only from the validation period on, so the
    calibration period is the validation set, not training.
  * Fair comparison: the same validation-calibrated construction with 20-day volatility
    and GARCH (intervals.py), so only the volatility forecast differs.
  * Step 1, validation only: calibrate on 2023 (minus the last 10 trading days, as a gap),
    evaluate on 2024.
  * Step 2, test, run once: calibrate on 2023-2024, evaluate on 2025-2026.
  * Metrics as in intervals.py: coverage, width, interval score, coverage on calm and
    turbulent days (split at the median 20-day volatility of the evaluation period).

Run from the repository root:  python src/lstm_vol_interval.py val   (then: test)
Output: reports/intervals/v4_val.csv, v4_test.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import intervals as iv

df = iv.load()
OUT = ROOT / "reports" / "intervals"; OUT.mkdir(parents=True, exist_ok=True)

def lstm_sigma(s, h):
    p = pd.read_csv(ROOT / "reports" / "predictions" / f"lstmvol_{s}_h{h}.csv",
                    parse_dates=["Date"], index_col="Date")["sigma_pred"]
    return p.reindex(df.index)

def evaluate(cal, ev, tag):
    rows = []
    for s in iv.STOCKS:
        gsig, _ = iv.garch_sigma(df, s)
        for h in iv.HORIZONS:
            y = df[f"y_{s}_h{h}"]
            sig = {"vol20": iv.vol20(df, s, h), "garch": gsig[h], "lstm": lstm_sigma(s, h)}
            c = cal(df) & y.notna(); e = ev(df) & y.notna()
            assert all(sig[k][c | e].notna().all() for k in sig), (s, h)
            v20 = iv.vol20(df, s, h); turb = v20 > v20[e].median()
            for name, sg in sig.items():
                lo, hi = iv.skalert_intervall(y, sg, c)
                r = {"part": tag, "stock": s, "h": h, "sigma": name, **iv.evaluer(y[e], lo[e], hi[e]),
                     "score": iv.interval_score(y[e], lo[e], hi[e])}
                r["calm %"] = iv.evaluer(y[e & ~turb], lo[e & ~turb], hi[e & ~turb])["dekning %"]
                r["turbulent %"] = iv.evaluer(y[e & turb], lo[e & turb], hi[e & turb])["dekning %"]
                r["below %"] = round(100 * float((y[e] < lo[e]).mean()), 1)
                r["above %"] = round(100 * float((y[e] > hi[e]).mean()), 1)
                rows.append(r)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "val"   # check | val | test
    if step == "check":   # reproduce the existing train-calibrated GARCH test row
        out = []
        for s in iv.STOCKS:
            gsig, _ = iv.garch_sigma(df, s)
            for h in iv.HORIZONS:
                y = df[f"y_{s}_h{h}"]; tr = (df.split == "train") & y.notna() & gsig[h].notna()
                e = (df.split == "test") & y.notna()
                lo, hi = iv.skalert_intervall(y, gsig[h], tr)
                out.append((s, h, iv.evaluer(y[e], lo[e], hi[e]), iv.interval_score(y[e], lo[e], hi[e])))
        for o in out: print(o)
    elif step == "val":
        yr = lambda d: d.index.year
        first24 = df.index[(df.split == "val") & (df.index.year == 2024)][0]
        cut = df.index[df.index.get_loc(first24) - 10]   # 10-day gap, as between the periods
        r = evaluate(lambda d: (d.split == "val") & (d.index < cut),
                     lambda d: (d.split == "val") & (yr(d) == 2024), "val: cal 2023, eval 2024")
        r.to_csv(OUT / "v4_val.csv", index=False); print(r.to_string(index=False))
        print(r.groupby("sigma")[["dekning %", "bredde pp", "score", "calm %", "turbulent %"]].mean().round(2))
    elif step == "test":
        r = evaluate(lambda d: d.split == "val", lambda d: d.split == "test", "test: cal 2023-24")
        r.to_csv(OUT / "v4_test.csv", index=False); print(r.to_string(index=False))
        print(r.groupby("sigma")[["dekning %", "bredde pp", "score", "calm %", "turbulent %"]].mean().round(2))

"""
run_lstm_vol.py

Post-hoc extension: can the return LSTM's architecture forecast realised volatility
over the next h trading days, and does it beat simple volatility baselines?
The idea came after the return results were known (see src/README_lstm_vol.md).

    python src/run_lstm_vol.py            # grid on val, then the final model on val
    python src/run_lstm_vol.py --final    # final model from reports/lstm_vol/config.json on val
    python src/run_lstm_vol.py --quick    # 2 seeds, one config, writes to reports/lstm_vol/quick/
    python src/run_lstm_vol.py --test     # run once, at the very end: final model on test

Target: z[t, h] = log sqrt(sum_{k=1..h} ret[t+k]^2), only returns after day t.
Samples, split and input scaling come from lstm_data, so they are the same as for
the return models. Baselines: zero (train mean of z), vol20 (20-day std scaled by
sqrt(h)) and garch (GARCH(1,1) with t errors, fitted before the first val date).
"zero" always predicts the train mean of z. LSTMRegressor standardises the target with
the train mean and std, so this is the forecast 0 in the model's own units: the same role
as the zero-return forecast for the return model.

Outputs in reports/lstm_vol/ (--quick: reports/lstm_vol/quick/):
    grid_val.csv          every grid config on every stock/horizon
    config.json           the chosen config (lowest mean ratio_vs_garch)
    metrics_val.json      LSTM and baselines on val (--test: metrics_test.json)
    summary.md            tables built from the files above
    eqnr_h5_val.png       actual and forecast log volatility, EQNR h=5, val
Predictions (Date, logvol_pred, sigma_pred) go to reports/predictions/lstmvol_<STOCK>_h<h>.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import arch  # noqa: E402
from arch import arch_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data" / "processed")]

from datasets import BASE_FEATURES, PARTS, STOCKS, load, lstm_data  # noqa: E402
from lstm_model import LSTMRegressor  # noqa: E402

OUT = ROOT / "reports" / "lstm_vol"
PRED = ROOT / "reports" / "predictions"
HORIZONS = [3, 5, 10]
LOOKBACKS = [20, 40]
HIDDEN_SIZES = [16, 32]
DROPOUT = 0.2
N_SEEDS = 5
MODELS = ["lstm", "zero", "vol20", "garch"]


# ------------------------------------------------------------ target and baselines
def log_rv(ret: pd.Series, h: int) -> pd.Series:
    """z[t] = log sqrt(ret[t+1]^2 + ... + ret[t+h]^2). ret[t] itself is not included."""
    future = sum(ret.shift(-k) ** 2 for k in range(1, h + 1))
    return np.log(np.sqrt(future))


def vol20(ret: pd.Series, h: int) -> pd.Series:
    """log(sqrt(h) * std(ret[t-19..t])), past data only."""
    return np.log(np.sqrt(h) * ret.rolling(20).std())


def garch_logvol(df: pd.DataFrame, stock: str, horizons=HORIZONS):
    """GARCH(1,1) with t errors on 100 * ret, parameters fitted only on rows before the
    first val date (same rule as intervals.py). The forecast made at t uses returns up
    to and including t. Returns ({h: log h-day volatility}, fit result)."""
    first_val = df.index[df["split"] == "val"][0]
    r = 100 * df[f"{stock}_ret"].dropna()
    model = arch_model(r, vol="GARCH", p=1, q=1, dist="t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(last_obs=first_val, disp="off")
        fc = res.forecast(horizon=max(horizons), start=r.index[0], reindex=False)
    var = fc.variance                       # column h.k = variance for day t+k
    out = {h: np.log(np.sqrt(var.iloc[:, :h].sum(axis=1)) / 100).reindex(df.index) for h in horizons}
    return out, res


def vol_data(df: pd.DataFrame, stock: str, h: int, lookback: int) -> dict:
    """X and dates from lstm_data (same rows, split and scaler as the return models),
    with y replaced by the log realised volatility z."""
    d = lstm_data(df, stock, h, lookback=lookback, features=BASE_FEATURES)
    z = log_rv(df[f"{stock}_ret"], h)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    out = {"scaler": d["scaler"]}
    for part in PARTS:
        X, _, dates = d[part]
        zp = z.loc[dates].to_numpy()
        assert np.isfinite(zp).all(), f"{stock} h={h} {part}: target has NaN or inf"
        out[part] = (X, zp, dates)
    # Same embargo as the return targets: the last day a sample's target uses (t + h)
    # comes before the first row of the next part.
    for a, b in (("train", "val"), ("val", "test")):
        first_b = pos[df["split"] == b].iloc[0]
        assert pos.loc[out[a][2]].max() + h < first_b, f"{stock} h={h}: {a} targets reach into {b}"
    return out


def zero_level(d: dict) -> float:
    """The "zero" forecast: train mean of z (= 0 in standardised units). Train samples only."""
    return float(d["train"][1].mean())


# ---------------------------------------------------------------------- metrics
def dm_test(loss_a, loss_b, h: int) -> float:
    """Diebold-Mariano test, H0: equal mean loss. Two-sided p-value. The h-day targets
    overlap, so the variance uses Newey-West weights up to lag h-1."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n, dbar = len(d), d.mean()
    dc = d - dbar
    lrv = dc @ dc / n
    for k in range(1, h):
        lrv += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return float("nan")
    stat = dbar / math.sqrt(lrv / n)
    return float(math.erfc(abs(stat) / math.sqrt(2)))


def qlike(z, f) -> float:
    """mean(RV^2/s^2 - log(RV^2/s^2) - 1) with RV = exp(z), s = exp(f). 0 for a perfect forecast."""
    x = np.exp(2 * (np.asarray(z, float) - np.asarray(f, float)))
    return float(np.mean(x - np.log(x) - 1))


def vol_metrics(z, f) -> dict:
    """mse_log = bias_sq + mse_rest: bias_sq is the error from a wrong average level
    ((mean f - mean z)^2), mse_rest the error from not following the variation around it."""
    z, f = np.asarray(z, float), np.asarray(f, float)
    corr = float(np.corrcoef(z, f)[0, 1]) if f.std() > 0 else float("nan")   # zero has no corr
    mse = float(np.mean((f - z) ** 2))
    bias_sq = float((f.mean() - z.mean()) ** 2)
    return {"mse_log": mse, "qlike": qlike(z, f), "corr": corr,
            "mean_forecast": float(f.mean()), "bias_sq": bias_sq, "mse_rest": mse - bias_sq}


# -------------------------------------------------------------------- one model
def run_pair(df, stock, h, cfg, n_seeds, garch, parts=("val",)):
    """Trains the LSTM once and scores it and the baselines on every part in `parts`.
    Returns {part: (metrics, forecasts DataFrame)}."""
    d = vol_data(df, stock, h, cfg["lookback"])
    X_tr, z_tr, _ = d["train"]
    model = LSTMRegressor(hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
                          n_seeds=n_seeds).fit(X_tr, z_tr)
    v20 = vol20(df[f"{stock}_ret"], h)
    out = {}
    for part in parts:
        X, z, dates = d[part]
        members = model.predict_members(X).astype(float)      # (n_seeds, n)
        f = {"lstm": members.mean(axis=0),                     # same as model.predict(X)
             "zero": np.full(len(z), zero_level(d)),
             "vol20": v20.loc[dates].to_numpy(),
             "garch": garch[h].loc[dates].to_numpy()}
        for name, v in f.items():
            assert np.isfinite(v).all(), f"{stock} h={h} {part}: {name} forecast has NaN"
        m = {name: vol_metrics(z, v) for name, v in f.items()}
        loss = {name: (v - z) ** 2 for name, v in f.items()}
        for base in ("zero", "vol20", "garch"):
            m["lstm"][f"ratio_vs_{base}"] = m["lstm"]["mse_log"] / m[base]["mse_log"]
            m["lstm"][f"dm_p_vs_{base}"] = dm_test(loss["lstm"], loss[base], h)
        m["lstm"]["qlike_ratio_vs_zero"] = m["lstm"]["qlike"] / m["zero"]["qlike"]
        seed_ratio = [np.mean((p - z) ** 2) / m["zero"]["mse_log"] for p in members]
        m["lstm"]["seed_ratio_vs_zero_mean"] = float(np.mean(seed_ratio))
        m["lstm"]["seed_ratio_vs_zero_std"] = float(np.std(seed_ratio))
        m["lstm"]["mean_best_epoch"] = float(np.mean(model.best_epochs_))
        m["mean_z"] = float(np.mean(z))
        out[part] = (m, pd.DataFrame({"Date": dates, "z": z, **f}))
    return out


def clean(o):
    """numpy types to Python types and nan/inf to None, so the JSON is valid."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ------------------------------------------------------------------------ grid
def grid_step(df, grid, n_seeds, garch, out_dir):
    rows, t0 = [], time.time()
    for cfg in grid:
        for stock in STOCKS:
            for h in HORIZONS:
                m, _ = run_pair(df, stock, h, cfg, n_seeds, garch[stock])["val"]
                rows.append({"stock": stock, "h": h, **cfg, **m["lstm"]})
                print(f"[{time.time() - t0:6.0f}s] {cfg} {stock} h={h}: "
                      f"ratio_vs_garch={m['lstm']['ratio_vs_garch']:.4f}", flush=True)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(out_dir / "grid_val.csv", index=False)
    best = rank_grid(grid_df)[1]
    (out_dir / "config.json").write_text(json.dumps(best, indent=2))
    print("chosen config:", best)
    return best


def rank_grid(grid_df):
    """One config for all six stock/horizon pairs: lowest mean ratio_vs_garch."""
    keys = ["lookback", "hidden_size", "dropout"]
    ranking = grid_df.groupby(keys)["ratio_vs_garch"].mean().sort_values()
    best = dict(zip(keys, ranking.index[0]))
    best = {k: (float(v) if k == "dropout" else int(v)) for k, v in best.items()}
    return ranking, best


# ----------------------------------------------------------------- final model
def final_step(df, cfg, n_seeds, garch, garch_params, out_dir, pred_dir, parts=("val",)):
    """parts=("val",): metrics_val.json, prediction files with val rows, figure.
    parts=("val", "test"): metrics_test.json, prediction files with val + test rows."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    results = {p: {} for p in parts}
    t0 = time.time()
    for stock in STOCKS:
        for h in HORIZONS:
            res = run_pair(df, stock, h, cfg, n_seeds, garch[stock], parts=parts)
            for part, (m, f) in res.items():
                results[part].setdefault(stock, {})[f"h{h}"] = m
                print(f"[{time.time() - t0:6.0f}s] final {part} {stock} h={h}: "
                      f"ratio_vs_garch={m['lstm']['ratio_vs_garch']:.4f}", flush=True)
            f = pd.concat([f for _, f in res.values()])
            pd.DataFrame({"Date": f["Date"], "logvol_pred": f["lstm"],
                          "sigma_pred": np.exp(f["lstm"])}).to_csv(
                pred_dir / f"lstmvol_{stock}_h{h}.csv", index=False)
            if stock == "EQNR" and h == 5 and parts == ("val",):
                plot_eqnr_h5(res["val"][1], out_dir / "eqnr_h5_val.png")

    meta = {"config": cfg, "features": "BASE_FEATURES", "n_seeds": n_seeds,
            "torch_version": torch.__version__, "arch_version": arch.__version__,
            "garch_params_fitted_before_val": garch_params}
    part = "test" if "test" in parts else "val"
    doc = {**meta, "split": part, "results": results[part]}
    (out_dir / f"metrics_{part}.json").write_text(json.dumps(clean(doc), indent=2))


def plot_eqnr_h5(f: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(f["Date"], f["z"], color="lightgray", linewidth=1, label="actual z")
    ax.plot(f["Date"], f["garch"], color="tab:orange", label="garch")
    ax.plot(f["Date"], f["vol20"], color="tab:green", label="vol20")
    ax.plot(f["Date"], f["lstm"], color="tab:blue", label="lstm")
    ax.set_title("EQNR, h=5, validation")
    ax.set_ylabel("log volatility over 5 days")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(out_dir: Path) -> None:
    grid_df = pd.read_csv(out_dir / "grid_val.csv")
    ranking, _ = rank_grid(grid_df)
    doc = json.loads((out_dir / "metrics_val.json").read_text())
    rows, zero_rows, lstm_rows, dec_rows = [], [], [], []
    for stock, hs in doc["results"].items():
        for hk, m in hs.items():
            zm, lm = m["zero"], m["lstm"]
            dec_rows.append({"stock": stock, "h": int(hk[1:]), "mean_z": m["mean_z"],
                             "mean_zero": zm["mean_forecast"], "bias_sq_zero": zm["bias_sq"],
                             "mse_zero": zm["mse_log"], "mean_lstm": lm["mean_forecast"],
                             "bias_sq_lstm": lm["bias_sq"], "mse_lstm": lm["mse_log"],
                             # share of the LSTM's MSE gain over zero that comes from the level
                             "level_share_of_gain": (zm["bias_sq"] - lm["bias_sq"])
                                                    / (zm["mse_log"] - lm["mse_log"])})
            for name in MODELS:
                rows.append({"stock": stock, "h": int(hk[1:]), "model": name,
                             **{k: m[name][k] for k in ("mse_log", "qlike", "corr")}})
            lm = m["lstm"]
            zero_rows.append({"stock": stock, "h": int(hk[1:]),
                              "mse_lstm": lm["mse_log"], "mse_zero": m["zero"]["mse_log"],
                              "ratio_vs_zero": lm["ratio_vs_zero"], "dm_p_vs_zero": lm["dm_p_vs_zero"],
                              "qlike_ratio_vs_zero": lm["qlike_ratio_vs_zero"], "corr": lm["corr"],
                              "seed_ratio_mean": lm["seed_ratio_vs_zero_mean"],
                              "seed_ratio_std": lm["seed_ratio_vs_zero_std"]})
            lstm_rows.append({"stock": stock, "h": int(hk[1:]),
                              **{k: m["lstm"][k] for k in ("ratio_vs_vol20", "dm_p_vs_vol20",
                                                           "ratio_vs_garch", "dm_p_vs_garch")}})
    lines = [
        "# LSTM volatility, validation results (2023-2024)",
        "",
        f"Seeds per model: {doc['n_seeds']}. Test set not used.",
        "",
        "## Grid (mean ratio_vs_garch over the six stock/horizon pairs)",
        "",
        ranking.round(4).to_frame("mean ratio_vs_garch").to_markdown(),
        "",
        f"Chosen config: `{json.dumps(doc['config'])}`",
        "",
        "## Final model and baselines",
        "",
        pd.DataFrame(rows).round(4).to_markdown(index=False),
        "",
        "## LSTM vs zero forecast (val)",
        "",
        pd.DataFrame(zero_rows).round(4).to_markdown(index=False),
        "",
        "## Error decomposition, LSTM vs zero (val)",
        "",
        "mse_log = bias_sq + mse_rest, bias_sq = (mean forecast - mean z)^2. "
        "level_share_of_gain = (bias_sq_zero - bias_sq_lstm) / (mse_zero - mse_lstm).",
        "",
        pd.DataFrame(dec_rows).round(4).to_markdown(index=False),
        "",
        "## LSTM against vol20 and garch (mse_log ratio and Diebold-Mariano p-value)",
        "",
        pd.DataFrame(lstm_rows).round(4).to_markdown(index=False),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds, one config; all outputs go to reports/lstm_vol/quick/")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true", help="final model from config.json on val")
    mode.add_argument("--test", action="store_true", help="final model on test, run once at the end")
    args = ap.parse_args()

    out_dir = OUT / "quick" if args.quick else OUT
    pred_dir = out_dir / "predictions" if args.quick else PRED
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = 2 if args.quick else N_SEEDS

    df = load()
    garch, params = {}, {}
    for stock in STOCKS:
        garch[stock], res = garch_logvol(df, stock)
        params[stock] = res.params.to_dict()

    if args.final or args.test:
        cfg = json.loads((out_dir / "config.json").read_text())
    else:
        grid = ([{"lookback": 20, "hidden_size": 16, "dropout": DROPOUT}] if args.quick else
                [{"lookback": lb, "hidden_size": hs, "dropout": DROPOUT}
                 for lb in LOOKBACKS for hs in HIDDEN_SIZES])
        cfg = grid_step(df, grid, n_seeds, garch, out_dir)

    parts = ("val", "test") if args.test else ("val",)
    final_step(df, cfg, n_seeds, garch, params, out_dir, pred_dir, parts)
    if not args.test:
        write_summary(out_dir)
        print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()

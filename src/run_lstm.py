"""
run_lstm.py

Training and validation experiments for the LSTM (Assignment 2 / Task 3).
Uses only the train and val parts of the shared dataset. The test part is
touched only with --test, which should be run once, at the very end, after the
group has frozen both the RF and the LSTM configuration.

    python src/run_lstm.py              # grid + input ablation on val (about 30 min on a laptop CPU)
    python src/run_lstm.py --quick      # 2 seeds, smaller grid, for checking that everything runs
    python src/run_lstm.py --final      # final model from reports/lstm/config.json on val (no grid, no ablation)
    python src/run_lstm.py --test       # final run on test with the frozen config in reports/lstm/config.json

Outputs in reports/lstm/ (--quick writes all of them to reports/lstm/quick/ instead):
    grid_val.csv          every grid config on every stock/horizon
    ablation_val.csv      input groups added one at a time, chosen config
    val_predictions.csv   val predictions of the chosen config (for RF comparison and conformal intervals)
    config.json           the chosen config
    metrics.json          final model on val (--test: test_metrics.json)
    summary.md            tables for the report
Predictions per stock/horizon (Date,pred) go to reports/predictions/lstm_<STOCK>_h<h>.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data" / "processed")]

from datasets import BASE_FEATURES, STOCKS, evaluate, lstm_data, load  # noqa: E402
from lstm_model import LSTMRegressor  # noqa: E402

OUT = ROOT / "reports" / "lstm"
HORIZONS = [3, 5, 10]
LOOKBACKS = [10, 20, 40]
HIDDEN_SIZES = [16, 32]
# The final model = config.json + these inputs in exactly this order. Used by both the
# val run (summary, val_predictions.csv) and --test, so the two cannot diverge.
FINAL_FEATURES = list(BASE_FEATURES)
# LSTMRegressor settings taken from the command line (hidden_size and dropout come from the config).
TRAIN_KEYS = ["num_layers", "lr", "weight_decay", "batch_size", "max_epochs", "patience",
              "es_fraction", "embargo", "n_seeds", "seed"]


# ---------------------------------------------------------------- statistics
def dm_test(y, pred_a, pred_b, h: int) -> float:
    """Diebold-Mariano test, H0: model a and model b have equal mean squared error.
    Two-sided p-value. h-day targets overlap, so forecast errors are autocorrelated
    up to lag h-1; the variance uses Newey-West weights up to that lag."""
    d = (pred_a - y) ** 2 - (pred_b - y) ** 2
    n, dbar = len(d), d.mean()
    dc = d - dbar
    lrv = dc @ dc / n
    for k in range(1, h):
        lrv += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return float("nan")
    stat = dbar / math.sqrt(lrv / n)
    return float(math.erfc(abs(stat) / math.sqrt(2)))


def score(y, pred, members, h, model, y_train) -> dict:
    zero = np.zeros_like(y)
    m = evaluate(y, pred)
    rmse_zero = evaluate(y, zero)["rmse"]
    # The net starts out predicting the training mean (the target is standardised),
    # so this baseline shows whether it learned anything beyond the in-sample drift.
    rmse_train_mean = evaluate(y, np.full_like(y, y_train.mean()))["rmse"]
    member_rmse = [evaluate(y, p)["rmse"] for p in members]
    return {
        **m,
        "rmse_zero": rmse_zero,
        "rmse_ratio": m["rmse"] / rmse_zero,           # < 1 means better than "no change"
        "rmse_train_mean": rmse_train_mean,
        "dm_p_vs_zero": dm_test(y, pred, zero, h),
        "majority_dir": float(max((y > 0).mean(), (y <= 0).mean())),   # best constant direction guess
        "member_rmse_mean": float(np.mean(member_rmse)),
        "member_rmse_std": float(np.std(member_rmse)),  # spread between random seeds
        "pred_std": float(np.std(pred)),               # near 0: the net predicts almost a constant
        "pred_up_share": float((pred > 0).mean()),
        "mean_best_epoch": float(np.mean(model.best_epochs_)),
    }


# ------------------------------------------------------------- input groups
def feature_groups(stock: str) -> dict:
    other = [s for s in STOCKS if s != stock][0]
    own = [f"{stock}_ret", f"{stock}_relvol", f"{stock}_range"]
    oth = [f"{other}_ret", f"{other}_relvol", f"{other}_range"]
    oslo = ["osebx_ret", "obx_ret", "smb", "liq", "mom"]
    glob = ["brent_ret", "sp500_ret", "usdnok_ret", "vix", "vix_chg"]
    groups = {
        "own stock": own,
        "+ other stock": own + oth,
        "+ Oslo Børs": own + oth + oslo,
        "+ global (all 16)": own + oth + oslo + glob,
    }
    assert sorted(groups["+ global (all 16)"]) == sorted(BASE_FEATURES)
    return groups


def train_settings(args) -> dict:
    """LSTMRegressor settings from the command line."""
    s = {k: getattr(args, k) for k in TRAIN_KEYS}
    s["refit_full"] = not args.no_refit_full
    return s


def run_one(df, stock, h, cfg, features, args, parts=("val",), keep_pred=False):
    """Trains one model and scores it on every part in `parts`. Returns {part: (row, preds)}."""
    d = lstm_data(df, stock, h, lookback=cfg["lookback"], features=features)
    X_tr, y_tr, _ = d["train"]
    model = LSTMRegressor(hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
                          **train_settings(args)).fit(X_tr, y_tr)
    out = {}
    for part in parts:
        X_ev, y_ev, dates = d[part]
        members = model.predict_members(X_ev)
        pred = members.mean(axis=0)
        row = {"stock": stock, "h": h, **cfg, "n_features": len(features),
               **score(y_ev, pred, members, h, model, y_tr)}
        preds = pd.DataFrame({"Date": dates, "stock": stock, "h": h,
                              "y_true": y_ev, "y_pred": pred}) if keep_pred else None
        out[part] = (row, preds)
    return out


def rank_grid(grid_df):
    """One config is chosen for all six stock/horizon pairs (lowest mean rmse_ratio)."""
    keys = ["lookback", "hidden_size", "dropout"]
    ranking = grid_df.groupby(keys)["rmse_ratio"].mean().sort_values()
    best = dict(zip(keys, ranking.index[0]))
    best = {k: (int(v) if k != "dropout" else float(v)) for k, v in best.items()}
    return ranking, best


# -------------------------------------------------------------- final model
def clean(o):
    """Converts numpy types to Python types and nan/inf to None, so the JSON is valid."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def direction_scores(y_up, pred_up) -> dict:
    """Confusion matrix [[TN, FP], [FN, TP]]: rows = actual [down, up], columns = predicted [down, up]."""
    y_up, pred_up = np.asarray(y_up, bool), np.asarray(pred_up, bool)
    cm = [[int((~y_up & ~pred_up).sum()), int((~y_up & pred_up).sum())],
          [int((y_up & ~pred_up).sum()), int((y_up & pred_up).sum())]]
    return {"direction_acc": float((y_up == pred_up).mean()),
            "direction_balanced_acc": float(balanced_accuracy_score(y_up, pred_up)),
            "confusion_matrix": cm}


def momentum_scores(df, stock, h, dates, y_true) -> dict:
    """Baseline: predicted direction = sign(log P[t] - log P[t-h]), only past adjclose.
    A change of exactly 0 counts as down (up only if > 0, same rule as for the model)."""
    p = np.log(df[f"{stock}_adjclose"])
    mom = (p - p.shift(h)).loc[pd.DatetimeIndex(dates)].to_numpy()
    return direction_scores(np.asarray(y_true) > 0, mom > 0)


def final_step(df, cfg, args, out_dir, parts=("val",)):
    """Final model = cfg + FINAL_FEATURES + command-line training settings, trained once per stock/horizon.
    parts=("val",): metrics.json, val_predictions.csv, predictions/lstm_*.csv with val rows.
    parts=("val", "test"): test_metrics.json, predictions/lstm_*.csv with val + test rows.
    Returns {part: DataFrame with one row per stock/horizon}."""
    pred_dir = ROOT / "reports" / "predictions" if out_dir == OUT else out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = {p: [] for p in parts}
    preds = {p: [] for p in parts}
    results = {p: {} for p in parts}
    t0 = time.time()
    for stock in STOCKS:
        for h in HORIZONS:
            res = run_one(df, stock, h, cfg, FINAL_FEATURES, args, parts=parts, keep_pred=True)
            for part, (r, p) in res.items():
                rows[part].append(r)
                preds[part].append(p)
                m = {k: r[k] for k in ["rmse", "mae", "direction_acc"]}
                d = direction_scores(p["y_true"] > 0, p["y_pred"] > 0)
                m["direction_balanced_acc"] = d["direction_balanced_acc"]
                m["confusion_matrix"] = d["confusion_matrix"]
                m.update({k: r[k] for k in ["rmse_zero", "rmse_train_mean", "rmse_ratio",
                                            "dm_p_vs_zero", "majority_dir"]})
                m["momentum_dir"] = momentum_scores(df, stock, h, p["Date"], p["y_true"])
                results[part].setdefault(stock, {})[f"h{h}"] = m
                print(f"[{time.time() - t0:6.0f}s] final {part} {stock} h={h}: "
                      f"rmse_ratio={r['rmse_ratio']:.4f} dir={r['direction_acc']:.3f}", flush=True)
            both = pd.concat([p for _, p in res.values()])
            pd.DataFrame({"Date": both["Date"], "pred": both["y_pred"]}).to_csv(
                pred_dir / f"lstm_{stock}_h{h}.csv", index=False)

    meta = {"config": cfg, "training_settings": train_settings(args), "n_seeds": args.n_seeds,
            "torch_version": torch.__version__}
    if "test" in parts:
        doc = {**meta, "split": "test", "results": results["test"], "val": results["val"]}
        (out_dir / "test_metrics.json").write_text(json.dumps(clean(doc), indent=2))
    else:
        doc = {**meta, "split": "val", "results": results["val"]}
        (out_dir / "metrics.json").write_text(json.dumps(clean(doc), indent=2))
        pd.concat(preds["val"]).to_csv(out_dir / "val_predictions.csv", index=False)
    return {p: pd.DataFrame(rows[p]) for p in parts}


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds, lookback 20, hidden 32; all outputs go to reports/lstm/quick/")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true",
                      help="final model from config.json on val, no grid and no ablation")
    mode.add_argument("--test", action="store_true")
    ap.add_argument("--lookbacks", type=int, nargs="+", default=LOOKBACKS)
    ap.add_argument("--hidden-sizes", type=int, nargs="+", default=HIDDEN_SIZES)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--es-fraction", type=float, default=0.15)
    ap.add_argument("--embargo", type=int, default=10)
    ap.add_argument("--no-refit-full", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=None, help="default 5 (2 with --quick)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = OUT / "quick" if args.quick else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load()
    if args.n_seeds is None:
        args.n_seeds = 2 if args.quick else 5

    if args.test:
        cfg = json.loads((out_dir / "config.json").read_text())
        final = final_step(df, cfg, args, out_dir, parts=("val", "test"))["test"]
        print(final[["stock", "h", "rmse", "rmse_zero", "rmse_ratio", "dm_p_vs_zero",
                     "direction_acc", "majority_dir"]].round(4).to_string(index=False))
        return

    if args.final:
        cfg = json.loads((out_dir / "config.json").read_text())
        cfg = {k: cfg[k] for k in ("lookback", "hidden_size", "dropout")}
        final_df = final_step(df, cfg, args, out_dir)["val"]
        grid_df = pd.read_csv(out_dir / "grid_val.csv")
        abl_df = pd.read_csv(out_dir / "ablation_val.csv")
        ranking, _ = rank_grid(grid_df)
        write_summary(grid_df, ranking, cfg, abl_df, final_df, args.n_seeds, out_dir)
        return

    # 1) Small grid on val, all 16 inputs. One config is chosen for all six
    #    stock/horizon pairs (lowest mean rmse_ratio), so val is not fitted six times over.
    lookbacks = [20] if args.quick and args.lookbacks == LOOKBACKS else args.lookbacks
    hidden_sizes = [32] if args.quick and args.hidden_sizes == HIDDEN_SIZES else args.hidden_sizes
    grid = [{"lookback": lb, "hidden_size": hs, "dropout": args.dropout}
            for lb in lookbacks
            for hs in hidden_sizes]
    rows = []
    t0 = time.time()
    for cfg in grid:
        for stock in STOCKS:
            for h in HORIZONS:
                r, _ = run_one(df, stock, h, cfg, BASE_FEATURES, args)["val"]
                rows.append(r)
                print(f"[{time.time() - t0:6.0f}s] {cfg} {stock} h={h}: "
                      f"rmse_ratio={r['rmse_ratio']:.4f} dir={r['direction_acc']:.3f}", flush=True)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(out_dir / "grid_val.csv", index=False)

    ranking, best = rank_grid(grid_df)
    (out_dir / "config.json").write_text(json.dumps(best, indent=2))
    print("chosen config:", best)

    # 2) Input ablation with the chosen config: add input groups one at a time.
    abl_rows = []
    for stock in STOCKS:
        for h in HORIZONS:
            for name, feats in feature_groups(stock).items():
                r, _ = run_one(df, stock, h, best, feats, args)["val"]
                r["inputs"] = name
                abl_rows.append(r)
                print(f"[{time.time() - t0:6.0f}s] ablation {stock} h={h} {name}: "
                      f"rmse_ratio={r['rmse_ratio']:.4f}", flush=True)
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(out_dir / "ablation_val.csv", index=False)

    # 3) The final model (chosen config, FINAL_FEATURES) on val: the same model that --test runs.
    final_df = final_step(df, best, args, out_dir)["val"]
    write_summary(grid_df, ranking, best, abl_df, final_df, args.n_seeds, out_dir)


def write_summary(grid_df, ranking, best, abl_df, final, n_seeds, out_dir):
    cols = ["stock", "h", "rmse", "rmse_zero", "rmse_train_mean", "rmse_ratio", "dm_p_vs_zero", "mae",
            "direction_acc", "majority_dir", "pred_std", "member_rmse_std", "mean_best_epoch"]
    piv = abl_df.pivot_table(index=["stock", "h"], columns="inputs", values="rmse_ratio")
    piv = piv[list(feature_groups("EQNR").keys())]
    lines = [
        "# LSTM, validation results (2023-2024)",
        "",
        f"Seeds per model: {n_seeds}. Test set not used.",
        "",
        "## Grid (mean RMSE ratio vs. 'no change' over the six stock/horizon pairs, lower is better)",
        "",
        ranking.round(4).to_frame("mean rmse_ratio").to_markdown(),
        "",
        f"Chosen config: `{json.dumps(best)}`",
        "",
        "## Final model (chosen config, all 16 inputs in the column order used by --test)",
        "",
        final[cols].round(4).to_markdown(index=False),
        "",
        "## Input ablation (RMSE ratio vs. 'no change')",
        "",
        piv.round(4).to_markdown(),
        "",
        "Column guide: rmse_ratio < 1 beats predicting zero return. rmse_train_mean is the",
        "RMSE of always predicting the average training return. dm_p_vs_zero is the",
        "Diebold-Mariano p-value for equal MSE (Newey-West, lag h-1). majority_dir is the",
        "direction accuracy of always guessing the more common sign on val. pred_std near 0",
        "means the net predicts almost a constant. member_rmse_std is the spread over seeds.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

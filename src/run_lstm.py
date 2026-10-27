"""
run_lstm.py

Training and validation experiments for the LSTM (Assignment 2 / Task 3).
Uses only the train and val parts of the shared dataset. The test part is
touched only with --test, which should be run once, at the very end, after the
group has frozen both the RF and the LSTM configuration.

    python src/run_lstm.py              # grid + input ablation on val (about 30 min on a laptop CPU)
    python src/run_lstm.py --quick      # 2 seeds, smaller grid, for checking that everything runs
    python src/run_lstm.py --test       # final run on test with the frozen config in reports/lstm/config.json

Outputs in reports/lstm/:
    grid_val.csv          every grid config on every stock/horizon
    ablation_val.csv      input groups added one at a time, chosen config
    val_predictions.csv   val predictions of the chosen config (for RF comparison and conformal intervals)
    config.json           the chosen config
    summary.md            tables for the report
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data" / "processed")]

from datasets import BASE_FEATURES, STOCKS, evaluate, lstm_data, load  # noqa: E402
from lstm_model import LSTMRegressor  # noqa: E402

OUT = ROOT / "reports" / "lstm"
HORIZONS = [3, 5, 10]
# The final model = config.json + these inputs in exactly this order. Used by both the
# val run (summary, val_predictions.csv) and --test, so the two cannot diverge.
FINAL_FEATURES = list(BASE_FEATURES)


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


def run_one(df, stock, h, cfg, features, n_seeds, part="val", keep_pred=False):
    d = lstm_data(df, stock, h, lookback=cfg["lookback"], features=features)
    X_tr, y_tr, _ = d["train"]
    X_ev, y_ev, dates = d[part]
    model = LSTMRegressor(hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
                          n_seeds=n_seeds).fit(X_tr, y_tr)
    members = model.predict_members(X_ev)
    pred = members.mean(axis=0)
    row = {"stock": stock, "h": h, **cfg, "n_features": len(features),
           **score(y_ev, pred, members, h, model, y_tr)}
    preds = pd.DataFrame({"Date": dates, "stock": stock, "h": h,
                          "y_true": y_ev, "y_pred": pred}) if keep_pred else None
    return row, preds


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    n_seeds = 2 if args.quick else 5

    if args.test:
        cfg = json.loads((OUT / "config.json").read_text())
        rows, preds = [], []
        for stock in STOCKS:
            for h in HORIZONS:
                r, p = run_one(df, stock, h, cfg, FINAL_FEATURES, n_seeds, part="test", keep_pred=True)
                rows.append(r); preds.append(p)
        pd.DataFrame(rows).to_csv(OUT / "test_results.csv", index=False)
        pd.concat(preds).to_csv(OUT / "test_predictions.csv", index=False)
        print(pd.DataFrame(rows)[["stock", "h", "rmse", "rmse_zero", "rmse_ratio", "dm_p_vs_zero",
                                  "direction_acc", "majority_dir"]].round(4).to_string(index=False))
        return

    # 1) Small grid on val, all 16 inputs. One config is chosen for all six
    #    stock/horizon pairs (lowest mean rmse_ratio), so val is not fitted six times over.
    grid = [{"lookback": lb, "hidden_size": hs, "dropout": 0.2}
            for lb in ((20,) if args.quick else (10, 20, 40))
            for hs in ((32,) if args.quick else (16, 32))]
    rows = []
    t0 = time.time()
    for cfg in grid:
        for stock in STOCKS:
            for h in HORIZONS:
                r, _ = run_one(df, stock, h, cfg, BASE_FEATURES, n_seeds)
                rows.append(r)
                print(f"[{time.time() - t0:6.0f}s] {cfg} {stock} h={h}: "
                      f"rmse_ratio={r['rmse_ratio']:.4f} dir={r['direction_acc']:.3f}", flush=True)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(OUT / "grid_val.csv", index=False)

    keys = ["lookback", "hidden_size", "dropout"]
    ranking = grid_df.groupby(keys)["rmse_ratio"].mean().sort_values()
    best = dict(zip(keys, ranking.index[0]))
    best = {k: (int(v) if k != "dropout" else float(v)) for k, v in best.items()}
    (OUT / "config.json").write_text(json.dumps(best, indent=2))
    print("chosen config:", best)

    # 2) Input ablation with the chosen config: add input groups one at a time.
    abl_rows = []
    for stock in STOCKS:
        for h in HORIZONS:
            for name, feats in feature_groups(stock).items():
                r, _ = run_one(df, stock, h, best, feats, n_seeds)
                r["inputs"] = name
                abl_rows.append(r)
                print(f"[{time.time() - t0:6.0f}s] ablation {stock} h={h} {name}: "
                      f"rmse_ratio={r['rmse_ratio']:.4f}", flush=True)
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(OUT / "ablation_val.csv", index=False)

    # 3) The final model (chosen config, FINAL_FEATURES) on val: the same model that --test runs.
    final_rows, val_preds = [], []
    for stock in STOCKS:
        for h in HORIZONS:
            r, p = run_one(df, stock, h, best, FINAL_FEATURES, n_seeds, keep_pred=True)
            final_rows.append(r)
            val_preds.append(p)
            print(f"[{time.time() - t0:6.0f}s] final {stock} h={h}: "
                  f"rmse_ratio={r['rmse_ratio']:.4f}", flush=True)
    final_df = pd.DataFrame(final_rows)
    pd.concat(val_preds).to_csv(OUT / "val_predictions.csv", index=False)

    write_summary(grid_df, ranking, best, abl_df, final_df, n_seeds)


def write_summary(grid_df, ranking, best, abl_df, final, n_seeds):
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
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

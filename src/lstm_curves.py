"""
lstm_curves.py

Diagnostic (not tuning) study of the frozen LSTM: learning curves that show why early
stopping ends after few epochs. Uses only the train and val parts of the shared dataset;
the test part is never read.

    python src/lstm_curves.py            # 150 epochs, 3 seeds
    python src/lstm_curves.py --epochs 100 --seeds 2
    python src/lstm_curves.py --quick    # 2 epochs, 1 seed, for checking that everything runs

The training replicates stage 1 of LSTMRegressor.fit exactly (same settings, seeds, data
split, batch order, Adam, gradient clipping), but without early stopping: every net is
trained for the full number of epochs and the MSE (standardised target) is recorded after
each epoch on three parts: fit (early-stopping fit part), es (early-stopping part) and
val. The reference is "predict the training mean" = 0 in standardised units.

curves(df, runs, cfg, epochs, seeds) returns a long DataFrame with columns
stock, h, hidden_size, seed, epoch, part, mse. Per (run, seed): epochs 1..epochs for
part in {fit, es, val}, plus epoch 0 for part in {fit_ref, es_ref, val_ref}.
Rows = len(runs) * seeds * (3 * epochs + 3).

Outputs in reports/lstm/curves/ (--quick writes them to reports/lstm/curves/quick/ instead):
    curves.csv               the long table above
    learning_curves.png      stock x horizon, hidden size from config.json
    capacity_eqnr_h5.png     EQNR h=5, hidden size 8, 32, 128
    curves_summary.md        tables with epochs of minimum loss and val/ref ratios
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data" / "processed")]

from datasets import STOCKS, lstm_data, load  # noqa: E402
from lstm_model import LSTMRegressor  # noqa: E402

OUT = ROOT / "reports" / "lstm" / "curves"
CONFIG = ROOT / "reports" / "lstm" / "config.json"
HORIZONS = [3, 5, 10]
CAPACITY_SIZES = [8, 32, 128]
COLUMNS = ["stock", "h", "hidden_size", "seed", "epoch", "part", "mse"]
PARTS = ["fit", "es", "val"]
COLORS = {"fit": "tab:blue", "es": "tab:orange", "val": "tab:green"}


# ------------------------------------------------------------------ training
def curves(df: pd.DataFrame, runs: list, cfg: dict, epochs: int, seeds: int) -> pd.DataFrame:
    """runs: list of (stock, h, hidden_size). See the module docstring for the layout."""
    rows = []
    for stock, h, hs in runs:
        # Settings equal to the frozen model (default LSTMRegressor with config hidden_size/dropout).
        ref = LSTMRegressor(hidden_size=hs, dropout=cfg["dropout"])
        d = lstm_data(df, stock, h, lookback=cfg["lookback"])
        X, y = d["train"][:2]
        Xv, yv = d["val"][:2]
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        Xv = np.ascontiguousarray(Xv, dtype=np.float32)

        y_mean, y_std = float(y.mean()), float(y.std())
        ys = ((y - y_mean) / y_std).astype(np.float32)
        yvs = ((np.asarray(yv, dtype=np.float64) - y_mean) / y_std).astype(np.float32)

        n_es = int(round(ref.es_fraction * len(X)))
        cut = len(X) - n_es
        if n_es < 1 or cut - ref.embargo < 1:
            raise ValueError("too few samples: early-stopping part and fit part (after embargo) must both be non-empty")
        X_fit, y_fit = X[:cut - ref.embargo], ys[:cut - ref.embargo]
        X_es, y_es = X[cut:], ys[cut:]
        data = {"fit": (X_fit, y_fit), "es": (X_es, y_es), "val": (Xv, yvs)}

        for s in range(seeds):
            key = dict(stock=stock, h=h, hidden_size=hs, seed=s)
            for part, (_, yp) in data.items():
                rows.append({**key, "epoch": 0, "part": f"{part}_ref", "mse": float(np.mean(yp ** 2))})

            net = ref._new_net(X.shape[2], s)
            opt = torch.optim.Adam(net.parameters(), lr=ref.lr, weight_decay=ref.weight_decay)
            loss_fn = nn.MSELoss()
            Xt, yt = torch.from_numpy(X_fit), torch.from_numpy(y_fit)
            gen = torch.Generator().manual_seed(s)
            for epoch in range(1, epochs + 1):
                net.train()
                order = torch.randperm(len(Xt), generator=gen)
                for i in range(0, len(Xt), ref.batch_size):
                    b = order[i:i + ref.batch_size]
                    opt.zero_grad()
                    loss = loss_fn(net(Xt[b]), yt[b])
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                for part, (Xp, yp) in data.items():
                    rows.append({**key, "epoch": epoch, "part": part, "mse": ref._mse(net, Xp, yp)})
    return pd.DataFrame(rows, columns=COLUMNS)


# ------------------------------------------------------------------ outputs
def _plot_run(ax, c: pd.DataFrame, stock: str, h: int, hs: int) -> None:
    sub = c[(c["stock"] == stock) & (c["h"] == h) & (c["hidden_size"] == hs)]
    mean = sub.groupby(["part", "epoch"])["mse"].mean()
    for part in PARTS:
        curve = mean[part]
        ax.plot(curve.index, curve.values, color=COLORS[part], label=part)
        ax.axhline(mean[f"{part}_ref"].iloc[0], color=COLORS[part], linestyle="--", linewidth=1,
                   label=f"{part} ref (train mean)")
    es = mean["es"]
    ax.axvline(es.idxmin(), color="black", linestyle=":", label=f"min es: epoch {es.idxmin()}")
    ax.set_yscale("log")
    ax.set_title(f"{stock} h={h}, hidden={hs}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardised target)")
    ax.legend(fontsize=7)


def _summary_table(c: pd.DataFrame, keys: list) -> pd.DataFrame:
    rows = []
    for stock, h, hs in keys:
        sub = c[(c["stock"] == stock) & (c["h"] == h) & (c["hidden_size"] == hs)]
        mean = sub.groupby(["part", "epoch"])["mse"].mean()
        val, val_ref = mean["val"], mean["val_ref"].iloc[0]
        es_min = int(mean["es"].idxmin())
        rows.append({
            "stock": stock, "h": h, "hidden_size": hs,
            "es_min_epoch": es_min,
            "val_min_epoch": int(val.idxmin()),
            "val_ratio_at_es_min": float(val[es_min] / val_ref),
            "val_ratio_at_last": float(val.iloc[-1] / val_ref),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="2 epochs, 1 seed, output to curves/quick/")
    args = ap.parse_args()
    out = OUT
    if args.quick:
        args.epochs, args.seeds = 2, 1
        out = OUT / "quick"
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(CONFIG.read_text())
    hs = cfg["hidden_size"]
    df = load()

    curve_runs = [(s, h, hs) for s in STOCKS for h in HORIZONS]
    cap_runs = [("EQNR", 5, k) for k in CAPACITY_SIZES]
    runs = curve_runs + [r for r in cap_runs if r not in curve_runs]

    parts, t0 = [], time.time()
    for n, run in enumerate(runs, 1):
        parts.append(curves(df, [run], cfg, args.epochs, args.seeds))
        print(f"[{n}/{len(runs)}] {run} done, {time.time() - t0:.0f} s", flush=True)
    c = pd.concat(parts, ignore_index=True)
    c.to_csv(out / "curves.csv", index=False)

    fig, axes = plt.subplots(len(STOCKS), len(HORIZONS), figsize=(15, 8), squeeze=False)
    for i, stock in enumerate(STOCKS):
        for j, h in enumerate(HORIZONS):
            _plot_run(axes[i, j], c, stock, h, hs)
    fig.suptitle(f"Learning curves, mean over {args.seeds} seeds")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(CAPACITY_SIZES), figsize=(15, 4.5), squeeze=False)
    for j, k in enumerate(CAPACITY_SIZES):
        _plot_run(axes[0, j], c, "EQNR", 5, k)
    fig.suptitle(f"Capacity, EQNR h=5, mean over {args.seeds} seeds")
    fig.tight_layout()
    fig.savefig(out / "capacity_eqnr_h5.png", dpi=150)
    plt.close(fig)

    t1 = _summary_table(c, curve_runs).drop(columns="hidden_size")
    t2 = _summary_table(c, cap_runs)
    text = (f"Epochs: {args.epochs}, seeds: {args.seeds}\n\n"
            f"Hidden size {hs}\n\n{t1.to_markdown(index=False, floatfmt='.3f')}\n\n"
            f"Capacity, EQNR h=5\n\n{t2.to_markdown(index=False, floatfmt='.3f')}\n")
    (out / "curves_summary.md").write_text(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

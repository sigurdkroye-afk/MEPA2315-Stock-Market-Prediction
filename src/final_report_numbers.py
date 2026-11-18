"""
final_report_numbers.py

Numbers and figures for Sections 4.1-4.4 of the final report that were not produced
by the other scripts. Reads only existing data and existing predictions; no model is
tuned and nothing new is chosen on the test set.

  1. Random forest, gradient boosting and lasso rebuilt in scikit-learn with the Orange
     settings (report, Appendix "Settings"), refitted on train + val and run on test.
     The script checks that relative RMSE and direction accuracy match the report
     (Tables "rmse" and "direction"); if they match, the predictions are the Orange ones.
  2. Direction on the test set for all models: accuracy, balanced accuracy, share of
     'up' forecasts and confusion matrices (LSTM predictions read from
     reports/predictions/lstm_<STOCK>_h<h>.csv).
  3. Assignment 1 statistics recomputed on the cleaned data (closed days removed),
     on the same 2677 trading days the models use.
  4. Figures: LSTM learning curves relative to the constant forecast (from
     reports/lstm/curves/curves.csv) and rolling 21-day volatility.

Run from the repository root (needs the lstm-vol branch files):
    python src/final_report_numbers.py
Output: reports/final/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "processed"))
from datasets import STOCKS, load, rf_data  # noqa: E402

OUT = ROOT / "reports" / "final"
FIG = OUT / "figures"
MARKET = ROOT / "data" / "processed" / "data" / "raw" / "market.csv"
HORIZONS = [3, 5, 10]
CASES = [(s, h) for s in STOCKS for h in HORIZONS]
NAME = {"EQNR": "Equinor", "AKRBP": "Aker BP"}
MODELS = ["zero", "constant", "rf", "gb", "lasso", "lstm"]

# Test-set values in the report (Table "rmse" and Table "direction"), used as a check.
REPORT_REL_RMSE = {
    "constant": [0.997, 0.995, 0.991, 0.995, 0.992, 0.982],
    "rf": [0.997, 0.997, 0.996, 1.006, 1.014, 1.038],
    "gb": [0.996, 0.995, 0.991, 0.998, 0.993, 0.995],
    "lasso": [0.997, 0.995, 0.991, 0.995, 0.992, 0.984],
}
REPORT_DIR_ACC = {
    "rf": [53.2, 56.5, 56.0, 54.4, 51.9, 51.3],
    "gb": [55.1, 56.5, 56.2, 55.5, 57.2, 54.1],
    "lasso": [55.1, 56.2, 56.2, 56.0, 57.4, 57.6],
}


def orange_models(seed: int = 0) -> dict:
    """Appendix 'Settings'. Targets are in per cent, as in Orange."""
    return {
        "rf": RandomForestRegressor(n_estimators=300, max_features=4, min_samples_split=1000,
                                    random_state=seed, n_jobs=-1),
        "gb": GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=1,
                                        min_samples_split=50, random_state=seed),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=1.0, max_iter=100_000)),
    }


def test_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, h in CASES:
        d = rf_data(df, s, h)
        X_fit = pd.concat([d["train"][0], d["val"][0]])
        y_fit = 100 * pd.concat([d["train"][1], d["val"][1]])
        X_te, y_te, dates = d["test"]
        y_te = 100 * y_te.to_numpy()
        pred = {"zero": np.zeros(len(y_te)), "constant": np.full(len(y_te), y_fit.mean())}
        for name, m in orange_models().items():
            pred[name] = m.fit(X_fit, y_fit).predict(X_te)
        lstm = pd.read_csv(ROOT / "reports" / "predictions" / f"lstm_{s}_h{h}.csv",
                           parse_dates=["Date"]).set_index("Date")["pred"]
        pred["lstm"] = 100 * lstm.reindex(dates).to_numpy()
        assert not np.isnan(pred["lstm"]).any(), f"LSTM test predictions missing for {s} h={h}"
        for name, p in pred.items():
            rows.append(pd.DataFrame({"Date": dates, "stock": s, "h": h, "model": name,
                                      "y_true": y_te, "y_pred": p}))
    return pd.concat(rows, ignore_index=True)


def direction_scores(g: pd.DataFrame) -> pd.Series:
    up, pu = g.y_true > 0, g.y_pred > 0
    tpr = (pu & up).sum() / up.sum()
    tnr = (~pu & ~up).sum() / (~up).sum()
    return pd.Series({
        "rel_rmse": np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)) / np.sqrt(np.mean(g.y_true ** 2)),
        "acc": 100 * (pu == up).mean(), "balanced_acc": 100 * (tpr + tnr) / 2,
        "share_up": 100 * pu.mean(), "pred_sd_over_actual_sd": g.y_pred.std() / g.y_true.std(),
        "TP": int((pu & up).sum()), "FN": int((~pu & up).sum()),
        "FP": int((pu & ~up).sum()), "TN": int((~pu & ~up).sum()),
    })


def check_against_report(res: pd.DataFrame) -> None:
    idx = pd.MultiIndex.from_tuples(CASES, names=["stock", "h"])
    for name, vals in REPORT_REL_RMSE.items():
        got = res.xs(name, level="model")["rel_rmse"].reindex(idx).round(3).to_numpy()
        assert np.allclose(got, vals, atol=0.0011), f"rel RMSE {name}: {got} vs report {vals}"
    for name, vals in REPORT_DIR_ACC.items():
        got = res.xs(name, level="model")["acc"].reindex(idx).round(1).to_numpy()
        assert np.allclose(got, vals, atol=0.11), f"direction {name}: {got} vs report {vals}"
    print("OK: the scikit-learn models reproduce the report's test-set tables.")


def a1_statistics(df: pd.DataFrame) -> None:
    mk = pd.read_csv(MARKET, parse_dates=["Date"]).set_index("Date").sort_index()
    assert np.allclose(df["brent_usd"], mk.loc[df.index, "brent_usd"])
    same_day = mk[["brent_usd", "sp500", "usdnok"]].shift(-1).loc[df.index]  # undo the one-day lag
    r = pd.DataFrame({
        "Equinor": df["EQNR_ret"], "Aker BP": df["AKRBP_ret"], "OSEBX": df["osebx_ret"],
        "Brent": np.log(same_day["brent_usd"]).diff(), "USD/NOK": np.log(same_day["usdnok"]).diff(),
        "S&P 500": np.log(same_day["sp500"]).diff()})
    desc = pd.DataFrame({c: {"min": x.min(), "max": x.max(), "mean": x.mean(), "std": x.std(),
                             "skew": stats.skew(x), "excess_kurtosis": stats.kurtosis(x),
                             "annual_std_pct": 100 * x.std() * np.sqrt(252)}
                         for c, x in ((c, r[c].dropna()) for c in r)}).T
    desc.to_csv(OUT / "a1_descriptive.csv")
    corr = pd.DataFrame({c: {"brent_same_day": r[c].corr(r["Brent"]),
                             "brent_lagged_one_day": r[c].corr(df["brent_ret"])}
                         for c in ["Equinor", "Aker BP", "OSEBX"]}).T
    corr.to_csv(OUT / "a1_brent_correlation.csv")
    out = {}
    for c in r:
        x = r[c].dropna(); z = (x - x.mean()) / x.std()
        q1, q3 = x.quantile([0.25, 0.75]); iqr = q3 - q1
        out[c] = {"zscore_gt3_pct": 100 * (z.abs() > 3).mean(),
                  "iqr_rule_pct": 100 * ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean(),
                  "beyond_6sd": "; ".join(f"{d.date()} {100 * (np.exp(v) - 1):+.1f}%"
                                          for d, v in x[z.abs() > 6].items())}
    pd.DataFrame(out).T.to_csv(OUT / "a1_outliers.csv")
    print(f"\nA1 statistics, {df.index[0].date()} to {df.index[-1].date()}, {len(df)} days")
    print(desc.round(4).to_string())
    print(corr.round(3).to_string())
    print(pd.DataFrame(out).T.to_string())
    print("Equinor-Aker BP correlation", round(r["Equinor"].corr(r["Aker BP"]), 3))


STYLE = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": "#888", "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
         "legend.frameon": False, "savefig.dpi": 300}
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6f6f6f"


def fig_learning_curves() -> None:
    c = pd.read_csv(ROOT / "reports" / "lstm" / "curves" / "curves.csv")
    c = c[c.hidden_size == 32]
    ref = (c[c.part.str.endswith("_ref")].assign(part=lambda d: d.part.str[:-4])
           [["stock", "h", "seed", "part", "mse"]])
    cur = c[~c.part.str.endswith("_ref")].merge(ref, on=["stock", "h", "seed", "part"],
                                                suffixes=("", "_ref"))
    cur["ratio"] = cur.mse / cur.mse_ref
    m = cur.groupby(["stock", "h", "part", "epoch"]).ratio.mean().reset_index()
    style = {"fit": (BLUE, "-", "Training (fitting part)"),
             "es": (ORANGE, "--", "Early-stopping part (last 15 % of training)"),
             "val": (AQUA, "-", "Validation, 2023–2024")}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    rows = []
    for i, s in enumerate(STOCKS):
        for j, h in enumerate(HORIZONS):
            ax, g = axs[i, j], m[(m.stock == s) & (m.h == h)]
            for part, (col, ls, lab) in style.items():
                q = g[g.part == part]
                ax.plot(q.epoch, q.ratio, color=col, ls=ls, lw=1.4, label=lab)
            es, val, fit = (g[g.part == p].set_index("epoch").ratio for p in ("es", "val", "fit"))
            ax.axhline(1, color="#222", lw=0.8)
            ax.axvline(es.idxmin(), color="#222", lw=0.8, ls=":")
            rows.append({"stock": s, "h": h, "es_min_epoch": es.idxmin(), "val_min_epoch": val.idxmin(),
                         "val_at_last_epoch": val.iloc[-1], "fit_at_last_epoch": fit.iloc[-1]})
            ax.set_yscale("log"); ax.set_ylim(0.04, 8)
            ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.3, 1, 3]))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.grid(axis="y"); ax.set_title(f"{NAME[s]}, $h={h}$", fontsize=9)
            if j == 0:
                ax.set_ylabel("MSE / MSE of constant")
            if i == 1:
                ax.set_xlabel("Epoch")
    hd, lb = axs[0, 0].get_legend_handles_labels()
    fig.legend(hd + [plt.Line2D([], [], color="#222", ls=":", lw=0.8)], lb + ["Lowest early-stopping error"],
               loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "lstm_learning_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("\nLearning curves (ratios to the constant forecast)\n", pd.DataFrame(rows).round(3).to_string())


def fig_rolling_vol(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for col, lab, colr in [("EQNR_ret", "Equinor", BLUE), ("AKRBP_ret", "Aker BP", ORANGE),
                           ("osebx_ret", "OSEBX", GREY)]:
        ax.plot(df.index, 100 * df[col].rolling(21).std() * np.sqrt(252), color=colr, lw=1.1, label=lab)
    for part, txt in [("val", "Validation"), ("test", "Test")]:
        d = df.index[df.split == part]
        ax.axvspan(d.min(), d.max(), color="#000", alpha=0.06, lw=0)
        ax.text(d.min() + (d.max() - d.min()) / 2, 0.97, txt, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#444")
    ax.set_ylabel("Annualised volatility (%)"); ax.grid(axis="y"); ax.set_ylim(0, None)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "a1_fig3.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update(STYLE)
    FIG.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    df = load()

    preds = test_predictions(df)
    preds.to_csv(OUT / "test_predictions_all_models.csv", index=False)
    res = preds.groupby(["stock", "h", "model"]).apply(direction_scores)
    check_against_report(res)
    res.to_csv(OUT / "direction_test.csv")
    for col in ["acc", "balanced_acc", "share_up"]:
        t = res[col].unstack("model")[MODELS].reindex(CASES)
        t.loc[("Mean", 0), :] = t.mean()
        print(f"\n{col} (test, per cent)\n", t.round(1).to_string())
    print("\nConfusion matrices, Equinor h=5\n",
          res.loc[("EQNR", 5)].loc[["rf", "lstm"], ["TP", "FN", "FP", "TN"]].to_string())

    a1_statistics(df)
    fig_learning_curves()
    fig_rolling_vol(df)
    print(f"\nSaved to {OUT.relative_to(ROOT)}")

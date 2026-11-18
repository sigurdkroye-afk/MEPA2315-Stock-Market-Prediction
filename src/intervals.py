"""
intervals.py

80 % prediksjonsintervall for avkastningen til Equinor og Aker BP over
h = 3, 5 og 10 handelsdager. Oppgave til Isak, MEPA2315.

Metoder
    steg 1  fast         10. og 90. persentil av y i train, samme intervall hver dag
    steg 2  vol20        persentiler av y / vol, der vol = std(20 dager) * sqrt(h)
    steg 3  garch        som steg 2, men vol fra GARCH(1,1) med t-fordeling
    steg 4  kvantil_a/b  gradient boosting med kvantiltap, to valg av input
    steg 5  conformal    intervall rundt en punktprognose, kalibrert på val

Regler: alle tall læres fra train (conformal kalibreres på val, slik oppgaven sier),
val brukes til å velge metode, test kjøres én gang til slutt. Gap-dager brukes aldri.

Kjør fra roten av repoet
    python src/intervals.py          bare val (steg 1 til 4, og 5 hvis prognoser finnes)
    python src/intervals.py --test   også den ene testkjøringen, figur og kurser

Prognoser fra RF og LSTM (steg 5) leses fra reports/predictions/<modell>_<AKSJE>_h<h>.csv
med kolonnene Date, pred og dekker både val og test. Mangler filen, hoppes modellen over.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "data" / "processed" / "dataset.csv"
OUT = ROOT / "reports" / "intervals"
PRED_DIR = ROOT / "reports" / "predictions"

STOCKS = ["EQNR", "AKRBP"]
HORIZONS = [3, 5, 10]
LO, HI = 0.10, 0.90          # 80 % intervall
VOL_WINDOW = 20


# ---------------------------------------------------------------- hjelpere

def load(path: Path = DATA) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def evaluer(y, nedre, ovre) -> dict:
    y, nedre, ovre = (np.asarray(a, float) for a in (y, nedre, ovre))
    inne = (y >= nedre) & (y <= ovre)
    return {"dekning %": round(float(100 * inne.mean()), 1),
            "bredde pp": round(float(100 * np.mean(ovre - nedre)), 2)}


def interval_score(y, nedre, ovre, alpha=0.2) -> float:
    """Winkler/interval score i prosentpoeng: bredde pluss straff for bom. Lavere er bedre."""
    y, nedre, ovre = (np.asarray(a, float) for a in (y, nedre, ovre))
    s = (ovre - nedre) + 2 / alpha * ((nedre - y) * (y < nedre) + (y - ovre) * (y > ovre))
    return round(float(100 * s.mean()), 2)


def vol20(df: pd.DataFrame, aksje: str, h: int) -> pd.Series:
    # Bruker bare avkastning til og med dag t.
    return df[f"{aksje}_ret"].rolling(VOL_WINDOW).std() * np.sqrt(h)


def garch_sigma(df: pd.DataFrame, aksje: str) -> pd.DataFrame:
    """GARCH(1,1) med t-fordeling, lært bare på data før første val-dag.
    Returnerer volatilitet over h dager for h = 1..10, prognose laget på dag t."""
    forste_val = df.index[df["split"] == "val"][0]
    modell = arch_model(100 * df[f"{aksje}_ret"], vol="GARCH", p=1, q=1, dist="t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = modell.fit(last_obs=forste_val, disp="off")
        prog = res.forecast(horizon=max(HORIZONS), start=df.index[0], reindex=False)
    var = prog.variance                      # kolonne h.k = varians for dag t+k
    out = {h: np.sqrt(var.iloc[:, :h].sum(axis=1)) / 100 for h in range(1, max(HORIZONS) + 1)}
    return pd.DataFrame(out, index=var.index).reindex(df.index), res


def skalert_intervall(y, sigma, train):
    """Steg 2 og 3: persentiler av y / sigma i train, ganget med sigma hver dag."""
    z = (y / sigma)[train]
    qlo, qhi = z.quantile([LO, HI])
    return qlo * sigma, qhi * sigma


def kvantilmodell(alpha):
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=200,
                                     max_depth=2, learning_rate=0.05,
                                     min_samples_leaf=50, random_state=0)


def kvantil_intervall(X: pd.DataFrame, y, train):
    """Steg 4: én modell for nedre og én for øvre grense, trent på train."""
    ok = X.notna().all(axis=1)
    fit = train & ok & y.notna()
    nedre = kvantilmodell(LO).fit(X[fit], y[fit])
    ovre = kvantilmodell(HI).fit(X[fit], y[fit])
    lo = pd.Series(np.nan, index=X.index)
    hi = pd.Series(np.nan, index=X.index)
    lo[ok] = nedre.predict(X[ok])
    hi[ok] = ovre.predict(X[ok])
    # Kvantilmodellene kan i sjeldne tilfeller krysse. Sorter så nedre <= øvre.
    return np.minimum(lo, hi), np.maximum(lo, hi)


def conformal(y, pred, sigma, cal, alpha=0.2, skaler=True):
    """Steg 5: split conformal. Kalibrerer på `cal` (val), gir intervall for alle dager.
    Med skaler=True deles bommene på volatiliteten (steg 2), så intervallet følger uroen."""
    s = sigma if skaler else pd.Series(1.0, index=y.index)
    bom = (y - pred).abs() / s
    n = int(cal.sum())
    # Endelig-utvalg-korreksjon fra Angelopoulos & Bates (2021).
    nivaa = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = bom[cal].quantile(nivaa)
    return pred - q * s, pred + q * s


# ---------------------------------------------------------------- alle metoder

def alle_intervall(df: pd.DataFrame, aksje: str, h: int, gsig: pd.DataFrame) -> dict:
    y = df[f"y_{aksje}_h{h}"]
    train = (df["split"] == "train") & y.notna()
    v20 = vol20(df, aksje, h)
    g = gsig[h]

    iv = {}
    # Steg 1
    lo, hi = y[train].quantile([LO, HI])
    iv["fast"] = (pd.Series(lo, index=df.index), pd.Series(hi, index=df.index))
    # Steg 2
    iv["vol20"] = skalert_intervall(y, v20, train & v20.notna())
    # Steg 3
    iv["garch"] = skalert_intervall(y, g, train & g.notna())
    # Steg 4, input a: det oppgaven foreslår
    Xa = pd.DataFrame({"vol20": v20, "vix": df["vix"],
                       "range": df[f"{aksje}_range"], "absret": df[f"{aksje}_ret"].abs()})
    iv["kvantil_a"] = kvantil_intervall(Xa, y, train)
    # Steg 4, input b: GARCH i stedet for vol20, pluss oljeuro og markedsuro
    Xb = pd.DataFrame({"garch": g, "vix": df["vix"], "range": df[f"{aksje}_range"],
                       "absret": df[f"{aksje}_ret"].abs(),
                       "brent_vol20": df["brent_ret"].rolling(VOL_WINDOW).std() * np.sqrt(h),
                       "osebx_vol20": df["osebx_ret"].rolling(VOL_WINDOW).std() * np.sqrt(h)})
    iv["kvantil_b"] = kvantil_intervall(Xb, y, train)
    return iv


def tabell(df, gsig, part: str, metoder=None) -> pd.DataFrame:
    rader = []
    for aksje in STOCKS:
        for h in HORIZONS:
            y = df[f"y_{aksje}_h{h}"]
            mask = (df["split"] == part) & y.notna()
            iv = alle_intervall(df, aksje, h, gsig[aksje])
            for navn, (lo, hi) in iv.items():
                if metoder and navn not in metoder:
                    continue
                assert lo[mask].notna().all() and hi[mask].notna().all(), (aksje, h, navn)
                r = evaluer(y[mask], lo[mask], hi[mask])
                rader.append({"aksje": aksje, "h": h, "metode": navn, **r,
                              "score": interval_score(y[mask], lo[mask], hi[mask])})
    return pd.DataFrame(rader)


def bred(t: pd.DataFrame) -> pd.DataFrame:
    """Steg side om side: én rad per aksje og horisont."""
    w = t.pivot_table(index=["aksje", "h"], columns="metode",
                      values=["dekning %", "bredde pp"], sort=False)
    w.columns = [f"{m} {v.split()[0]}" for v, m in w.columns]
    orden = [f"{m} {v}" for m in ["fast", "vol20", "garch", "kvantil_a", "kvantil_b"]
             for v in ("dekning", "bredde") if f"{m} {v}" in w.columns]
    return w[orden]


def regimer(df, gsig, part, metoder) -> pd.DataFrame:
    """Betinget dekning: rolige vs urolige dager (delt på median vol20 i perioden).
    Et godt intervall har dekning nær 80 % i begge."""
    rader = []
    for aksje in STOCKS:
        for h in HORIZONS:
            y = df[f"y_{aksje}_h{h}"]
            mask = (df["split"] == part) & y.notna()
            v = vol20(df, aksje, h)
            urolig = v > v[mask].median()
            iv = alle_intervall(df, aksje, h, gsig[aksje])
            for navn in metoder:
                lo, hi = iv[navn]
                for reg, m in (("rolig", mask & ~urolig), ("urolig", mask & urolig)):
                    rader.append({"aksje": aksje, "h": h, "metode": navn, "regime": reg,
                                  **evaluer(y[m], lo[m], hi[m])})
    return pd.DataFrame(rader)


# ---------------------------------------------------------------- steg 5

def les_prognose(modell: str, aksje: str, h: int, index) -> pd.Series | None:
    f = PRED_DIR / f"{modell}_{aksje}_h{h}.csv"
    if not f.exists():
        return None
    p = pd.read_csv(f, parse_dates=["Date"], index_col="Date")["pred"]
    return p.reindex(index)


def steg5(df, modeller=("zero", "rf", "lstm"), part="test") -> pd.DataFrame:
    """Conformal rundt punktprognoser. 'zero' (prognose 0) er med som referanse
    og fungerer uten filer, så metoden kan sjekkes før RF og LSTM er klare."""
    rader = []
    for aksje in STOCKS:
        for h in HORIZONS:
            y = df[f"y_{aksje}_h{h}"]
            cal = (df["split"] == "val") & y.notna()
            ev = (df["split"] == part) & y.notna()
            sigma = vol20(df, aksje, h)
            for modell in modeller:
                pred = (pd.Series(0.0, index=df.index) if modell == "zero"
                        else les_prognose(modell, aksje, h, df.index))
                if pred is None or pred[cal | ev].isna().any():
                    continue
                for skaler in (False, True):
                    lo, hi = conformal(y, pred, sigma, cal, skaler=skaler)
                    rader.append({"aksje": aksje, "h": h,
                                  "metode": f"conf_{modell}" + ("_vol" if skaler else ""),
                                  **evaluer(y[ev], lo[ev], hi[ev]),
                                  "score": interval_score(y[ev], lo[ev], hi[ev])})
    return pd.DataFrame(rader)


# ---------------------------------------------------------------- test, kurs, figur

def kurser(df, gsig, metode, part="test") -> pd.DataFrame:
    """Intervallet som kurs: close_t * exp(grense)."""
    rader = []
    for aksje in STOCKS:
        for h in HORIZONS:
            y = df[f"y_{aksje}_h{h}"]
            mask = (df["split"] == part) & y.notna()
            lo, hi = alle_intervall(df, aksje, h, gsig[aksje])[metode]
            close = df[f"{aksje}_close"]
            d = pd.DataFrame({"aksje": aksje, "h": h, "close": close,
                              "nedre_kurs": close * np.exp(lo),
                              "ovre_kurs": close * np.exp(hi),
                              "nedre_avk": lo, "ovre_avk": hi, "faktisk_avk": y})[mask]
            rader.append(d)
    return pd.concat(rader).round(4)


def figur(df, gsig, metode, fil: Path, aksje="EQNR", h=5, part="test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = df[f"y_{aksje}_h{h}"]
    mask = (df["split"] == part) & y.notna()
    iv = alle_intervall(df, aksje, h, gsig[aksje])
    lo, hi = iv[metode]
    flo, fhi = iv["fast"]
    d = df.index[mask]
    inne = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(d, 100 * lo[mask], 100 * hi[mask], color="#4C78A8", alpha=0.25,
                    label=f"80 % intervall ({metode})", step="mid")
    ax.plot(d, 100 * flo[mask], color="#888", lw=1, ls="--", label="fast intervall (steg 1)")
    ax.plot(d, 100 * fhi[mask], color="#888", lw=1, ls="--")
    ax.scatter(d[inne], 100 * y[mask][inne], s=7, color="#222", label="faktisk avkastning, inne")
    ax.scatter(d[~inne], 100 * y[mask][~inne], s=12, color="#D62728", label="faktisk avkastning, ute")
    ax.axhline(0, color="#bbb", lw=0.6)
    r = evaluer(y[mask], lo[mask], hi[mask])
    ax.set_title(f"{aksje}, h = {h}, testperioden: dekning {r['dekning %']} %, "
                 f"bredde {r['bredde pp']} pp")
    ax.set_ylabel(f"avkastning over {h} handelsdager, %")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fil, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="kjør den ene testkjøringen")
    ap.add_argument("--metode", default="garch,vol20",
                    help="metodene som er valgt på val, kommaseparert. Den første brukes i figur og kurser")
    args = ap.parse_args()

    pd.set_option("display.width", 200, "display.max_columns", 30)
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()

    gsig, params = {}, {}
    for a in STOCKS:
        gsig[a], res = garch_sigma(df, a)
        params[a] = res.params.round(4)
    print("GARCH-parametre (lært før val):\n", pd.DataFrame(params), "\n")

    # Sjekk mot oppgaveteksten: EQNR h=5 steg 1 skal gi ca 82,5 % og 9,0 pp.
    val = tabell(df, gsig, "val")
    print("VAL, alle metoder (score = interval score i pp, lavere er bedre)")
    print(val.to_string(index=False), "\n")
    print("VAL, side om side")
    print(bred(val).to_string(), "\n")
    print("VAL, snitt over aksjer og horisonter")
    snitt = val.groupby("metode", sort=False)[["dekning %", "bredde pp", "score"]].mean().round(2)
    snitt["avvik fra 80"] = (snitt["dekning %"] - 80).abs().round(2)
    print(snitt.to_string(), "\n")
    reg_val = regimer(df, gsig, "val", ["fast", "vol20", "garch", "kvantil_a", "kvantil_b"])
    print("VAL, dekning på rolige og urolige dager")
    print(reg_val.pivot_table(index="metode", columns="regime", values="dekning %",
                              sort=False).round(1).to_string(), "\n")

    val.to_csv(OUT / "val_alle.csv", index=False)
    bred(val).to_csv(OUT / "val_tabell.csv")
    reg_val.to_csv(OUT / "val_regimer.csv", index=False)

    if args.test:
        valgte = args.metode.split(",")
        valgt = ["fast"] + valgte
        test = tabell(df, gsig, "test", metoder=valgt)
        s5 = steg5(df)
        test_all = pd.concat([test, s5], ignore_index=True)
        print(f"TEST (kjørt én gang), valgte metoder {valgte}, fast som referanse")
        print(test_all.to_string(index=False), "\n")
        reg_test = regimer(df, gsig, "test", valgt)
        print("TEST, dekning på rolige og urolige dager")
        print(reg_test.pivot_table(index="metode", columns="regime", values="dekning %",
                                   sort=False).round(1).to_string(), "\n")
        test_all.to_csv(OUT / "test_alle.csv", index=False)
        reg_test.to_csv(OUT / "test_regimer.csv", index=False)
        k = kurser(df, gsig, valgte[0])
        k.to_csv(OUT / "test_kurser.csv", index_label="Date")
        print("Siste testdag som kurs:")
        print(k.groupby(["aksje", "h"]).tail(1).to_string(), "\n")
        figur(df, gsig, valgte[0], OUT / "eqnr_h5_test.png")
        print("Lagret i", OUT.relative_to(ROOT))

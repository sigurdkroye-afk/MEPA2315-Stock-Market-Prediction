"""Shared fixtures. Synthetic data only: the tests never read data/processed/dataset.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

torch.set_num_threads(1)       # tiny LSTMs are much faster single-threaded on CPU

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data" / "processed")]

from datasets import BASE_FEATURES, STOCKS  # noqa: E402

HORIZONS = [3, 5, 10]
N_ROWS = 600
GAP = 10
N_TRAIN, N_VAL = 340, 100      # rows labelled train / val, each followed by GAP gap rows


@pytest.fixture
def df():
    """Synthetic frame with the same layout as dataset.csv:
    train, 10 gap, val, 10 gap, test (labels as in prepare_data)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=N_ROWS, name="Date")
    out = pd.DataFrame(rng.normal(size=(N_ROWS, len(BASE_FEATURES))), index=dates, columns=BASE_FEATURES)

    split = np.full(N_ROWS, "test", dtype=object)
    split[:N_TRAIN] = "train"
    split[N_TRAIN:N_TRAIN + GAP] = "gap"
    split[N_TRAIN + GAP:N_TRAIN + GAP + N_VAL] = "val"
    split[N_TRAIN + GAP + N_VAL:N_TRAIN + 2 * GAP + N_VAL] = "gap"
    out.insert(0, "split", split)

    for s in STOCKS:
        out[f"{s}_adjclose"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, N_ROWS)))
        logp = np.log(out[f"{s}_adjclose"])
        for h in HORIZONS:
            out[f"y_{s}_h{h}"] = logp.shift(-h) - logp   # NaN for the last h rows
    return out

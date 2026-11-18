"""Tests for lstm_curves.curves on the synthetic frame (never reads the real dataset)."""
from datasets import STOCKS
from lstm_curves import curves


def test_curves_layout(df):
    cfg = {"lookback": 10, "hidden_size": 8, "dropout": 0.2}
    c = curves(df, runs=[(STOCKS[0], 3, 8)], cfg=cfg, epochs=2, seeds=1)
    assert list(c.columns) == ["stock", "h", "hidden_size", "seed", "epoch", "part", "mse"]
    assert len(c) == 1 * 1 * (3 * 2 + 3)
    assert not c["mse"].isna().any()
    assert set(c["part"]) == {"fit", "es", "val", "fit_ref", "es_ref", "val_ref"}

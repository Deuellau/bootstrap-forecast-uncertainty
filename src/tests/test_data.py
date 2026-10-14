import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import descriptives, load_log_returns, upto, yearly_table


@pytest.fixture
def csv(tmp_path):
    """Synthetic data in the real file's format (first row = account base date with return 0)."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-09-21", periods=600)
    df = pd.DataFrame(rng.normal(0.0005, 0.01, (600, len(C.ASSETS))), index=dates, columns=C.ASSETS)
    df.iloc[0, 0] = 0.0
    p = tmp_path / "r.csv"
    df.rename_axis("date").to_csv(p)
    return p


def test_base_row_dropped_and_logs(csv):
    raw = pd.read_csv(csv, parse_dates=["date"]).set_index("date")
    lr = load_log_returns(csv)
    assert len(lr) == len(raw) - 1 and lr.index[0] == raw.index[1]
    assert np.expm1(lr["spy"].sum()) == pytest.approx((1 + raw["spy"].iloc[1:]).prod() - 1)


def test_upto_has_no_future_data(csv):
    lr = load_log_returns(csv)
    h = upto(lr["acct"], "2023-06-30")
    assert h.index.max() <= pd.Timestamp("2023-06-30")
    assert len(h) < len(lr)


def test_yearly_table_counts_days(csv):
    lr = load_log_returns(csv)
    t = yearly_table(lr)
    assert t.days.sum() == len(lr)
    assert list(t.columns) == ["days"] + ["vol_" + a for a in C.ASSETS]


def test_descriptives_shapes(csv):
    d = descriptives(load_log_returns(csv))
    assert list(d.index) == C.ASSETS
    assert (d.vol_ann > 0).all()


def test_loader_rejects_unsorted_dates(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-02", "2024-01-01"], **{a: [0.01, 0.02] for a in C.ASSETS}}).to_csv(p, index=False)
    with pytest.raises(AssertionError):
        load_log_returns(p)

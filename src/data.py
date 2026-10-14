from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parents[1]


def load_log_returns(path=None):
    """Read daily simple returns and return daily log returns, ln(1 + R), so multi-day returns add.

    The first row is the account's base date (return exactly 0.0, not a real return) and is dropped.
    """
    df = pd.read_csv(path or ROOT / C.DATA_FILE, parse_dates=["date"]).set_index("date")
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert not df.isna().any().any()
    df = df[C.ASSETS]                       # keep only the series used in the study
    if df["acct"].iloc[0] == 0:
        df = df.iloc[1:]
    return np.log1p(df)


def upto(s, origin):
    """History up to and including `origin`; nothing after it (no look-ahead)."""
    return s[s.index <= pd.Timestamp(origin)]


def yearly_table(lr):
    """Per calendar year: trading days and annualised volatility of each asset."""
    g = lr.groupby(lr.index.year)
    out = g.std().mul(np.sqrt(C.N)).add_prefix("vol_")
    out.insert(0, "days", g.size())
    return out.rename_axis("year")


def descriptives(lr):
    """Whole-sample summary per asset: annualised mean and volatility, skew, excess kurtosis."""
    return pd.DataFrame({
        "days": lr.count(),
        "mean_ann": lr.mean() * C.N,
        "vol_ann": lr.std() * np.sqrt(C.N),
        "skew": lr.skew(),
        "excess_kurtosis": lr.kurt(),
    }).rename_axis("asset")

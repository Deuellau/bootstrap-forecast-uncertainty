"""Turn daily returns of an account (already in SGD) and three US-listed ETFs (in USD) into one SGD file.

Input: a DataFrame of daily simple returns (decimals) with columns acct, spy, bnd, gld, indexed by date, and
data/usdsgd_raw.csv (columns Date, usdsgd = SGD per USD). Output: data/returns_daily_sgd.csv on US trading days.
Reading a particular report format into that DataFrame is done outside the repository.

Calendar. A US holiday row is one where SPY and GLD are both exactly 0. Its account return is compounded into the
next trading day, so the compounded account return equals the platform's. Benchmark returns are converted to SGD with
R_sgd = (1 + R_usd) * FX_t / FX_prev - 1, where FX_prev is the FX close on the previous trading day, so holiday FX
moves land on the next trading day.
"""
import pandas as pd

from .data import ROOT

def to_trading_days(d):
    hol = (d["spy"] == 0) & (d["gld"] == 0)
    key = d.index.to_series().where(~hol).bfill()                  # each row belongs to its next trading day
    assert not key.isna().any(), "export ends on a holiday row"
    out = (1 + d).groupby(key.values).prod() - 1                    # compound the holiday rows into the next trading day
    for s in ["spy", "bnd", "gld"]:
        out[s] = d.loc[~hol, s].values                              # benchmarks are exactly 0 on holidays
    return out, int(hol.sum())


def to_sgd(out, fx):
    idx = out.index
    fx_t = fx.reindex(fx.index.union(idx)).ffill().reindex(idx)
    prev = idx.to_series().shift(1)
    prev.iloc[0] = fx.index[fx.index < idx[0]].max()                # last FX close before the first date
    fx_prev = fx.reindex(fx.index.union(prev)).ffill().reindex(prev.values)
    g = fx_t.values / fx_prev.values
    res = out.copy()
    for s in ["spy", "bnd", "gld"]:
        res[s] = (1 + out[s]) * g - 1
    return res


def main(d):
    out, nhol = to_trading_days(d)
    fx = pd.read_csv(ROOT / "data" / "usdsgd_raw.csv", parse_dates=["Date"]).set_index("Date")["usdsgd"]
    res = to_sgd(out, fx)
    res.rename_axis("date").to_csv(ROOT / "data" / "returns_daily_sgd.csv")
    print(f"{len(d)} input rows, {nhol} holiday rows folded into the next trading day, {len(res)} trading days written")
    print(f"{res.index[0].date()} to {res.index[-1].date()}; account cumulative return {(1 + res.acct).prod() - 1:.4f}")

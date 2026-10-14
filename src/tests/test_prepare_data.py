import pandas as pd
import pytest

from src import prepare_data as P

ROWS = [("2022-09-21", 0.005, 0.002, -0.010, 0.000),
        ("2022-09-22", 0.010, 0.001, 0.020, 0.010),
        ("2022-09-23", 0.000, 0.000, 0.000, 0.005),     # US holiday: SPY and GLD exactly 0, account moves
        ("2022-09-26", -0.010, 0.003, 0.010, 0.020)]


def returns():
    d = pd.DataFrame(ROWS, columns=["date", "gld", "bnd", "spy", "acct"])
    return d.assign(date=pd.to_datetime(d.date)).set_index("date")[["acct", "spy", "bnd", "gld"]]


def test_holiday_account_return_is_compounded_into_next_trading_day():
    d = returns()
    out, nhol = P.to_trading_days(d)
    assert nhol == 1 and len(out) == 3
    assert out.loc["2022-09-26", "acct"] == pytest.approx(1.005 * 1.02 - 1)
    assert (1 + out.acct).prod() == pytest.approx((1 + d.acct).prod())       # total account return preserved
    assert out.loc["2022-09-26", "spy"] == pytest.approx(0.01)


def test_sgd_conversion_uses_previous_trading_day_fx():
    idx = pd.to_datetime(["2022-09-21", "2022-09-22", "2022-09-26"])
    out = pd.DataFrame({"acct": 0.0, "spy": [0.0, 0.02, 0.01], "bnd": 0.0, "gld": 0.0}, index=idx)
    fx = pd.Series([1.40, 1.40, 1.40, 1.50, 1.50], index=pd.to_datetime(
        ["2022-09-20", "2022-09-21", "2022-09-22", "2022-09-23", "2022-09-26"]))
    res = P.to_sgd(out, fx)
    assert res.loc["2022-09-22", "spy"] == pytest.approx(1.02 * 1.40 / 1.40 - 1)
    assert res.loc["2022-09-26", "spy"] == pytest.approx(1.01 * 1.50 / 1.40 - 1)   # holiday FX move lands here
    assert res.loc["2022-09-26", "acct"] == 0.0                                  # account is already in SGD

import numpy as np
import pytest

from src import config as C
from src import evaluate as E
from src import methods as M
from src import simulate as S
from src import theory as T


def rng(s=0):
    return np.random.default_rng(s)


def test_iid_normal_moments():
    r = S.iid_normal(5_000_000, rng(1))
    assert r.mean() * C.N == pytest.approx(C.MU_ANNUAL, abs=0.01)
    assert r.std() * np.sqrt(C.N) == pytest.approx(C.SIGMA_ANNUAL, rel=0.01)


def test_outcome_is_the_continuation_of_the_same_path():
    history, outcome = S.make_case(300, 63, rng(7))
    full = S.iid_normal(363, rng(7))
    assert np.array_equal(history, full[:300])
    assert outcome == pytest.approx(full[300:].sum())


def test_band_is_central_and_ordered():
    x = np.arange(1, 1001)
    lo, hi = E.band(x, 0.80)
    assert lo == pytest.approx(np.quantile(x, 0.10)) and hi == pytest.approx(np.quantile(x, 0.90))
    assert lo < hi


def test_replication_is_reproducible_and_wellformed():
    a = E.one_replication(250, 63, 3, B=300)
    b = E.one_replication(250, 63, 3, B=300)
    assert a == b
    assert set(a) == {(m, l) for m in M.METHODS for l in C.LEVELS}
    for (m, l), (inside, width) in a.items():
        assert width > 0
    # the wider band at a higher level always contains the narrower one's verdict
    for m in M.METHODS:
        assert a[(m, 0.95)][1] > a[(m, 0.80)][1]


def test_small_grid_matches_theory_and_nominal():
    df = E.run_grid(reps=400, B=1000, grid_n=(250,), grid_h=(63, 250), workers=2)
    s = df[(df.method == "simple") & (df.level == 0.95)]
    for _, r in s.iterrows():
        assert r.coverage == pytest.approx(r.theory, abs=0.05)
    simple = df[df.method == "simple"].sort_values(["n", "H", "level"]).coverage.values
    mean_unc = df[df.method == "mean_uncertainty"].sort_values(["n", "H", "level"]).coverage.values
    assert (simple <= mean_unc + 0.03).all()
    m = df[(df.method == "mean_uncertainty") & (df.level == 0.95)]
    assert (m.coverage > 0.90).all()
    assert (df.width_vs_simple[df.method == "simple"] == 1).all()

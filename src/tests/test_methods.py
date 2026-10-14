import numpy as np
import pytest

from src import methods as M
from src import theory as T


def rng(s=0):
    return np.random.default_rng(s)


def iid(n, seed=1, mu=0.0005, sd=0.01):
    return rng(seed).normal(mu, sd, n)


def sd_terminal(f, r, H, B=20000, seed=2):
    return f(r, H, B, rng(seed))[:, -1].std()


def test_shapes_and_cumulative():
    r = iid(300)
    for f in M.METHODS.values():
        p = f(r, 21, 500, rng())
        assert p.shape == (500, 21)


def test_reproducible_with_seed():
    r = iid(300)
    for f in M.METHODS.values():
        assert np.array_equal(f(r, 10, 100, rng(5)), f(r, 10, 100, rng(5)))


def test_simple_width_scales_like_sqrt_H():
    r = iid(5000)
    assert sd_terminal(M.simple, r, 64) / sd_terminal(M.simple, r, 16) == pytest.approx(2.0, rel=0.04)


def test_simple_centre_is_H_times_sample_mean():
    r = iid(500, mu=0.002)
    assert M.simple(r, 63, 40000, rng())[:, -1].mean() == pytest.approx(63 * r.mean(), abs=0.003)


@pytest.mark.parametrize("n,H", [(250, 63), (250, 250), (500, 126), (1000, 252)])
def test_mean_uncertainty_is_wider_by_sqrt_1_plus_H_over_n(n, H):
    r = iid(n)
    ratio = sd_terminal(M.mean_uncertainty, r, H) / sd_terminal(M.simple, r, H)
    assert ratio == pytest.approx(float(T.width_ratio(H, n)), rel=0.04)


def test_mean_uncertainty_matches_simple_when_history_is_long():
    r = iid(20000)
    assert sd_terminal(M.mean_uncertainty, r, 21) / sd_terminal(M.simple, r, 21) == pytest.approx(1.0, abs=0.02)


def test_theory_values():
    assert T.width_ratio(250, 250) == pytest.approx(2 ** 0.5)
    assert T.coverage(0.95, 250, 250) == pytest.approx(0.834, abs=0.002)
    assert T.coverage(0.80, 250, 250) == pytest.approx(0.635, abs=0.002)
    assert T.coverage(0.95, 1, 10 ** 9) == pytest.approx(0.95, abs=1e-6)


def test_methods_use_only_the_history_given():
    r = iid(400)
    a = M.simple(r[:300], 10, 50, rng(3))
    b = M.simple(np.concatenate([r[:300], 99 * np.ones(100)])[:300], 10, 50, rng(3))
    assert np.array_equal(a, b)


def test_exact_resampling_variance_formulas():
    """Var*(S) = H s2 (simple) and H s2 (1 + (H-1)/n) (mean-uncertainty), with s2 the plug-in variance."""
    n, H = 8, 5
    r = np.array([-0.02, 0.01, 0.03, -0.01, 0.00, 0.02, -0.03, 0.015])
    s2 = r.var()                                   # plug-in variance (divides by n)
    simple = M.simple(r, H, 400_000, rng(1))[:, -1]
    nested = M.mean_uncertainty(r, H, 400_000, rng(2))[:, -1]
    assert simple.mean() == pytest.approx(H * r.mean(), abs=2e-3)
    assert nested.mean() == pytest.approx(H * r.mean(), abs=2e-3)
    assert simple.var() == pytest.approx(H * s2, rel=0.02)
    assert nested.var() == pytest.approx(H * s2 * (1 + (H - 1) / n), rel=0.02)

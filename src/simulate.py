"""Simulated worlds with a known truth."""
import numpy as np

from . import config as C


def iid_normal(n_total, rng, mu_annual=None, sigma_annual=None):
    """i.i.d. Normal daily log returns with the given annual mean and volatility."""
    mu = C.MU_ANNUAL if mu_annual is None else mu_annual
    sigma = C.SIGMA_ANNUAL if sigma_annual is None else sigma_annual
    return rng.normal(mu / C.N, sigma / np.sqrt(C.N), n_total)


def make_case(n, H, rng):
    """One path of n + H days: the first n are the history, the outcome is the sum of the next H.

    The outcome is the continuation of the same path, never an independent draw.
    """
    r = iid_normal(n + H, rng)
    return r[:n], float(r[n:].sum())

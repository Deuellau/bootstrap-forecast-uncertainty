"""Bootstrap forecasting methods.

Each takes the history `r` (1-D array of daily log returns up to the forecast origin), a horizon `H` in days,
the number of paths `B` and a numpy Generator, and returns an array of shape (B, H) of simulated
cumulative log-return paths. The last column is the H-day return.
"""
import numpy as np


def simple(r, H, B, rng):
    """Resample past daily returns i.i.d. with replacement."""
    r = np.asarray(r)
    return np.cumsum(r[rng.integers(0, len(r), size=(B, H))], axis=1)


def mean_uncertainty(r, H, B, rng, chunk=2000):
    """Two-stage (nested) bootstrap that carries the uncertainty of the sample mean.

    Stage 1 draws a pseudo-history of n days from `r` for every path (its mean differs from the sample mean
    by about s/sqrt(n)); stage 2 draws the H forecast days from that pseudo-history.
    The pseudo-history must have exactly n days: a different size would change the uncertainty of its mean.
    """
    r = np.asarray(r)
    n = len(r)
    out = np.empty((B, H))
    for s in range(0, B, chunk):
        m = min(chunk, B - s)
        pseudo = r[rng.integers(0, n, size=(m, n))]
        pick = rng.integers(0, n, size=(m, H))
        out[s:s + m] = np.cumsum(np.take_along_axis(pseudo, pick, axis=1), axis=1)
    return out


METHODS = {"simple": simple, "mean_uncertainty": mean_uncertainty}

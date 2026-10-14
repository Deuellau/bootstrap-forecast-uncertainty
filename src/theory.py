"""Closed-form results for the plain bootstrap (see the paper's Proposition 1)."""
import numpy as np
from scipy.stats import norm


def width_ratio(H, n):
    """True forecast-error sd divided by the plain bootstrap's sd: sqrt(1 + H/n)."""
    return np.sqrt(1 + np.asarray(H) / np.asarray(n))


def coverage(level, H, n):
    """Approximate coverage of a nominal central band of the plain bootstrap: 2 Phi(z / sqrt(1+H/n)) - 1."""
    z = norm.ppf(0.5 + np.asarray(level) / 2)
    return 2 * norm.cdf(z / width_ratio(H, n)) - 1

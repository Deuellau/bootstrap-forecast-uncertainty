"""Coverage of bootstrap bands in a world with a known truth."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import methods as M
from . import simulate as S
from . import theory as T


def band(sums, level):
    """Central band at the given level: the empirical (1-level)/2 and (1+level)/2 quantiles."""
    a = (1 - level) / 2
    return np.quantile(sums, [a, 1 - a])


def one_replication(n, H, rep, B=None, methods=None):
    """One history, one outcome, one band per method and level. Returns {(method, level): (inside, width)}."""
    B = C.B_SIM if B is None else B
    methods = methods or M.METHODS
    rng = np.random.default_rng([C.SEED, n, H, rep])
    history, outcome = S.make_case(n, H, rng)
    out = {}
    for name, f in methods.items():
        sums = f(history, H, B, rng)[:, -1]
        for level in C.LEVELS:
            lo, hi = band(sums, level)
            out[(name, level)] = (lo <= outcome <= hi, hi - lo)
    return out


def _chunk(args):
    n, H, reps, B = args
    return [(r, one_replication(n, H, r, B)) for r in reps]


def run_grid(reps=None, B=None, grid_n=None, grid_h=None, chunk=50, workers=None):
    """Coverage and mean width for every (n, H, method, level) cell. One forecast per replication."""
    reps = C.SIM_REPS if reps is None else reps
    B = C.B_SIM if B is None else B
    tasks = [(n, H, list(range(s, min(s + chunk, reps))), B)
             for n in (grid_n or C.GRID_N) for H in (grid_h or C.GRID_H) for s in range(0, reps, chunk)]
    acc = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (n, H, _, _), res in zip(tasks, ex.map(_chunk, tasks)):
            for _, d in res:
                for key, (inside, width) in d.items():
                    acc.setdefault((n, H) + key, []).append((inside, width))
    rows = []
    for (n, H, method, level), v in sorted(acc.items()):
        v = np.array(v, dtype=float)
        p = v[:, 0].mean()
        rows.append(dict(n=n, H=H, H_over_n=H / n, method=method, level=level, reps=len(v), coverage=p,
                         se=np.sqrt(p * (1 - p) / len(v)), width=v[:, 1].mean(),
                         theory=float(T.coverage(level, H, n)) if method == "simple" else level))
    df = pd.DataFrame(rows)
    base = df[df.method == "simple"].set_index(["n", "H", "level"]).width
    df["width_vs_simple"] = df.width.values / base.reindex(pd.MultiIndex.from_frame(df[["n", "H", "level"]])).values
    return df

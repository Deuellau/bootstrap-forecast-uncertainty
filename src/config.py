"""Fixed choices used so far. Priors and simulation settings are added with the parts that use them."""
N = 252                                   # trading days per year
DATA_FILE = "data/returns_daily_sgd.csv"  # local only, never committed
ASSETS = ["acct", "spy", "bnd", "gld"]

SEED = 20261012                           # base seed; every random draw derives from it
B = 20_000                                # simulated paths per real-data forecast

# Part 3: i.i.d. simulation study (frozen before any simulation was run)
SIGMA_ANNUAL = 0.28                       # the account's own annualised volatility, rounded
MU_ANNUAL = 0.08                          # true mean of the annual log return in the simulated world
LEVELS = (0.80, 0.95)                     # nominal band levels
B_SIM = 2_000                             # simulated paths per forecast inside the simulation loops
SIM_REPS = 2_000                          # independent replications per (n, H) cell
GRID_N = (250, 500, 750, 1000)            # history lengths in days
GRID_H = (21, 63, 126, 252)               # horizons in days

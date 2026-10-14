"""Fixed choices used so far. Priors and simulation settings are added with the parts that use them."""
N = 252                                   # trading days per year
DATA_FILE = "data/returns_daily_sgd.csv"  # local only, never committed
ASSETS = ["acct", "spy", "bnd", "gld"]

SEED = 20261012                           # base seed; every random draw derives from it
B = 20_000                                # simulated paths per real-data forecast

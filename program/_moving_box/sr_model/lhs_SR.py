"""
Latin Hypercube Sampling for COMSOL Parametric Sweep - "Specified combinations" mode.

COMSOL note: In "Specified combinations" mode, the i-th value of C_rate is paired
with the i-th value of u_par (not a full Cartesian grid), so N combinations yield
exactly N simulations. This script generates those paired (C_rate, u_par) samples.
"""

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
N = int(10000)                       # number of sample points (simulations to run)

PARAMS = {
    # name        : (lower_bound, upper_bound, decimal_places)
    "u_par":  (0.0, 30.0, 3),
    "C_rate": (0.0,  5.0,  3),    
    'SOC': (0.0, 1.0, 3)
}

SAVE_CSV     = True
CSV_PATH     = "lhs_samples_2.csv"
RANDOM_SEED  = None             # set to None for a different draw each run
# ──────────────────────────────────────────────────────────────────────────────


def latin_hypercube_sample(n, bounds, decimals, seed = None):
    """
    Standard LHS: divide [0,1]^d into n equal strata per dimension,
    place one sample per stratum, then shuffle independently per dimension.
    """
    rng  = np.random.default_rng(seed)
    d    = len(bounds)
    cuts = np.linspace(0.0, 1.0, n + 1)          # stratum edges

    samples = np.zeros((n, d))
    for j in range(d):
        lo, hi = bounds[j]
        # one uniform draw per strat map to [lo, hi]
        u         = rng.uniform(cuts[:-1], cuts[1:])   # shape (n,)
        perm      = rng.permutation(n)
        scaled    = lo + u[perm] * (hi - lo)
        samples[:, j] = np.round(scaled, decimals[j])

    return samples


def build_comsol_string(values: np.ndarray) -> str:
    """Return a brace-wrapped, comma-separated string ready for COMSOL."""
    inner = ", ".join(str(v) for v in values)
    return "{" + inner + "}"


# ── Run ────────────────────────────────────────────────────────────────────────
param_names   = list(PARAMS.keys())
bounds_list   = [(v[0], v[1]) for v in PARAMS.values()]
decimals_list = [v[2]         for v in PARAMS.values()]

raw = latin_hypercube_sample(N, bounds_list, decimals_list, seed=RANDOM_SEED)

# 1. DataFrame
df = pd.DataFrame(raw, columns=param_names)
df.insert(0, "case_id", range(1, N + 1))

# 3. Optional CSV export
if SAVE_CSV:
    df.to_csv(CSV_PATH, index=False)
    print(f"  Samples saved → {CSV_PATH}")

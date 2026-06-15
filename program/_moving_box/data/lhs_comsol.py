"""
Latin Hypercube Sampling for COMSOL Parametric Sweep
"""

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
N = 15                       # number of sample points (simulations to run)

PARAMS = {
    # name        : (lower_bound, upper_bound, decimal_places)
    # "C_rate": (1.0,  5.0,  1),    # Pulse
    "u_par":  (0.0, 0, 1),
    "C_rate": (0.5,  5.0,  2),     # CC
    # "u_par":  (0.0, 0.0, 1),
}

SAVE_CSV     = False
CSV_PATH     = "lhs_samples.csv"
RANDOM_SEED  = None             # set to None for a different draw each run
# ──────────────────────────────────────────────────────────────────────────────


def latin_hypercube_sample(n: int, bounds: list[tuple], decimals: list[int],
                            seed: int | None = None) -> np.ndarray:
    """
    Standard LHS: divide [0,1]^d into n equal strata per dimension,
    place one sample per stratum, then shuffle independently per dimension.
    """
    rng  = np.random.default_rng(seed)
    d    = len(bounds)
    cuts = np.linspace(0.0, 1.0, n + 1)          # stratum edges

    samples = np.empty((n, d))
    for j in range(d):
        lo, hi = bounds[j]
        # one uniform draw per stratum → map to [lo, hi]
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
# print("═" * 55)
# print(f"  LHS samples  (N={N})")
# print("═" * 55)
# print(df.to_string(index=False))
# print()

# 2. COMSOL strings  (one per parameter)
print("─── COMSOL value lists (paste into Parameter Value List) ───")
for col in param_names:
    cs = build_comsol_string(df[col].values)
    print(f"\n  {col}:\n  {cs}")
print()

# # 3. CSV export
# if SAVE_CSV:
#     df.to_csv(CSV_PATH, index=False)
#     print(f"  Samples saved → {CSV_PATH}")
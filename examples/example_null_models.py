# -*- coding: utf-8 -*-
"""Example: the three declination null models of astrocult.

Two synthetic study areas of 100 structures around 52.1N, each structure with
its own latitude and measured horizon altitude:
  A: 45 face sunrise on random days, 15 face sunrise near the equinoxes,
     40 have no astronomical intent
  B: no astronomical intent at all (random azimuths)

Each dataset is analysed with null_model='uniform' (previous behaviour),
'orientations' (new default) and 'solar'.
"""
import numpy as np
import matplotlib.pyplot as plt

from astrocult import Declination   # after `pip install astrocult` (or `pip install -e .` in the repository)

D = Declination.Declination()
rng = np.random.default_rng(2024)


def declination(altitude, latitude, azimuth):
    """Vectorized declination (degrees) from horizon altitude, latitude and azimuth."""
    h, p, a = np.radians(altitude), np.radians(latitude), np.radians(azimuth)
    return np.degrees(np.arcsin(np.sin(h) * np.sin(p) + np.cos(h) * np.cos(p) * np.cos(a)))


N = 100
latitudes = rng.uniform(51.8, 52.4, N)     # one latitude per structure
horizon = rng.uniform(0.0, 5.0, N)         # measured horizon altitude per structure
err = 1.0                                  # declination error per structure (deg)
binwidths = np.full(N, 2 * err)            # recommended binwidth: twice the error

n_sol, n_eq = 45, 15
decl_A = np.r_[D.solar_declination_approx(rng.uniform(0, 365.2422, n_sol), 23.56),
               rng.normal(0.0, 1.0, n_eq),
               declination(horizon[n_sol + n_eq:], latitudes[n_sol + n_eq:],
                           rng.uniform(0, 360, N - n_sol - n_eq))] + rng.normal(0, err, N)
decl_B = declination(horizon, latitudes, rng.uniform(0, 360, N)) + rng.normal(0, err, N)

fig, axs = plt.subplots(3, 2, figsize=(19, 11.5))
for col, (decl, name) in enumerate([(decl_A, "A"), (decl_B, "B")]):
    for row, model in enumerate(["uniform", "orientations", "solar"]):
        *_, info = D.plot_decl_adaptive(
            decl, latitudes, binwidths,
            null_model=model,              # 'orientations' is the default
            null_altitudes=horizon,        # measured horizons, paired with the latitudes
            iterations=300,
            null_iterations=1500,          # >= 1000 recommended for final results
            rng=np.random.default_rng(1),  # reproducible
            title=f"{name} — null '{model}'",
            ax=axs[row, col], show=False, xlim=(-60, 60))
        print(f"dataset {name}, null {model:12s} -> significant peaks at "
              f"{[round(p['angle'], 1) for p in info['peaks_significant']]}")
fig.tight_layout(h_pad=2.5, w_pad=2)
fig.savefig("example_null_models.png", dpi=100, bbox_inches="tight")
plt.show()

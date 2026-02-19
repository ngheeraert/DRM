"""
qfunction_plot.py
=================

Standalone plotting helper for Husimi-Q data produced by the master-equation runs.

Expected workflow
-----------------
1) A simulation produces a text file:
      data/qfunction_<filename>.d
   where the content is a 2D array representing Q(α) on a rectangular grid.

2) This script loads that array, displays it as an image over a phase-space extent
   α = x + i y with x,y in [-max_lambda, +max_lambda], and saves PDF figures:
      figures/qfunction_<filename>.pdf
      figures/LOG_qfunction_<filename>.pdf

Notes
-----
- This script assumes the array has already been computed on a square grid with
  symmetric limits, and that you know the corresponding `max_lambda`.
- For logarithmic plotting, a small `zmin` floor is used (LogNorm requires >0).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------
# `filename` should match the simulation tag used when saving Q-function outputs.
filename = "tmax1199_Nq2_Nc30_amp0.0060_kappa0.0014_wq4.4619_anh-0.1592_wc7.4150_wd7.4215_ms1e-03_dimexp120_qb[0,1]_11"
qfunction_arr = np.loadtxt('data/qfunction_' + filename + '.d')

# Phase-space plotting limit (extent of Re(α), Im(α) axes)
max_lambda = 9

# ---------------------------------------------------------------------------
# Linear-scale plot
# ---------------------------------------------------------------------------
interval = np.linspace(0.00, 1.0)
colors = plt.cm.magma(interval)
my_cmap = LinearSegmentedColormap.from_list('name', colors)

plt.imshow(
    qfunction_arr,
    extent=(-max_lambda, max_lambda, -max_lambda, max_lambda),
    cmap=my_cmap,
    vmin=0,
    vmax=0.5
)
plt.colorbar()
plt.savefig("figures/qfunction_" + filename + '.pdf', format='pdf')
plt.show()

# ---------------------------------------------------------------------------
# Log-scale plot (useful when Q spans orders of magnitude)
# ---------------------------------------------------------------------------
interval = np.linspace(0.00, qfunction_arr.max())
colors = plt.cm.magma(interval)
zmin, zmax = 0.005, qfunction_arr.max()
norm = LogNorm(vmin=zmin, vmax=zmax)

my_cmap = LinearSegmentedColormap.from_list('name', colors)
ext = (-max_lambda, max_lambda, -max_lambda, max_lambda)
plt.imshow(qfunction_arr, extent=ext, norm=norm, cmap=my_cmap)
plt.colorbar()
plt.savefig("figures/LOG_qfunction_" + filename + '.pdf', format='pdf')
plt.show()

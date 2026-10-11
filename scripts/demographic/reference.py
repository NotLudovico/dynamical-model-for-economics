"""The two stored economies at the paper's reference point, shared by the theory scripts.

alpha 2.5, sigma 1.70, D 0.1, N 16000, C 200 (code sign mu = 1.76), burn-in 100, window 480, samples
every 0.5.  The trajectories are those of multiscaling_theory.py (data/demographic/multiscaling/
powerlaw_seed<k>.npz, seeds 0 and 1); run that script first to create them.
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALPHA, SIGMA, D, N, C, MU = 2.5, 1.70, 0.1, 16000, 200, 1.76
START, TOBS, DT = 100.0, 480.0, 0.5
SEEDS = (0, 1)
TRAJ = os.path.join(ROOT, "data", "demographic", "multiscaling")
LABEL = rf"$\alpha={ALPHA}$, $\sigma={SIGMA:.2f}$, $D={D}$, $N={N}$"


def economy(seed, kind="powerlaw"):
    """Sizes S (N, T), death counters L, degrees, couplings a and the field Phi = 1 - aS - fbar.
    kind="regular" loads the ablation economy of multiscaling_theory.py (every degree = C)."""
    from relative_glv.model import configuration_coupling
    d = np.load(os.path.join(TRAJ, f"{kind}_seed{seed}.npz"))
    S, L, deg = d["S"].astype(float), d["L"], d["deg"]
    a = configuration_coupling(N, MU, SIGMA, alpha=ALPHA, mean_degree=C, seed=seed, regular=(kind == "regular"))
    assert np.array_equal(np.diff(a.indptr), deg), "coupling does not match the stored economy"
    aS = a @ S
    fbar = (S * (1.0 - S - aS)).sum(0) / N
    return S, L, deg, a, 1.0 - aS - fbar


def tidy_log_axes(ax, xticks=None, yticks=None):
    """Plain-number major ticks, no minor labels, light major grid, no top/right spines."""
    from matplotlib.ticker import NullFormatter, FormatStrFormatter
    for axis, ticks, scale in ((ax.xaxis, xticks, ax.get_xscale()), (ax.yaxis, yticks, ax.get_yscale())):
        if scale == "log":
            axis.set_minor_formatter(NullFormatter()); axis.set_major_formatter(FormatStrFormatter("%g"))
            if ticks is not None:
                axis.set_ticks(ticks)
    ax.grid(True, which="major", color="#e4e3df", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

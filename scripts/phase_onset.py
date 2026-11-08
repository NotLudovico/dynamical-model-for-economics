"""Direct test of the noisy instability line sigma_c(D) = 1.451 (static renewal closure, alpha 6, mu -1.76, D 0.1,
N 4000, C 100).  For each sigma and graph, two replicas with the same couplings but different initial sizes and
noise.  Order parameter: distance between their time-averaged size profiles over a window W,
d(W) = mean_i (Sbar_i^A - Sbar_i^B)^2 / var_i(Sbar_i).  A unique static state gives d ~ 1/W (W d constant);
a fluctuating or multi-attractor composition keeps d finite as W grows.  Run: uv run python scripts/phase_onset.py"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import numpy as np
from joblib import Parallel, delayed
from economy import DATA, powerlaw_coupling, integrate_demographic

SIG = np.array([1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.7])
WS = (125, 250, 500, 1000, 2000)
T0, TMAX = 200.0, 4200.0
PATH = DATA / "demographic" / "phase_onset.npz"


def run(sigma, graph, rep):
    a = powerlaw_coupling(4000, 1.76, sigma, alpha=6.0, mean_degree=100, seed=graph)
    return integrate_demographic(a, D=0.1, tmax=TMAX, t_start=T0, dt_save=1.0, seed=100 + 10 * graph + rep,
                                 compact=True)["S"]


def distances(A, B):
    """d_AB(W) over the last W samples, and d_AA(W): replica A against its own previous window."""
    dab, daa = [], []
    for W in WS:
        a, b, a0 = A[:, -W:].mean(1), B[:, -W:].mean(1), A[:, -2 * W:-W].mean(1)
        dab.append(((a - b) ** 2).mean() / a.var())
        daa.append(((a - a0) ** 2).mean() / a.var())
    return dab, daa


if __name__ == "__main__":
    if not PATH.exists():
        jobs = [(s, g, r) for s in SIG for g in (0, 1) for r in (0, 1)]
        out = Parallel(n_jobs=10, verbose=5)(delayed(run)(*j) for j in jobs)
        D = np.array([distances(out[i], out[i + 1]) for i in range(0, len(out), 2)]).reshape(SIG.size, 2, 2, len(WS))
        np.savez(PATH, sigma=SIG, W=WS, d_ab=D[:, :, 0], d_aa=D[:, :, 1])
    p = np.load(PATH)
    print("W:", p["W"], "  (unique static state: W*d flat in W; fluctuating: d flat)")
    for s, dab, daa in zip(p["sigma"], p["d_ab"].mean(1), p["d_aa"].mean(1)):
        print(f"sigma {s:.2f}  d_AB {np.round(dab, 4)}  W*d_AB {np.round(dab * p['W'], 1)}  d_AA {np.round(daa, 4)}")

"""HDMFT phase lines for a family of power-law graphs (N=4000, C=100, SCM kernel, 60 classes):
stability line sigma_c(mu) and zero-growth line mu0(sigma), lam = 0.  The fully connected limit
is sigma_c = sqrt(2) and mu0 = -g0(sigma).  Run: uv run python scripts/phase/phase_theory_lines.py
-> data/phase_theory_lines.npz"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from joblib import Parallel, delayed

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "phase_theory_lines.npz")
ALPHAS = (2.2, 2.5, 3.5, 6.0)
MUS = np.linspace(0.0, 2.5, 41)                 # code convention (competition > 0)
SIGMAS = np.linspace(0.4, 1.95, 32)
LAM = 0.0                                    # closed-form Gaussian integrals; lam = 1e-3 shifts sigma_c by ~+0.01


def classes(alpha):
    import relative_glv.model as M                # _ALPHA_PL is read at call time in this process
    from relative_glv.hdmft import bin_degree_distribution, soft_configuration_kernel
    M._ALPHA_PL = alpha
    a = M.coupling(4000, mu=0.0, sigma=1.0, kind="powerlaw", seed=0, mean_degree=100)
    cl = bin_degree_distribution(np.diff(a.indptr), n_bins=60)
    K = soft_configuration_kernel(cl["degree_fractions"], cl["weights"], cl["connectance"])["kernel"]
    return cl["weights"], K, cl["connectance"]


def sigma_c(alpha, mu, lam=LAM):
    from relative_glv.hdmft import heterogeneous_sigma_c
    w, K, c = classes(alpha)
    try:
        return heterogeneous_sigma_c(mu, w, K, c, lam=lam, bracket=(0.3, 3.0))["sigma_c"]
    except Exception:
        return np.nan


def mu0(alpha, sigma):
    from relative_glv.hdmft import heterogeneous_zero_growth
    w, K, c = classes(alpha)
    try:
        r = heterogeneous_zero_growth(sigma, w, K, c, lam=LAM, bracket=(0.0, 6.0))
        return -r["mu_c"], float(r["controlled"])
    except Exception:
        return np.nan, np.nan


if __name__ == "__main__":
    SC = np.array(Parallel(n_jobs=10)(delayed(sigma_c)(a, m) for a in ALPHAS for m in MUS)).reshape(len(ALPHAS), -1)
    Z = np.array(Parallel(n_jobs=10)(delayed(mu0)(a, s) for a in ALPHAS for s in SIGMAS)).reshape(len(ALPHAS), -1, 2)
    SC25 = np.array(Parallel(n_jobs=10)(delayed(sigma_c)(2.5, m, 1e-3) for m in MUS))   # simulated grid: lam = 1e-3
    np.savez(OUT, alphas=ALPHAS, mus=MUS, sigmas=SIGMAS, sigma_c=SC, mu0=Z[..., 0], controlled=Z[..., 1],
             sigma_c_a25_lam1e3=SC25)
    print("sigma_c(mu); mu = " + " ".join(f"{-m:5.2f}" for m in MUS))
    for a, row in zip(ALPHAS, SC):
        print(f"  alpha={a:3.1f}  " + " ".join(f"{x:5.3f}" for x in row))
    print("zero-growth mu0(sigma) [* = stable fixed point]; sigma = " + " ".join(f"{s:5.2f}" for s in SIGMAS))
    for a, row in zip(ALPHAS, Z):
        print(f"  alpha={a:3.1f}  " + " ".join(f"{x:5.2f}{'*' if c == 1 else ' '}" for x, c in row))

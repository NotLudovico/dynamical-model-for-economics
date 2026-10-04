"""Does the HDMFT underestimate of the chaos onset shrink with N?  mu = -1.97 (strong competition),
power law 2.5, C = N/40, lam = 1e-3, N = 2000, 4000, 8000, three graphs each.  For every graph its own
HDMFT sigma_c (SCM kernel, 60 classes) is computed, then sigma is scanned from sigma_c to sigma_c + 0.1.
Amplitude = median time std of ln S over firms listed throughout (S > 0.1) in the window [600, 800].
Run: uv run python scripts/phase_onset_N.py  -> data/phase_onset_N.npz"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

MU, LAM, TMAX, WIN = 1.97, 1e-3, 800.0, (600.0, 800.0)      # code convention (competition > 0)
NS, SEEDS, DS = (2000, 4000, 8000), (0, 1, 2), np.arange(0.0, 0.1001, 0.0125)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "phase_onset_N.npz")


def sigma_c(N, seed):
    from relative_glv.model import coupling
    from relative_glv.hdmft import bin_degree_distribution, soft_configuration_kernel, heterogeneous_sigma_c
    a = coupling(N, mu=0.0, sigma=1.0, kind="powerlaw", seed=seed, mean_degree=N // 40)
    cl = bin_degree_distribution(np.diff(a.indptr), n_bins=60)
    K = soft_configuration_kernel(cl["degree_fractions"], cl["weights"], cl["connectance"])["kernel"]
    return heterogeneous_sigma_c(MU, cl["weights"], K, cl["connectance"], lam=LAM, bracket=(1.2, 1.9))["sigma_c"]


def amplitude(N, seed, sigma):
    from relative_glv.model import coupling
    a = coupling(N, MU, sigma, kind="powerlaw", seed=seed, mean_degree=N // 40)
    rng = np.random.default_rng(seed); w0 = rng.uniform(0.5, 1.5, N); w0 /= w0.sum()
    def rhs(t, s):
        w = np.clip(s, 0.0, None); w = w / w.sum()
        f = 1.0 - N * w - N * (a @ w)
        return w * (f - w @ f) + LAM * (1.0 / N - w)
    r = solve_ivp(rhs, (0.0, TMAX), w0, method="RK45", t_eval=np.arange(WIN[0], WIN[1] + 1e-9, 0.5),
                  rtol=1e-4, atol=1e-7)
    if not r.success:
        return np.nan
    lnS = np.log(np.clip(r.y * N, 1e-2, None))
    big = (lnS > np.log(0.1)).all(1)
    return float(np.median(lnS[big].std(1))) if big.any() else np.nan


if __name__ == "__main__":
    keys = [(N, s) for N in NS for s in SEEDS]
    SC = dict(zip(keys, Parallel(n_jobs=9)(delayed(sigma_c)(N, s) for N, s in keys)))
    print("HDMFT sigma_c per graph:", {k: round(v, 4) for k, v in SC.items()}, flush=True)
    jobs = [(N, s, SC[(N, s)] + d) for N, s in keys for d in DS]
    A = np.array(Parallel(n_jobs=10, verbose=2)(delayed(amplitude)(*j) for j in jobs)).reshape(len(NS), len(SEEDS), len(DS))
    np.savez(OUT, Ns=NS, seeds=SEEDS, ds=DS, sigma_c=np.array([[SC[(N, s)] for s in SEEDS] for N in NS]), amp=A)
    print(f"amplitude vs sigma - sigma_c  ({' '.join(f'{d:+.4f}' for d in DS)})")
    for i, N in enumerate(NS):
        for j, s in enumerate(SEEDS):
            print(f"  N={N:5d} seed {s}  sigma_c {SC[(N, s)]:.4f} | " + " ".join(f"{x:.4f}" for x in A[i, j]))

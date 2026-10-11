"""Above the HDMFT line: trapping, slow relaxation or chaos?  Power law 2.5, N = 4000, C = 100,
lam = 1e-3, ten graphs per point, each with its own HDMFT sigma_c (SCM, 60 classes).  Amplitude
(median time std of ln S over firms listed throughout, S > 0.1) in two windows, [600, 800] and
[1400, 1600]: frozen in both = trapped (or unique), decaying = slow relaxation, persisting = chaos.
Run: uv run python scripts/phase/phase_coexistence.py  -> data/phase_coexistence.npz"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

N, C, LAM = 4000, 100, 1e-3
MUS, SEEDS, DS = (1.97, 1.05), range(10), (0.0, 0.025, 0.05, 0.1, 0.2)   # code convention (competition > 0)
W1, W2 = (600.0, 800.0), (1400.0, 1600.0)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "phase_coexistence.npz")


def sigma_c(mu, seed):
    from relative_glv.model import coupling
    from relative_glv.hdmft import bin_degree_distribution, soft_configuration_kernel, heterogeneous_sigma_c
    a = coupling(N, mu=0.0, sigma=1.0, kind="powerlaw", seed=seed, mean_degree=C)
    cl = bin_degree_distribution(np.diff(a.indptr), n_bins=60)
    K = soft_configuration_kernel(cl["degree_fractions"], cl["weights"], cl["connectance"])["kernel"]
    return heterogeneous_sigma_c(mu, cl["weights"], K, cl["connectance"], lam=LAM, bracket=(0.8, 2.0))["sigma_c"]


def amp(lnS):
    big = (lnS > np.log(0.1)).all(1)
    return float(np.median(lnS[big].std(1))) if big.any() else np.nan


def run(mu, seed, sigma):
    from relative_glv.model import coupling
    a = coupling(N, mu, sigma, kind="powerlaw", seed=seed, mean_degree=C)
    rng = np.random.default_rng(seed); w0 = rng.uniform(0.5, 1.5, N); w0 /= w0.sum()
    def rhs(t, s):
        w = np.clip(s, 0.0, None); w = w / w.sum()
        f = 1.0 - N * w - N * (a @ w)
        return w * (f - w @ f) + LAM * (1.0 / N - w)
    te = np.concatenate([np.arange(*W1, 0.5), np.arange(*W2, 0.5)])
    r = solve_ivp(rhs, (0.0, W2[1]), w0, method="RK45", t_eval=te, rtol=1e-4, atol=1e-7)
    if not r.success:
        return np.nan, np.nan
    lnS = np.log(np.clip(r.y * N, 1e-2, None))
    k = np.searchsorted(r.t, W2[0])
    return amp(lnS[:, :k]), amp(lnS[:, k:])


if __name__ == "__main__":
    keys = [(m, s) for m in MUS for s in SEEDS]
    SC = dict(zip(keys, Parallel(n_jobs=10)(delayed(sigma_c)(m, s) for m, s in keys)))
    jobs = [(m, s, SC[(m, s)] + d) for m, s in keys for d in DS]
    R = np.array(Parallel(n_jobs=10, verbose=2)(delayed(run)(*j) for j in jobs)).reshape(len(MUS), len(SEEDS), len(DS), 2)
    np.savez(OUT, mus=MUS, ds=DS, sigma_c=np.array([[SC[(m, s)] for s in SEEDS] for m in MUS]), amp=R)
    for i, m in enumerate(MUS):
        print(f"mu={-m:.2f}: per sigma - sigma_c, counts of [frozen both | decaying | fluctuating at T=1600] out of 10")
        for j, d in enumerate(DS):
            a1, a2 = R[i, :, j, 0], R[i, :, j, 1]
            frozen = (a1 < 1e-3) & (a2 < 1e-3); fl = a2 >= 1e-2; dec = ~frozen & ~fl
            print(f"   +{d:.3f}: {frozen.sum():2d} | {dec.sum():2d} | {fl.sum():2d}   median amp T=800 {np.nanmedian(a1):.4f}, T=1600 {np.nanmedian(a2):.4f}")

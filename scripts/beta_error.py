"""Error bar on the own-degree size-variance exponent beta.

Measures beta per economy at the yearly protocol (dt=0.5 decline-branch, the SAME
size_volatility estimator behind growing_stationary_msb.png) over many disorder
realizations, and reports the economy-to-economy spread. That spread IS the honest
measurement error: the empirical beta~0.2 is one real economy, so the comparable
model uncertainty is how much beta varies economy to economy. No r^2 gate (gating
biases beta steep -- memory).

    uv run python growing_glv/beta_error.py [--smoke]
"""
import sys, os
import numpy as np
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate
from relative_glv.msb import size_volatility

SMOKE = "--smoke" in sys.argv
MU, SIGMA, LAM = 1.76, 1.75, 1e-3          # locked point (chapter mu = -1.76)
N, SEEDS = (800, 4) if SMOKE else (6400, 24)
TMAX, N_EVAL = (120.0, 240) if SMOKE else (400.0, 800)
LATE, DT = ((95.0, 118.0) if SMOKE else (320.0, 390.0)), 0.5


def beta_one(seed):
    a = coupling(N, MU, SIGMA, kind="powerlaw_owndeg", seed=seed)
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                  method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT, n_bins=20)
    g = sv["growth"]
    # persistent (fluctuating) economies only: frozen ones have ~zero growth spread
    if g.size == 0 or not np.isfinite(sv["beta"]) or g.std() < 0.02:
        return None
    return dict(seed=seed, beta=float(sv["beta"]), r2=float(sv["r2"]))


if __name__ == "__main__":
    recs = [r for r in Parallel(n_jobs=6, backend="loky")(
        delayed(beta_one)(s) for s in range(SEEDS)) if r is not None]
    b = np.array([r["beta"] for r in recs])
    r2 = np.array([r["r2"] for r in recs])
    med = float(np.median(b)); mad = float(np.median(np.abs(b - med)))
    mean, std = float(b.mean()), float(b.std(ddof=1))
    q1, q3 = np.percentile(b, [25, 75])
    sem = std / np.sqrt(b.size)
    print(f"[beta_error] N={N}  persistent economies={b.size}/{SEEDS}")
    print(f"  betas: {np.round(np.sort(b), 2)}")
    print(f"  median = {med:.2f}   MAD = {mad:.2f}   IQR = [{q1:.2f}, {q3:.2f}]")
    print(f"  mean   = {mean:.2f} +/- {std:.2f} (sd)   SEM = {sem:.2f}")
    print(f"  1.4826*MAD (sd-equiv) = {1.4826*mad:.2f}   r2 median = {np.median(r2):.2f}")
    print(f"  => gap vs empirical 0.2:  {med/0.2:.1f}x  (range {(med-mad)/0.2:.1f}-{(med+mad)/0.2:.1f}x)")
    np.savez(os.path.join(os.path.dirname(__file__), "beta_error.npz"),
             betas=b, r2=r2, N=N, median=med, mad=mad, mean=mean, std=std)

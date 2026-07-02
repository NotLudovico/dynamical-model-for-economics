"""Stronger N->infinity limit of the size-variance exponent beta for the relative GLV.

The size-variance law sigma(S) ~ S^-beta is the second MSB stylized fact; its
exponent has a finite-size bias, and only the N->infinity value is comparable to
the empirical band (~0.15-0.20). This script measures that limit robustly while
keeping every economy a SEPARATE realization (no pooling of firms across seeds:
each seed is its own interaction network, with its own connected-component
structure and disorder draw; mixing their firms would fabricate one unphysical
mega-economy).

Method:
  - Per system size N, simulate many seeds at the locked regime (sigma=1.75,
    mu=1.76, lam=1e-3). beta is measured PER ECONOMY with the repo estimator
    (msb.size_volatility: decline-branch power-law fit of MAD-volatility vs mean
    size). The fit r2 is REPORTED per economy as a power-law-quality check, it is
    NOT used to discard economies (the old r2>0.9 gate starved the large-N points).
  - Aggregate ACROSS REALIZATIONS: the typical exponent at size N is the median of
    the per-economy betas; its uncertainty is a bootstrap over economies (resample
    the set of per-economy betas with replacement, re-take the median).
  - Extrapolate median-beta(N) -> beta_inf with three finite-size forms,
        beta(N) = beta_inf + c * f(N),   f in {1/sqrt(N), 1/N, N^(-1/3)},
    reporting beta_inf and the fit residual for each form (robustness across forms),
    and a bootstrap 95% CI on beta_inf (resample economies within every N, recompute
    the median-beta curve, refit).

Outputs data/beta_limit.npz (compute once, replot freely) and prints a summary.

    uv run python scripts/beta_limit.py [--smoke]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate
from relative_glv.msb import size_volatility

SMOKE = "--smoke" in sys.argv
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SIGMA, MU, LAM = 1.75, 1.76, 1e-3
TMAX, N_EVAL = 400.0, 800            # dt=0.5 across [0, 400]
LATE = (320.0, 390.0)
DT = 0.5
N_BINS = 20                          # matches the repo's size_volatility default
RTOL, ATOL = 1e-4, 1e-7
N_JOBS = 6
MEM_CAP = 1.2e9                      # cap transient memory (jobs * one trajectory array) ~1.2 GB
N_BOOT = 2000


def jobs_for(N):
    """Parallel workers for size N, bounding peak memory (each worker holds one (N, N_EVAL) array)."""
    per = N * N_EVAL * 8                              # bytes for one trajectory array W
    return int(np.clip(MEM_CAP // per, 2, N_JOBS))

# --- the scan grid (finalised from the timing probe) ---
NSCAN = [800, 1600, 3200, 6400, 12800, 25600, 51200]
SEEDS = 32

if SMOKE:
    NSCAN = [400, 800]
    SEEDS = 4
    TMAX, N_EVAL = 120.0, 240
    LATE = (60.0, 110.0)
    N_BOOT = 200

FORMS = {
    "1/sqrt(N)": lambda N: 1.0 / np.sqrt(np.asarray(N, float)),
    "1/N": lambda N: 1.0 / np.asarray(N, float),
    "N^(-1/3)": lambda N: np.asarray(N, float) ** (-1.0 / 3.0),
}


def economy_beta(N, seed):
    """One economy (one network realization) -> (beta, r2). (nan, nan) on failure."""
    a = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed)
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                  method="RK45", rtol=RTOL, atol=ATOL)
    if not r["success"]:
        return np.nan, np.nan
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT, n_bins=N_BINS)
    return float(sv["beta"]), float(sv["r2"])


def boot_median(vals, rng, n_boot):
    """Bootstrap 95% CI of the median of a 1-D sample (resample economies w/ replacement)."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan, np.nan
    meds = np.median(vals[rng.integers(0, vals.size, (n_boot, vals.size))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fit_betainf(Ns, betas):
    """beta(N) = beta_inf + c f(N) per finite-size form. -> {form: (beta_inf, c, rmse)}."""
    Ns, betas = np.asarray(Ns, float), np.asarray(betas, float)
    ok = np.isfinite(betas)
    out = {}
    for name, f in FORMS.items():
        x = f(Ns[ok])
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, betas[ok], rcond=None)
        resid = betas[ok] - A @ coef
        out[name] = (float(coef[0]), float(coef[1]), float(np.sqrt(np.mean(resid ** 2))))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(20260622)
    t0 = time.time()
    print(f"[beta_limit] smoke={SMOKE}  regime sigma={SIGMA} mu={MU}  "
          f"N={NSCAN}  seeds={SEEDS}  jobs={N_JOBS}  (per-economy, no pooling)", flush=True)

    # 1) per-economy betas, all economies kept (no r2 gate)
    betas_by_N, r2_by_N = {}, {}
    for N in NSCAN:
        res = Parallel(n_jobs=jobs_for(N), backend="loky")(
            delayed(economy_beta)(N, s) for s in range(SEEDS))
        b = np.array([x[0] for x in res]); q = np.array([x[1] for x in res])
        betas_by_N[N] = b; r2_by_N[N] = q
        nfin = int(np.isfinite(b).sum())
        print(f"  N={N:>6}: economies={nfin:>2}/{SEEDS}  median beta={np.nanmedian(b):+.3f}  "
              f"median r2={np.nanmedian(q):.3f}  spread[{np.nanpercentile(b,25):+.2f},"
              f"{np.nanpercentile(b,75):+.2f}]  ({(time.time()-t0)/60:.1f} min)", flush=True)

    Ns = np.array(NSCAN, float)
    beta_med = np.array([np.nanmedian(betas_by_N[N]) for N in NSCAN])
    r2_med = np.array([np.nanmedian(r2_by_N[N]) for N in NSCAN])
    beta_ci = np.array([boot_median(betas_by_N[N], rng, N_BOOT) for N in NSCAN])  # (len, 2)

    # 2) extrapolation per form + bootstrap CI on beta_inf (resample economies within each N)
    fit_pt = fit_betainf(Ns, beta_med)
    boot_inf = {name: np.full(N_BOOT, np.nan) for name in FORMS}
    for k in range(N_BOOT):
        med_k = []
        for N in NSCAN:
            v = betas_by_N[N][np.isfinite(betas_by_N[N])]
            med_k.append(np.median(v[rng.integers(0, v.size, v.size)]) if v.size else np.nan)
        for name, (binf, _c, _r) in fit_betainf(Ns, med_k).items():
            boot_inf[name][k] = binf

    print("\n[beta_limit] median beta(N) across economies (95% bootstrap CI):")
    for j, N in enumerate(NSCAN):
        print(f"  N={int(N):>6}: beta={beta_med[j]:+.3f}  [{beta_ci[j,0]:+.3f}, {beta_ci[j,1]:+.3f}]  "
              f"median r2={r2_med[j]:.3f}  ({int(np.isfinite(betas_by_N[N]).sum())} economies)")
    print("\n[beta_limit] beta_inf by finite-size form:")
    for name in FORMS:
        binf, c, rmse = fit_pt[name]
        lo, hi = np.nanpercentile(boot_inf[name], [2.5, 97.5])
        print(f"  {name:>10}:  beta_inf={binf:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  (c={c:+.2f}, RMSE={rmse:.3f})")

    out = dict(
        N=Ns, seeds=SEEDS, n_boot=N_BOOT, n_bins=N_BINS,
        betas=np.array([betas_by_N[N] for N in NSCAN]),     # (nN, seeds): every economy's beta
        r2s=np.array([r2_by_N[N] for N in NSCAN]),
        beta_med=beta_med, beta_lo=beta_ci[:, 0], beta_hi=beta_ci[:, 1], r2_med=r2_med,
        n_econ=np.array([int(np.isfinite(betas_by_N[N]).sum()) for N in NSCAN]),
        forms=np.array(list(FORMS), dtype=object),
        betainf=np.array([fit_pt[n][0] for n in FORMS]),
        betainf_c=np.array([fit_pt[n][1] for n in FORMS]),
        betainf_rmse=np.array([fit_pt[n][2] for n in FORMS]),
        betainf_lo=np.array([np.nanpercentile(boot_inf[n], 2.5) for n in FORMS]),
        betainf_hi=np.array([np.nanpercentile(boot_inf[n], 97.5) for n in FORMS]),
    )
    np.savez(os.path.join(DATA, "beta_limit.npz"), **out)
    print(f"\n[beta_limit] saved -> data/beta_limit.npz  ({(time.time()-t0)/60:.1f} min)", flush=True)

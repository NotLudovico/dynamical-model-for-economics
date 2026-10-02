"""Long windows (T_obs up to 1920) for alpha = 2.2 and 6: spell estimator (>=2, >=20) and the
all-firm per-firm estimator on the same trajectories. Does beta converge?
Run: uv run python scripts/lifecycle/long_windows.py   (~40 min)
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import time
import numpy as np
from joblib import Parallel, delayed
from beta_spells import model, cq_one, MU, SIGMA, LAM, N, C, START, DT, SEEDS
from relative_glv.msb import spell_volatility, _decline_beta

TOBS = (240.0, 480.0, 960.0, 1920.0)
NAN4 = np.full(4, np.nan)

def economy(alpha, seed):
    model._ALPHA_PL = alpha
    a = model.coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)
    tmax = START + max(TOBS)
    r = model.integrate(a, tmax=tmax, n_eval=int(tmax / DT) + 1, lam=LAM, seed=seed,
                        method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    t, W, out = r["t"], r["W"].astype(np.float32), {}
    del r
    for T in TOBS:
        win = (START, START + T)
        row = []
        for m in (2, 20):
            sp = spell_volatility(W, t, window=win, dt=DT, min_growth=m)
            z = cq_one(sp["Sbar"], sp["vol"])
            row += [sp["beta"], *(NAN4 if z is None else z), sp["censored"].mean()]
        lnS = np.log(np.maximum(N * W[:, (t >= win[0] - 1e-9) & (t <= win[1] + 1e-9)].astype(float), 1e-300))
        g = np.diff(lnS, axis=1)
        Sb, v = np.exp(lnS).mean(1), np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
        del lnS, g
        z = cq_one(Sb, v)
        row += [_decline_beta(Sb, v)[0], *(NAN4 if z is None else z), np.nan]
        out[T] = row
    return out

t0 = time.time()
for alpha in (2.2, 6.0):
    R = [x for x in Parallel(n_jobs=4)(delayed(economy)(alpha, s) for s in range(SEEDS)) if x]
    np.save(f"{DATA}/long_alpha{alpha}.npy", np.array([[x[T] for T in TOBS] for x in R]))
    for T in TOBS:
        X = np.array([x[T] for x in R])
        for k, name in enumerate(("spells>=2 ", "spells>=20", "all-firm  ")):
            b, z, cens = X[:, 6 * k], X[:, 6 * k + 1:6 * k + 5], X[:, 6 * k + 5]
            z = z[np.isfinite(z).all(1)]
            zr = z.mean(0) / z.mean(0)[0] if len(z) else NAN4
            print(f"alpha={alpha} T_obs={T:5.0f} {name} beta {np.nanmedian(b):.3f}±{np.nanstd(b) / np.sqrt(len(b)):.3f}"
                  f" | zeta_q/zeta_1 {zr[1]:.2f} {zr[2]:.2f} {zr[3]:.2f} | cens {np.nanmedian(cens):.2f}", flush=True)
    print(f"  alpha={alpha} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

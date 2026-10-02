"""Exact fitness Phi_i(t) = 1 - g(t) - (alpha S(t))_i of every firm in the full model, on the Delta grid.
Saves per-firm mean m_i, fluctuation std f_i, degree k_i, the mean ACF of the fluctuation, and the
spell-estimator output of the full model (reference for the single-firm surrogate)."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from joblib import Parallel, delayed
from beta_spells import model, cq_one, MU, N, C, START, DT
from relative_glv.msb import spell_volatility

T = 480.0
ALPHA, SIGMA, LAM = (float(x) for x in sys.argv[1:4])
LAGS = 61

def economy(seed):
    model._ALPHA_PL = ALPHA
    a = model.coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)
    r = model.integrate(a, tmax=START + T, n_eval=int((START + T) / DT) + 1, lam=LAM, seed=seed,
                        method="RK45", rtol=1e-4, atol=1e-7)
    t, W = r["t"], r["W"]
    win = t >= START - 1e-9
    S = N * W[:, win]
    AS = np.asarray(a @ S)                                   # (alpha S)_i
    g = (W[:, win] * (1 - S - AS)).sum(0)                    # <f>, exact aggregate growth
    Phi = 1 - g[None, :] - AS
    m, f = Phi.mean(1), Phi.std(1)
    d = (Phi - m[:, None]) / np.maximum(f[:, None], 1e-12)
    acf = np.array([np.mean(d[:, :d.shape[1] - L] * d[:, L:]) for L in range(LAGS)])
    sp = spell_volatility(W, t, window=(START, START + T), dt=DT, floor=8 * LAM / N)
    z = cq_one(sp["Sbar"], sp["vol"])
    listed = S > 8 * LAM
    Sbar_listed = np.array([s[l].mean() if l.any() else np.nan for s, l in zip(S, listed)])
    mbar_listed = np.array([p[l].mean() if l.any() else np.nan for p, l in zip(Phi, listed)])
    return dict(m=m, f=f, k=np.diff(a.indptr), acf=acf, beta=sp["beta"], z=z, frac_listed=listed.mean(1),
                Sbar_listed=Sbar_listed, mbar_listed=mbar_listed, Phi_sample=Phi[:200])

R = Parallel(n_jobs=8)(delayed(economy)(s) for s in range(8))
m = np.concatenate([x["m"] for x in R]); f = np.concatenate([x["f"] for x in R])
acf = np.mean([x["acf"] for x in R], 0)
Z = np.array([x["z"] for x in R if x["z"] is not None])
np.savez(f"{DATA}/fit_a{ALPHA}_s{SIGMA}_l{LAM:.0e}.npz", m=m, f=f, k=np.concatenate([x["k"] for x in R]), acf=acf,
         beta=np.array([x["beta"] for x in R]), Z=Z, frac_listed=np.concatenate([x["frac_listed"] for x in R]),
         Sbar_listed=np.concatenate([x["Sbar_listed"] for x in R]),
         mbar_listed=np.concatenate([x["mbar_listed"] for x in R]), Phi_sample=R[0]["Phi_sample"])
tau_e = DT * np.argmax(acf < np.exp(-1))
print(f"alpha={ALPHA} sigma={SIGMA} lam={LAM:.0e}")
print(f"  mean fitness m: mean {m.mean():+.3f} sd {m.std():.3f} skew {((m - m.mean())**3).mean() / m.std()**3:+.2f}"
      f" exkurt {((m - m.mean())**4).mean() / m.std()**4 - 3:+.2f}")
print(f"  fluct amplitude f: mean {f.mean():.3f} sd {f.std():.3f}, corr(m,f) {np.corrcoef(m, f)[0, 1]:+.2f}")
print(f"  ACF at lag 0.5,1,2,4,8,16: " + " ".join(f"{acf[int(L / DT)]:.3f}" for L in (0.5, 1, 2, 4, 8, 16)) +
      f"   1/e time {tau_e:.1f}")
print(f"  full model spells: beta {np.median([x['beta'] for x in R]):.3f}, ratios "
      + " ".join(f"{v:.2f}" for v in Z.mean(0)[1:] / Z.mean(0)[0]))
ok = np.isfinite(np.concatenate([x['Sbar_listed'] for x in R]))
sl, ml = np.concatenate([x['Sbar_listed'] for x in R])[ok], np.concatenate([x['mbar_listed'] for x in R])[ok]
big = sl > 1
print(f"  listed firms, S>1: median Sbar/mean-fitness-while-listed {np.median(sl[big] / ml[big]):.3f}; "
      f"fraction of time listed: median {np.median(np.concatenate([x['frac_listed'] for x in R])):.2f}")

"""Per firm: spell rate and mean complete-spell length vs u = m/f, against Rice for Phi = m + f eta crossing 0:
upcrossing rate nu(u) = exp(-u^2/2) / (2 pi tau), mean excursion length Lbar(u) = Phi_N(u) / nu(u).
Also: time-mean size while listed vs m (size = mean fitness) and listed fraction vs Phi_N(u)."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from scipy.stats import norm
from joblib import Parallel, delayed
from surrogate import run, fit_tau, NE, DT

def one(m, f, tau, lam, seed):
    S = run(m, f, tau, lam, seed)
    on = S > 8 * lam
    d = np.diff(np.pad(on.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    out = []
    for i in range(S.shape[0]):
        a, b = np.where(d[i] == 1)[0], np.where(d[i] == -1)[0]
        comp = (a > 0) & (b < S.shape[1])
        out.append((m[i] / f[i], len(a) / (S.shape[1] * DT), (b - a)[comp].mean() * DT if comp.any() else np.nan,
                    on[i].mean(), S[i][on[i]].mean() if on[i].any() else np.nan, m[i]))
    return np.array(out)

if __name__ == "__main__":
    d = np.load(sys.argv[1]); tau = fit_tau(d["acf"]); lam = float(sys.argv[1].rsplit("_l", 1)[1][:-4])
    R = np.vstack(Parallel(n_jobs=8)(delayed(one)(d["m"][e * NE:(e + 1) * NE], d["f"][e * NE:(e + 1) * NE], tau, lam, 100 + e)
                                     for e in range(8)))
    u, rate, L, frac, Sl, m = R.T
    print(f"{sys.argv[1]}  tau={tau:.2f}")
    print(f"{'u':>5} {'firms':>6} | {'spells/t':>8} {'Rice nu':>8} | {'mean L':>7} {'Rice L':>7} | {'listed':>6} {'Phi(u)':>6} | {'S|listed':>8} {'m':>6}")
    for lo in np.arange(-1.5, 3.01, 0.5):
        s = (u >= lo) & (u < lo + 0.5)
        uc = lo + 0.25
        nu = np.exp(-uc ** 2 / 2) / (2 * np.pi * tau)
        print(f"{uc:5.2f} {s.sum():6d} | {np.mean(rate[s]):8.4f} {nu:8.4f} | {np.nanmedian(L[s]):7.1f} {norm.cdf(uc) / nu:7.1f} | "
              f"{np.mean(frac[s]):6.2f} {norm.cdf(uc):6.2f} | {np.nanmedian(Sl[s]):8.2f} {np.median(m[s]):6.2f}")

"""Lifetime dispersion vs size (complete spells, decline-branch bins): Var(ln n | S), the second-cumulant
prediction of the lifetime bend, zeta_q(1/n) - q zeta_1(1/n) ~ -(q(q-1)/2) dVar(ln n)/dlnS, and the shape
of the length distribution (exponential => Var ln n = pi^2/6 = 1.64)."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from joblib import Parallel, delayed
from surrogate import run, fit_tau, NE
from anatomy import spells, branch

def one(m, f, tau, lam, seed):
    sp = spells(run(m, f, tau, lam, seed), 8 * lam)
    Sb, v, n, comp, Smax, TV = sp.T
    P = [p[comp[p].astype(bool)] for p in branch(Sb, v)]
    lx = np.log([Sb[p].mean() for p in P])
    V = np.array([np.var(np.log(n[p])) for p in P])
    z = np.array([-np.polyfit(lx, np.log([np.mean(n[p] ** -q) for p in P]), 1)[0] for q in (1, 2, 3, 4)])
    mean_n = np.array([n[p].mean() for p in P]) * 0.5
    return V, np.polyfit(lx, V, 1)[0], z, mean_n

if __name__ == "__main__":
    d = np.load(sys.argv[1]); tau = fit_tau(d["acf"]); lam = float(sys.argv[1].rsplit("_l", 1)[1][:-4])
    R = Parallel(n_jobs=8)(delayed(one)(d["m"][e * NE:(e + 1) * NE], d["f"][e * NE:(e + 1) * NE], tau, lam, 100 + e)
                           for e in range(8))
    nb = min(len(x[0]) for x in R)
    V = np.mean([x[0][:nb] for x in R], 0); dV = np.mean([x[1] for x in R]); z = np.mean([x[2] for x in R], 0)
    L = np.mean([x[3][:nb] for x in R], 0)
    print(f"{sys.argv[1]}")
    print("  Var(ln n | S) across decline bins: " + " ".join(f"{a:.2f}" for a in V[::max(1, nb // 8)]) + f"  (exp: 1.64)")
    print("  mean lifetime (time units) across bins: " + " ".join(f"{a:.1f}" for a in L[::max(1, nb // 8)]))
    q = np.arange(1, 5)
    print(f"  dVar/dlnS {dV:.3f}; lifetime bend zeta_q - q zeta_1: measured " +
          " ".join(f"{a:+.3f}" for a in z - q * z[0]) + " | 2nd-cumulant prediction " +
          " ".join(f"{a:+.3f}" for a in -q * (q - 1) / 2 * dV))

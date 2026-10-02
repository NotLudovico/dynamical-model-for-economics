"""Lifetime representation of the spell exponents in the surrogate (complete spells, decline branch):
zeta_q(sigma) vs zeta_q(1/n) - q dln(ell)/dlnS, ell = ln(S_max/thr); short-spell density exponent gamma0."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from joblib import Parallel, delayed
from surrogate import run, fit_tau, NE
from anatomy import spells, branch

Q = np.arange(1, 5)

def slope(S, y, P):
    return -np.polyfit(np.log([S[p].mean() for p in P]), np.log([max(np.mean(y[p]), 1e-12) for p in P]), 1)[0]

def one(m, f, tau, lam, seed):
    thr = 8 * lam
    sp = spells(run(m, f, tau, lam, seed), thr)
    Sb, v, n, comp, Smax, TV = sp.T
    P = [p[comp[p].astype(bool)] for p in branch(Sb, v)]
    ell = np.log(Smax / thr)
    tr = np.sqrt(np.pi / 2) * 2 * ell / n
    return dict(z_v=[slope(Sb, v ** q, P) for q in Q], z_tr=[slope(Sb, tr ** q, P) for q in Q],
                z_n=[slope(Sb, n ** -q, P) for q in Q], dl=-slope(Sb, ell, P),
                ell=np.median(ell[np.concatenate(P)]),
                g0=[slope(Sb, (n <= n0).astype(float), P) for n0 in (4, 8)])

if __name__ == "__main__":
    d = np.load(sys.argv[1]); tau = fit_tau(d["acf"])
    for lam in (1e-4, 1e-3, 1e-2):
        R = Parallel(n_jobs=8)(delayed(one)(d["m"][e * NE:(e + 1) * NE], d["f"][e * NE:(e + 1) * NE], tau, lam, 100 + e)
                               for e in range(8))
        r = lambda k: np.mean([x[k] for x in R], 0)
        zv, ztr, zn, dl = r("z_v"), r("z_tr"), r("z_n"), r("dl")
        pred = zn - Q * dl
        f2 = lambda z: " ".join(f"{a:+.3f}" for a in z)
        print(f"lam={lam:.0e}  median ell {r('ell'):.2f}, dln ell/dlnS {dl:.3f}")
        print(f"   zeta_q sigma (complete)     {f2(zv)}   ratios " + " ".join(f"{a:.2f}" for a in zv[1:] / zv[0]))
        print(f"   zeta_q transient            {f2(ztr)}")
        print(f"   zeta_q(1/n) - q dln ell     {f2(pred)}")
        print(f"   lifetime zeta_q(1/n)        {f2(zn)}   short-spell density exponent (n<=4, n<=8): "
              + " ".join(f"{a:.3f}" for a in r("g0")), flush=True)

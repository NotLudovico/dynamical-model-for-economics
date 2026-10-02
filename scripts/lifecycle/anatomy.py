"""Spell anatomy in the single-firm surrogate. Exact for complete spells (mean growth 0):
sigma = sqrt(pi/2) TV / n,  TV = 2 ln(S_max/thr) + V_p  (rise + fall + extra plateau variation).
Splits zeta_q into the transient part (rise/fall) and the plateau part; spell length vs size."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from joblib import Parallel, delayed
from surrogate import run, fit_tau, NE, DT, START
from beta_spells import cq_one

def spells(S, thr, min_growth=2):
    """Per spell: Sbar, sigma (MAD), n growth steps, complete?, S_max, TV."""
    out = []
    lnS = np.log(S)
    for x in lnS:
        on = np.concatenate(([0], (x > np.log(thr)).astype(np.int8), [0]))
        d = np.diff(on)
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            n = b - a - 1
            if n >= min_growth:
                g = np.diff(x[a:b])
                out.append((np.exp(x[a:b]).mean(), np.sqrt(np.pi / 2) * np.abs(g - g.mean()).mean(), n,
                            a > 0 and b < x.size, np.exp(x[a:b].max()), np.abs(g).sum()))
    return np.array(out)

def branch(S, v, nb=20):
    o = np.argsort(S); P = np.array_split(o, nb)
    bx = np.array([S[p].mean() for p in P]); pk = int(np.argmax([np.median(v[p]) for p in P]))
    dec = np.where(S > bx[pk])[0]; dec = dec[np.argsort(S[dec])]
    return np.array_split(dec, max(6, nb - pk))

def zq(S, y, P):
    lx = np.log([S[p].mean() for p in P])
    return np.array([-np.polyfit(lx, np.log([np.mean(y[p] ** q) for p in P]), 1)[0] for q in (1, 2, 3, 4)])

def one(m, f, tau, lam, seed):
    S = run(m, f, tau, lam, seed)
    thr = 8 * lam
    sp = spells(S, thr)
    Sb, v, n, comp, Smax, TV = sp.T
    comp = comp.astype(bool)
    tr = np.sqrt(np.pi / 2) * 2 * np.log(Smax / thr) / n                  # rise + fall
    pl = np.sqrt(np.pi / 2) * np.maximum(TV - 2 * np.log(Smax / thr), 0) / n   # extra plateau variation
    ident = np.median(np.abs(v[comp] - np.sqrt(np.pi / 2) * TV[comp] / n[comp]) / v[comp])
    P = branch(Sb, v)
    Pc = [p[comp[p]] for p in P]                                        # complete spells only
    lx = np.log([Sb[p].mean() for p in Pc])
    share_tr = [np.median(tr[p] / (tr[p] + pl[p])) for p in Pc]
    nslope = np.polyfit(lx, np.log([np.mean(n[p]) for p in Pc]), 1)[0]
    cv_n = np.median([n[p].std() / n[p].mean() for p in Pc])
    return dict(ident=ident, z_all=zq(Sb, v, P), z_comp=zq(Sb, v, Pc), z_tr=zq(Sb, tr, Pc), z_pl=zq(Sb, pl + 1e-9, Pc),
                share_tr=(share_tr[0], share_tr[len(share_tr) // 2], share_tr[-1]), nslope=nslope, cv_n=cv_n,
                frac_comp=comp.mean())

if __name__ == "__main__":
    d = np.load(sys.argv[1]); lam = float(sys.argv[1].rsplit("_l", 1)[1][:-4]); tau = fit_tau(d["acf"])
    R = Parallel(n_jobs=8)(delayed(one)(d["m"][e * NE:(e + 1) * NE], d["f"][e * NE:(e + 1) * NE], tau, lam, 100 + e)
                           for e in range(8))
    r = lambda k: np.mean([x[k] for x in R], 0)
    rat = lambda z: " ".join(f"{a:.2f}" for a in z[1:] / z[0])
    print(f"identity sigma = sqrt(pi/2) TV/n on complete spells: median rel. error {np.median([x['ident'] for x in R]):.1e}")
    print(f"complete spells: {r('frac_comp'):.2f} of all")
    for k, name in (("z_all", "all spells"), ("z_comp", "complete spells"), ("z_tr", "transient part"), ("z_pl", "plateau part")):
        z = r(k); print(f"  {name:16s} zeta_1 {z[0]:.3f}  ratios {rat(z)}")
    print(f"  transient share of sigma (low / mid / high decline-branch size): " + " ".join(f"{a:.2f}" for a in r("share_tr")))
    print(f"  mean spell length ~ Sbar^{r('nslope'):.2f};  CV of length at fixed size {r('cv_n'):.2f} (exponential: 1)")

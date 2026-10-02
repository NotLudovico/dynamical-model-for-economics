"""Why is zeta_q/zeta_1 flat at alpha=2.5 under spells? Split zeta_q = zeta_q^A - e_q (beta note,
eq. zeta-split): zeta^A from v/sqrt(k), e_q = dln E[k^{q/2}|S]/dlnS, on the same decline bins.
Also: who are the most volatile spells (degree, length)?
Run: uv run python scripts/lifecycle/degree_split.py   (~6 min)
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import time
import numpy as np
from joblib import Parallel, delayed
from beta_spells import model, MU, SIGMA, LAM, N, C, START, DT
from relative_glv.msb import spell_volatility, size_volatility

T, NB, Q = 480.0, 20, np.arange(1, 5)

def slopes(S, ys):
    """cq_one decline branch (peak of binned median vol), then -dln E[y^q|S]/dlnS for each y."""
    v = ys[0]
    o = np.argsort(S); P = np.array_split(o, NB)
    bx = np.array([S[p].mean() for p in P]); pk = int(np.argmax([np.median(v[p]) for p in P]))
    dec = S > bx[pk]
    if dec.sum() < 200:
        return None
    Sx = S[dec]; o = np.argsort(Sx); P = np.array_split(o, max(6, NB - pk))
    lx = np.log([Sx[p].mean() for p in P])
    return np.array([[-np.polyfit(lx, np.log([np.mean(y[dec][p] ** q) for p in P]), 1)[0] for q in Q] for y in ys])

def economy(alpha, seed):
    model._ALPHA_PL = alpha
    a = model.coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)
    k = np.diff(a.indptr).astype(float)
    r = model.integrate(a, tmax=START + T, n_eval=int((START + T) / DT) + 1, lam=LAM, seed=seed,
                        method="RK45", rtol=1e-4, atol=1e-7)
    out = {}
    for name, f in (("spells", lambda: spell_volatility(r["W"], r["t"], window=(START, START + T), dt=DT)),
                    ("survivor", lambda: size_volatility(r["W"], r["t"], window=(START, START + T), dt=DT))):
        e = f()
        if name == "survivor":
            live = (N * r["W"][:, (r["t"] >= START - 1e-9)]).min(1) > N * 1e-6
            kk = k[live]
        else:
            kk = k[e["firm"]]
        S, v = e["Sbar"], e["vol"]
        z = slopes(S, [v, v / np.sqrt(kk), np.sqrt(kk)])         # zeta_q, zeta_q^A, -e_q
        top = v > np.quantile(v[S > np.median(S)], 0.95)          # most volatile 5% of the larger half
        big = S > np.median(S)
        out[name] = (z, np.median(kk[big & top]) / np.median(kk[big]),
                     np.median(e.get("length", np.full(S.size, T / DT))[big & top]) * DT,
                     np.median(e.get("length", np.full(S.size, T / DT))[big]) * DT,
                     (kk > np.quantile(k, 0.99))[big].mean(), S.size)
    return out

for alpha in (2.5, 6.0):
    t0 = time.time()
    R = Parallel(n_jobs=8)(delayed(economy)(alpha, s) for s in range(8))
    print(f"\nalpha={alpha}  ({(time.time() - t0) / 60:.1f} min)")
    for name in ("survivor", "spells"):
        Z = np.array([x[name][0] for x in R if x[name][0] is not None])     # (econ, 3, 4)
        m = Z.mean(0)
        zr, zAr, eq = m[0] / m[0, 0], m[1] / m[1, 0], -m[2]
        d = np.array([x[name][1:] for x in R])
        print(f"  {name:8s} zeta_1 {m[0, 0]:.3f}  ratios {zr[1]:.2f} {zr[2]:.2f} {zr[3]:.2f}"
              f" | zeta^A_1 {m[1, 0]:.3f} ratios {zAr[1]:.2f} {zAr[2]:.2f} {zAr[3]:.2f}"
              f" | e_q {eq[0]:.3f} {eq[1]:.3f} {eq[2]:.3f} {eq[3]:.3f}  e_1/zeta_1 {eq[0] / m[0, 0]:.2f}")
        print(f"           top-5% vol spells (larger half): degree x{np.median(d[:, 0]):.2f} vs larger half,"
              f" length {np.median(d[:, 1]):.1f} vs {np.median(d[:, 2]):.1f};"
              f" larger half in top-1% degree: {np.median(d[:, 3]):.3f}; records {np.median(d[:, 4]):.0f}", flush=True)

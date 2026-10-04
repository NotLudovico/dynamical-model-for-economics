"""Headline spell-panel numbers with the frozen estimator (msb.upper_zeta: larger half of spells by
Sbar, 10 equal-count bins), and the raw per-spell data so any later estimator change is free.
Full model, N = 8000, C = N/40, mu = -1.76, eight economies per point, spells >= 2 growth rates in
[180, 660], listing threshold share 8 lam / N.  The old peak-based estimator (beta_spells.cq_one) is
printed alongside for comparison.
Run: uv run python scripts/headline_spells.py   -> data/headline_spells/<point>.npz
     uv run python scripts/headline_spells.py summary"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
import numpy as np
from joblib import Parallel, delayed

N, MU, START, T, DT = 8000, 1.76, 180.0, 480.0, 0.5           # code convention (competition > 0)
POINTS = ((6.0, 1.75, 1e-3), (3.5, 1.9, 1e-3), (2.5, 1.75, 1e-3), (3.5, 1.9, 1e-2))
OUT = os.path.join(ROOT, "data", "headline_spells")


def path(a, s, l):
    return os.path.join(OUT, f"a{a}_s{s}_l{l:.0e}.npz")


def economy(alpha, sigma, lam, seed):
    from relative_glv import model
    from relative_glv.msb import spell_volatility
    model._ALPHA_PL = alpha
    a = model.coupling(N, MU, sigma, kind="powerlaw", seed=seed, mean_degree=N // 40)
    r = model.integrate(a, tmax=START + T, n_eval=int((START + T) / DT) + 1, lam=lam, seed=seed,
                        method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    sp = spell_volatility(r["W"], r["t"], window=(START, START + T), dt=DT, floor=8 * lam / N, min_growth=2)
    k = np.diff(a.indptr)
    return dict(Sbar=sp["Sbar"], vol=sp["vol"], length=sp["length"], censored=sp["censored"], degree=k[sp["firm"]])


def summary():
    from relative_glv.msb import upper_zeta
    from beta_spells import cq_one
    print(f"{'point':>22} | {'beta new':>8} {'ratios new':>16} | {'beta old':>8} {'ratios old':>16} | spells")
    for p in POINTS:
        if not os.path.exists(path(*p)):
            continue
        d = np.load(path(*p))
        e = d["economy"]
        zn = np.array([upper_zeta(d["Sbar"][e == i], d["vol"][e == i]) for i in np.unique(e)])
        zo = np.array([z for z in (cq_one(d["Sbar"][e == i], d["vol"][e == i]) for i in np.unique(e)) if z is not None])
        f = lambda Z: f"{np.median(Z[:, 0]):.3f}  " + " ".join(f"{x:.2f}" for x in Z.mean(0)[1:] / Z.mean(0)[0])
        print(f"a={p[0]} s={p[1]} l={p[2]:.0e} | {f(zn):>25} | {f(zo):>25} | {d['Sbar'].size}")
    print("data: beta ~ 0.15-0.20, ratios 2.0 2.6 2.9")


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    for p in POINTS:
        if os.path.exists(path(*p)):
            continue
        R = [r for r in Parallel(n_jobs=8)(delayed(economy)(*p, s) for s in range(8)) if r is not None]
        np.savez(path(*p), economy=np.concatenate([np.full(r["Sbar"].size, i) for i, r in enumerate(R)]),
                 **{k: np.concatenate([r[k] for r in R]) for k in R[0]})
        print("done", p, flush=True)
    summary()

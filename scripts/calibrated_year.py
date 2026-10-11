"""How the spell-panel statistics depend on the sampling step that is called a year.

The headline numbers (scripts/headline_spells.py) sample every DT = 0.5 time units, where yearly
growth within a spell is strongly autocorrelated; Compustat annual growth is nearly uncorrelated.
This reruns the same economies (same points, seeds and integrator) and scores the spell panel at
several steps dt: the pooled lag-1 autocorrelation of growth within spells, the frozen estimator
(msb.upper_zeta, larger half of spells), an all-bins fit (25 equal-count bins, MSB 2024) and the
binned curve E[v | Sbar] (30 bins) to measure the hump.  The calibrated year of a point is the
smallest dt whose pooled lag-1 autocorrelation is below 0.1 in absolute value.

Run: uv run python scripts/calibrated_year.py            -> data/calibrated_year/<point>.npz
     uv run python scripts/calibrated_year.py summary
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from joblib import Parallel, delayed
from headline_spells import N, MU, START, T, DT, POINTS

STEPS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
OUT = os.path.join(ROOT, "data", "calibrated_year")
LAG_OK = 0.1


def path(a, s, l):
    return os.path.join(OUT, f"a{a}_s{s}_l{l:.0e}.npz")


def lag1_pairs(W, t, window, dt, floor):
    """Consecutive growth pairs (g_t, g_{t+1}) within listed spells on the dt grid."""
    tg = np.arange(window[0], window[1] + 1e-9, dt)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * w, 1e-300))) for w in W])
    on = lnS > np.log(N * floor)
    g = np.diff(lnS, axis=1)
    ok = on[:, 2:] & on[:, 1:-1] & on[:, :-2]
    return g[:, :-1][ok], g[:, 1:][ok]


def economy(alpha, sigma, lam, seed):
    from relative_glv import model
    from relative_glv.msb import spell_volatility
    model._ALPHA_PL = alpha
    a = model.coupling(N, MU, sigma, kind="powerlaw", seed=seed, mean_degree=N // 40)
    r = model.integrate(a, tmax=START + T, n_eval=int((START + T) / DT) + 1, lam=lam, seed=seed,
                        method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    out = {}
    for dt in STEPS:
        sp = spell_volatility(r["W"], r["t"], window=(START, START + T), dt=dt, floor=8 * lam / N,
                              min_growth=2)
        x, y = lag1_pairs(r["W"], r["t"], (START, START + T), dt, 8 * lam / N)
        out[dt] = dict(Sbar=sp["Sbar"], vol=sp["vol"], length=sp["length"], lagx=x, lagy=y)
    return out


def all_bins_zeta(S, v, n_bins=25, qs=(1, 2, 3, 4)):
    ok = (v > 0) & (S > 0)
    S, v = S[ok], v[ok]
    P = np.array_split(np.argsort(S), n_bins)
    lx = np.log([S[p].mean() for p in P])
    return np.array([-np.polyfit(lx, np.log([np.mean(v[p] ** q) for p in P]), 1)[0] for q in qs])


def curve(S, v, n_bins=30):
    ok = (v > 0) & (S > 0)
    S, v = S[ok], v[ok]
    P = np.array_split(np.argsort(S), n_bins)
    return np.array([S[p].mean() for p in P]), np.array([v[p].mean() for p in P])


def summary():
    from relative_glv.msb import upper_zeta
    for p in POINTS:
        if not os.path.exists(path(*p)):
            continue
        d = np.load(path(*p))
        print(f"\nalpha={p[0]} sigma={p[1]} lam={p[2]:.0e}")
        print(f"{'dt':>4} {'lag1':>6} | {'beta(upper)':>11} {'ratios(upper)':>16} | {'beta(all)':>9} "
              f"{'ratios(all)':>16} | {'low':>5} {'peak':>5} {'top':>5} {'low/pk':>6} {'logS_pk':>7} | spells")
        cal = None
        for dt in STEPS:
            k = f"{dt:g}"
            e, S, v = d[f"econ_{k}"], d[f"Sbar_{k}"], d[f"vol_{k}"]
            lag = np.corrcoef(d[f"lagx_{k}"], d[f"lagy_{k}"])[0, 1]
            if cal is None and abs(lag) < LAG_OK:
                cal = dt
            Zu = np.array([upper_zeta(S[e == i], v[e == i]) for i in np.unique(e)])
            Za = np.array([all_bins_zeta(S[e == i], v[e == i]) for i in np.unique(e)])
            ru = Zu.mean(0)[1:] / Zu.mean(0)[0]
            ra = Za.mean(0)[1:] / Za.mean(0)[0]
            cs, cv = curve(S, v)
            ipk = int(np.argmax(cv))
            print(f"{dt:4g} {lag:6.2f} | {np.median(Zu[:, 0]):6.3f}±{Zu[:, 0].std():.2f} "
                  f"{ru[0]:5.2f} {ru[1]:5.2f} {ru[2]:5.2f} | {np.median(Za[:, 0]):9.3f} "
                  f"{ra[0]:5.2f} {ra[1]:5.2f} {ra[2]:5.2f} | {cv[0]:5.2f} {cv[ipk]:5.2f} {cv[-1]:5.2f} "
                  f"{cv[0] / cv[ipk]:6.2f} {np.log10(cs[ipk]):7.2f} | {S.size}")
        print(f"calibrated year (|lag1| < {LAG_OK}): dt = {cal}")
    print("\ndata: beta 0.15-0.20, ratios 2.0 2.6 2.9; smallest-bin vol 1.22, largest 0.22 (MSB 2024)")


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(p, s) for p in POINTS if not os.path.exists(path(*p)) for s in range(8)]
    res = Parallel(n_jobs=12, verbose=5)(delayed(economy)(*p, s) for p, s in jobs)
    for p in POINTS:
        R = [r for (pp, s), r in zip(jobs, res) if pp == p and r is not None]
        if not R:
            continue
        arr = {}
        for dt in STEPS:
            k = f"{dt:g}"
            arr[f"econ_{k}"] = np.concatenate([np.full(r[dt]["Sbar"].size, i) for i, r in enumerate(R)])
            for f in ("Sbar", "vol", "length", "lagx", "lagy"):
                arr[f"{f}_{k}"] = np.concatenate([r[dt][f] for r in R])
        np.savez_compressed(path(*p), **arr)
        print("saved", p, f"({len(R)} economies)", flush=True)
    summary()

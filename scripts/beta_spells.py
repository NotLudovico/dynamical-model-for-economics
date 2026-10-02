"""beta and zeta_q/zeta_1 on the MSB unbalanced panel (msb.spell_volatility) against the
balanced survivor panel of the thesis (msb.size_volatility).

The survivor panel keeps only firms that persist across the whole window, a set that empties
as the window grows, so its beta has no long-window limit. Moran-Secchi-Bouchaud (2024,
Sec. 3.1) let every firm contribute over its own listed life; in the model that is one record
per listed spell, and only the spells cut by the window edges depend on the window.

Thesis operating point, N=8000, C=N/40, power-law graphs, per economy (never pooled):
  window test    T_obs = 60, 120, 240, 480 from t=180; floor 1e-6; spells with >= 2 and >= 20
                 growth rates (MSB's sample and their robustness sample)
  listing floor  share floor 1e-6, 1e-5, 1e-4 at T_obs = 240 (spells >= 2)
zeta_q: thesis estimator (cq_one: own decline branch per economy), ratio of economy means.

Run: uv run python scripts/beta_spells.py   (-> data/beta_spells.npz, ~10 min)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from relative_glv import model
from relative_glv.msb import size_volatility, spell_volatility

MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N, SEEDS, ALPHAS = 8000, 8, (2.2, 2.5, 3.5, 6.0)
C, START, DT, NB = N // 40, 180.0, 0.5, 20
TOBS, FLOORS = (60.0, 120.0, 240.0, 480.0), (1e-5, 1e-4)
COLS = ("alpha", "seed", "T_obs", "spells", "floor", "min_growth",
        "beta", "z1", "z2", "z3", "z4", "n", "censored", "median_length")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "beta_spells.npz")


def cq_one(S, v, nb=NB):
    """zeta_1..4 for ONE economy on its OWN decline branch (thesis estimator, as in
    scripts/pl22_conditional.py)."""
    live = (v > 0) & np.isfinite(v) & (S > 0)
    S, v = S[live], v[live]
    if S.size < 200:
        return None
    o = np.argsort(S); P = np.array_split(np.arange(o.size), nb)
    bx = np.array([S[o][p].mean() for p in P]); by = np.array([np.median(v[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]
    pk = int(np.argmax(by)); dec = S > bx[pk]; nfit = max(6, nb - pk)
    if dec.sum() < 200:
        return None
    Sx, vx = S[dec], v[dec]
    ox = np.argsort(Sx); Px = np.array_split(np.arange(ox.size), nfit)
    bxx = np.array([Sx[ox][p].mean() for p in Px]); out = []
    for q in (1, 2, 3, 4):
        byy = np.array([np.mean(vx[ox][p] ** q) for p in Px])
        mm = (bxx > 0) & (byy > 0)
        if mm.sum() < 3:
            return None
        out.append(-np.polyfit(np.log10(bxx[mm]), np.log10(byy[mm]), 1)[0])
    c = np.array(out)
    return c if c[0] > 0 else None


def record(alpha, seed, T, spells, floor, m, r):
    z = cq_one(r["Sbar"], r["vol"])
    z = np.full(4, np.nan) if z is None else z
    extra = ((r["Sbar"].size, r["censored"].mean(), np.median(r["length"]) * DT) if spells
             else (r["Sbar"].size, 1.0, T))
    return (alpha, seed, T, spells, floor, m, r["beta"], *z, *extra)


def economy(alpha, seed):
    model._ALPHA_PL = alpha                         # read at call time by _powerlaw_adjacency
    a = model.coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)
    tmax = START + max(TOBS)
    r = model.integrate(a, tmax=tmax, n_eval=int(tmax / DT) + 1, lam=LAM, seed=seed,
                        method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return []
    t, W, rows = r["t"], r["W"], []
    for T in TOBS:
        win = (START, START + T)
        rows.append(record(alpha, seed, T, 0, 1e-6, 0, size_volatility(W, t, window=win, dt=DT)))
        for m in (2, 20):
            rows.append(record(alpha, seed, T, 1, 1e-6, m,
                               spell_volatility(W, t, window=win, dt=DT, min_growth=m)))
    for fl in FLOORS:
        rows.append(record(alpha, seed, 240.0, 1, fl, 2,
                           spell_volatility(W, t, window=(START, START + 240.0), dt=DT, floor=fl)))
    return rows


def summary(R):
    """Median beta, ratio-of-means zeta_q/zeta_1 and spell statistics over economies."""
    c = {k: i for i, k in enumerate(COLS)}
    z = R[:, c["z1"]:c["z4"] + 1]
    z = z[np.isfinite(z).all(1)]
    zr = z.mean(0) / z.mean(0)[0] if len(z) else np.full(4, np.nan)
    return (f"beta {np.nanmedian(R[:, c['beta']]):.3f}±{np.nanstd(R[:, c['beta']]) / np.sqrt(len(R)):.3f}"
            f" | zeta_q/zeta_1 {zr[1]:.2f} {zr[2]:.2f} {zr[3]:.2f} ({len(z)} econ)"
            f" | n {np.median(R[:, c['n']]):.0f}  cens {np.median(R[:, c['censored']]):.2f}"
            f"  med len {np.median(R[:, c['median_length']]):.1f}")


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for alpha in ALPHAS:
        for x in Parallel(n_jobs=min(8, os.cpu_count()))(delayed(economy)(alpha, s) for s in range(SEEDS)):
            rows += x
        print(f"alpha={alpha} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    R = np.array(rows, float)
    np.savez(OUT, cols=np.array(COLS), rows=R)
    c = {k: i for i, k in enumerate(COLS)}
    for alpha in ALPHAS:
        A = R[R[:, c["alpha"]] == alpha]
        print(f"\nalpha={alpha}")
        for T in TOBS:
            for sp, m, name in ((0, 0, "survivor  "), (1, 2, "spells>=2 "), (1, 20, "spells>=20")):
                sel = ((A[:, c["T_obs"]] == T) & (A[:, c["spells"]] == sp) & (A[:, c["min_growth"]] == m)
                       & (A[:, c["floor"]] == 1e-6))
                print(f"  T_obs={T:4.0f} {name} {summary(A[sel])}")
        for fl in FLOORS:
            sel = (A[:, c["floor"]] == fl)
            print(f"  T_obs= 240 floor {fl:.0e} {summary(A[sel])}")

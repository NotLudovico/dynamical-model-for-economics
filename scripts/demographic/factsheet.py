"""Fact sheet of the demographic-noise model against the firm-growth facts, at a calibrated year.

Model: relative_glv.model.integrate_demographic (relative GLV + sqrt(D m x) noise, newborn at S = D).
Points: the reference point alpha 2.5, sigma 1.75, D 0.1 and alpha 3.5, sigma 1.9, D 0.1; N = 8000,
C = 200, mu = 1.76 (code convention, competition > 0), 4 economies each, burn-in 100, window 480,
sampled every 0.5.  Trajectories are kept (float32) in data/demographic/factsheet/.

Year: the smallest step in STEPS whose pooled lag-1 autocorrelation of growth within lives is below
0.1 in absolute value (Compustat annual growth is nearly uncorrelated).  A life is a maximal run of
samples between deaths at a node (window-censored lives included, as in an unbalanced panel).

Facts (MSB 2024 unless stated): (1) beta and zeta_q/zeta_1 from E[v^q | Sbar], all 25 equal-count
bins, lives with >= n growth rates; local beta by fifths; (2) level, smallest and largest bin
(Compustat 1.22 and 0.22); (3) tail b of v/E[v|bin] (data 4.6); (4) excess kurtosis of rescaled
growth by size fifth (no Gaussianisation in data); (5) persistence: excess kurtosis of rescaled
3-step growth against the iid value kappa_1/3; (6) correlation of a firm's absolute growth with the
aggregate, by size fifth (rises with size in data); (7) size distribution: Hill index of the top 1%
and the share of firm-time per decade (Zipf: Hill ~ 1).

Run: uv run python scripts/demographic/factsheet.py          (simulate + analyse)
     uv run python scripts/demographic/factsheet.py analyse  (analyse cached trajectories)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

OUT = os.path.join(ROOT, "data", "demographic", "factsheet")
POINTS = ((2.5, 1.75, 0.1), (3.5, 1.9, 0.1))
N, C, MU, START, TOBS, DT, SEEDS = 8000, 200, 1.76, 100.0, 480.0, 0.5, 4
STEPS = (1, 2, 4, 6, 8)                      # in samples of DT
NMIN = (2, 5, 10, 20)
LAG_OK = 0.1


def path(a, s, D, seed):
    return os.path.join(OUT, f"a{a}_s{s}_D{D:g}_seed{seed}.npz")


def simulate(alpha, sigma, D, seed):
    from relative_glv import model
    f = path(alpha, sigma, D, seed)
    if os.path.exists(f):
        return f
    model._ALPHA_PL = alpha
    a = model.coupling(N, MU, sigma, kind="powerlaw", seed=seed, mean_degree=C)
    r = model.integrate_demographic(a, D=D, tmax=START + TOBS, t_start=START, dt_save=DT, seed=seed)
    S = r["S"]
    fb = np.array([S[:, k] @ (1.0 - S[:, k] - a @ S[:, k]) / N for k in range(S.shape[1])])
    np.savez_compressed(f, S=S.astype(np.float32), L=r["life"].astype(np.int32), fbar=fb)
    return f


def lives(S, L, k):
    """Per life on the k-subsampled grid: (Sbar, v, n, residuals g - gbar, raw g, start index)."""
    S, L = S[:, ::k], L[:, ::k]
    lnS, T = np.log(S), S.shape[1]
    out = []
    for i in range(S.shape[0]):
        for seg in np.split(np.arange(T), np.flatnonzero(np.diff(L[i])) + 1):
            if seg.size < 3:
                continue
            g = np.diff(lnS[i, seg])
            v = np.sqrt(np.pi / 2) * np.abs(g - g.mean()).mean()
            if v > 0:
                out.append((S[i, seg].mean(), v, g.size, g - g.mean(), g, i, seg[0]))
    return out


def lag1(S, L, k):
    S, L = S[:, ::k], L[:, ::k]
    g = np.diff(np.log(S), axis=1)
    same = (L[:, 2:] == L[:, 1:-1]) & (L[:, 1:-1] == L[:, :-2])
    return g[:, :-1][same], g[:, 1:][same]


def zetas(Sb, v, nb=25):
    P = np.array_split(np.argsort(Sb), nb)
    lx = np.log([Sb[p].mean() for p in P])
    z = np.array([-np.polyfit(lx, np.log([np.mean(v[p] ** q) for p in P]), 1)[0] for q in (1, 2, 3, 4)])
    mv = np.log([v[p].mean() for p in P])
    loc = [-np.polyfit(lx[i:i + 5], mv[i:i + 5], 1)[0] for i in range(0, nb, 5)]
    return z, loc, np.exp(lx), np.exp(mv), (lx[-1] - lx[0]) / np.log(10)


def hill(z, p):
    z = np.sort(z)[::-1]
    k = max(int(p * len(z)), 10)
    return float(1.0 / np.mean(np.log(z[:k] / z[k])))


def exkurt(x):
    return float(np.mean(x ** 4) / np.mean(x ** 2) ** 2 - 3)


def analyse_point(p):
    files = [path(*p, s) for s in range(SEEDS) if os.path.exists(path(*p, s))]
    if not files:
        return
    data = [np.load(f) for f in files]
    print(f"\n=== alpha={p[0]} sigma={p[1]} D={p[2]}  ({len(files)} economies, N={N}, C={C}) ===")
    lags = {}
    for k in STEPS:
        xs, ys = zip(*[lag1(d["S"].astype(float), d["L"], k) for d in data])
        lags[k] = np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1]
    year = next((k for k in STEPS if abs(lags[k]) < LAG_OK), None)
    print("lag-1 by step: " + "  ".join(f"{k * DT:g}: {lags[k]:+.2f}" for k in STEPS),
          f"-> year = {year * DT if year else None}")
    for k in sorted({1, year or 1}):
        print(f"\n-- step {k * DT:g} {'(headline)' if k == 1 else '(calibrated year)'} --")
        per = [lives(d["S"].astype(float), d["L"], k) for d in data]
        allv = [x for e in per for x in e]
        Sb = np.array([x[0] for x in allv]); v = np.array([x[1] for x in allv]); n = np.array([x[2] for x in allv])
        for nm in NMIN:
            sel = n >= nm
            Z = np.array([zetas(np.array([x[0] for x in e if x[2] >= nm]), np.array([x[1] for x in e if x[2] >= nm]))[0]
                          for e in per])
            z, loc, bs, bv, dec = zetas(Sb[sel], v[sel])
            print(f"n>={nm:2d}: beta {z[0]:.3f} (econ sd {Z[:, 0].std():.3f})  ratios {z[1]/z[0]:.2f} {z[2]/z[0]:.2f} "
                  f"{z[3]/z[0]:.2f}  local {' '.join(f'{x:.2f}' for x in loc)}  range {dec:.1f} dec  "
                  f"vol {bv[0]:.2f} -> {bv[-1]:.2f}  lives {sel.sum()}")
        sel = n >= 10
        P = np.array_split(np.argsort(Sb[sel]), 25)
        r = np.empty(sel.sum()); vv = v[sel]
        for q in P:
            r[q] = vv[q] / vv[q].mean()
        print(f"tail b of v/E[v|bin] (n>=10, top 5%): {hill(r, 0.05):.2f}   (data 4.6)")
        # Gaussianisation and persistence, lives with >= 20 growth rates
        L20 = [x for x in allv if x[2] >= 20]
        sb20 = np.array([x[0] for x in L20]); qs = np.quantile(sb20, [0.2, 0.4, 0.6, 0.8])
        fifth = np.digitize(sb20, qs)
        k1, k3 = [], []
        for f5 in range(5):
            res = [x[3] / x[1] for x, ff in zip(L20, fifth) if ff == f5]
            z1 = np.concatenate(res)
            z3 = np.concatenate([np.convolve(rr, np.ones(3), "valid")[::3] for rr in res if rr.size >= 3])
            k1.append(exkurt(z1)); k3.append(exkurt(z3 / z3.std()))
        print("excess kurtosis of rescaled growth by size fifth: " + " ".join(f"{x:.2f}" for x in k1))
        print("3-step excess kurtosis by fifth: " + " ".join(f"{x:.2f}" for x in k3)
              + "   iid value k1/3: " + " ".join(f"{x / 3:.2f}" for x in k1))
        # correlation with the aggregate: absolute growth = d ln S + integral of fbar
        cor_S, cor_c = [], []
        for d, e in zip(data, per):
            G = d["fbar"][:-1] * DT                       # aggregate log growth per sample interval
            nb = G.size // k
            Gk = G[:nb * k].reshape(nb, k).sum(1)         # per step of k samples
            for x in e:
                if x[2] < 20:
                    continue
                s0 = x[6]; agg = Gk[s0:s0 + x[2]]
                if agg.size != x[2]:
                    continue
                gabs = x[4] + agg
                if agg.std() > 0:
                    cor_S.append(x[0]); cor_c.append(np.corrcoef(gabs, agg)[0, 1])
        cor_S, cor_c = np.array(cor_S), np.array(cor_c)
        qq = np.digitize(cor_S, np.quantile(cor_S, [0.2, 0.4, 0.6, 0.8]))
        print("corr(own absolute growth, aggregate) by size fifth: "
              + " ".join(f"{np.nanmean(cor_c[qq == f]):.3f}" for f in range(5))
              + f"   aggregate vol per step {np.std(Gk):.3f}")
    # sizes (all nodes, every 5th sample)
    allS = np.concatenate([d["S"][:, ::5].ravel() for d in data]).astype(float)
    lx = np.log10(allS)
    dec = " ".join(f"[{lo},{lo + 1}): {100 * np.mean((lx >= lo) & (lx < lo + 1)):.0f}%" for lo in range(-4, 2))
    print(f"\nsizes: Hill top 1% {hill(allS, 0.01):.2f}, top 5% {hill(allS, 0.05):.2f};  firm-time by decade of S: {dec}")
    print(f"largest firm share (mean) {np.mean([d['S'].max(0).mean() / N for d in data]):.4f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:2] != ["analyse"]:
        from joblib import Parallel, delayed
        jobs = [(p, s) for p in POINTS for s in range(SEEDS)]
        Parallel(n_jobs=len(jobs), verbose=5)(delayed(simulate)(*p, s) for p, s in jobs)
    for p in POINTS:
        analyse_point(p)
    print("\nCompustat (MSB 2024): beta 0.20, ratios 2.0 2.6 2.9, vol 1.22 -> 0.22 over 4 decades, b 4.6, "
          "no Gaussianisation, 3-step tails fatter than iid, corr with aggregate rising with size; Zipf Hill ~1")

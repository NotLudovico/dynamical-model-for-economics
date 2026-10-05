"""Other stylised facts for the replacement demographic-noise model (alpha=2.5, sigma=1.75, D=0.1, N=8000).
Two economies, full trajectories (sampling 0.5). Sample: lives with >= 20 growth rates unless stated.
Run: uv run python scripts/demographic/facts.py   (add "plot" to skip the simulation)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "demographic")
import numpy as np

OUT = os.path.join(DATA, "facts")
ALPHA, SIGMA, D, MING = 2.5, 1.75, 0.1, 20

if "plot" not in sys.argv:
    from joblib import Parallel, delayed

    def traj(seed):
        from relative_glv import model
        model._ALPHA_PL = ALPHA
        a = model.coupling(8000, 1.76, SIGMA, kind="powerlaw", seed=seed, mean_degree=200)
        r = model.integrate_demographic(a, D=D, tmax=580.0, t_start=100.0, dt_save=0.5, seed=seed)
        return r["S"].astype(np.float32), r["life"].astype(np.int32)

    R = Parallel(n_jobs=2)(delayed(traj)(s) for s in range(2))
    np.savez_compressed(OUT + ".npz", S=np.stack([r[0] for r in R]), L=np.stack([r[1] for r in R]))

# ---------------- analysis
d = np.load(OUT + ".npz")
G, AGE, SZ, VOL, EXITNEXT, LIFELEN, COMPLETE, AC1, SNOW, SB = [], [], [], [], [], [], [], [], [], []
for S, L in zip(d["S"], d["L"]):
    S = S.astype(float); T = S.shape[1]
    for i in range(S.shape[0]):
        cuts = np.flatnonzero(np.diff(L[i])) + 1
        segs = np.split(np.arange(T), cuts)
        for k, seg in enumerate(segs):
            born_in = k > 0                          # birth observed inside the window
            dies_in = k < len(segs) - 1
            if born_in and dies_in:
                LIFELEN.append(seg.size * 0.5); COMPLETE.append(True)
            if seg.size < MING + 1:
                continue
            x = S[i, seg]; g = np.diff(np.log(x))
            G.append(g); SZ.append(x[:-1]); SB.append(np.full(g.size, x.mean()))
            VOL.append(np.full(g.size, np.sqrt(np.pi / 2) * np.abs(g - g.mean()).mean()))
            AGE.append(np.arange(g.size) * 0.5 if born_in else np.full(g.size, np.nan))
            ex = np.zeros(g.size, bool)
            if dies_in: ex[-1] = True                 # last observed step before death
            EXITNEXT.append(ex)
            if g.size > 3: AC1.append(np.corrcoef(g[:-1], g[1:])[0, 1])
        SNOW.append(S[i, -1])
G, SZ, SB, VOL, AGE, EXITNEXT = map(np.concatenate, (G, SZ, SB, VOL, AGE, EXITNEXT))
LIFELEN = np.array(LIFELEN); AC1 = np.array(AC1)
print(f"sample: {G.size} growth rates in lives with >= {MING} growth rates; complete lives (all lengths) {LIFELEN.size}")

# 1 tent: rescale growth by its size-bin scale
bins = np.array_split(np.argsort(SZ), 25)
z = np.empty_like(G)
for b in bins:
    gb = G[b] - G[b].mean(); z[b] = gb / (np.sqrt(np.pi / 2) * np.abs(gb).mean())
kurt = np.mean(z**4) / np.mean(z**2) ** 2 - 3
print(f"1 growth distribution: excess kurtosis {kurt:.2f} (Gaussian 0, Laplace 3); skew {np.mean(z**3) / np.mean(z**2)**1.5:+.2f}")
# 2 Gibrat
mg = np.array([G[b].mean() for b in bins]); sb = np.array([SZ[b].mean() for b in bins])
print(f"2 mean growth by size fifths: {np.round([mg[i:i+5].mean() for i in range(0, 25, 5)], 3)} (sd of g ~ {G.std():.2f})")
# 3 size distribution of established firms (current size, lives >= 20)
ls = np.log10(SZ)
print(f"3 log10 size of established firms: mean {ls.mean():.2f}, sd {ls.std():.2f}, skew {np.mean((ls-ls.mean())**3)/ls.std()**3:+.2f}, "
      f"excess kurt {np.mean((ls-ls.mean())**4)/ls.std()**4-3:+.2f}")
# 4 lifetimes (complete lives, any length)
print(f"4 complete lifetimes: median {np.median(LIFELEN):.1f}, mean {LIFELEN.mean():.1f}, P(>10) {np.mean(LIFELEN > 10):.3f}, P(>50) {np.mean(LIFELEN > 50):.4f}")
# 5 age: volatility and exit hazard by age (lives born inside window)
ok = np.isfinite(AGE)
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 50), (50, 500)):
    m = ok & (AGE >= lo) & (AGE < hi)
    print(f"5 age [{lo},{hi}): |g| mean {np.abs(G[m]).mean():.3f}, exit hazard per step {EXITNEXT[m].mean():.4f}, n {m.sum()}")
for k, b in enumerate(np.array_split(np.argsort(SZ), 5)):
    print(f"5b size fifth {k+1}: exit hazard per step {EXITNEXT[b].mean():.4f}")
# 6 autocorrelation
print(f"6 lag-1 growth autocorrelation within lives: median {np.median(AC1):+.3f}, mean {AC1.mean():+.3f}")
np.savez(OUT + "_summary.npz", z=z[::5], mg=mg, sb=sb, ls=ls[::5], lifelen=LIFELEN, age=AGE, G=G, ex=EXITNEXT, sz=SZ, ac1=AC1)

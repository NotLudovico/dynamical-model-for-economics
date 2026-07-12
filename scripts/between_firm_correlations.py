"""Between-firm growth-rate correlations vs firm size (the model's distinctive prediction).

Moran et al. conjecture the granular picture's missing ingredient is correlation between firm
growth rates that GROWS with size. This measures it in the model, two ways, on the thesis graph
(power-law 2.5, mean-degree, C = N/40):

  1. market beta:  rho_i = corr_t(g_i(t), gbar(t)),  gbar = cross-sectional mean growth.
                   How strongly a firm co-moves with the aggregate, binned by size.
  2. same-size pairwise:  mean pairwise corr_t(g_i, g_j) among firms in a size bin.
                   Moran's "correlation of the growth of firms of similar sizes".

Both are read off the per-firm growth time series g (live firms x time) that size_volatility
already returns. Quenched: computed per economy, then averaged. Saves data/between_firm_corr.npz
and thesis/between_firm_corr.png.

Env: BFC_SEEDS (10), BFC_N (16000), BFC_NJOBS (6), BFC_SMOKE=1.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate
from relative_glv.msb import size_volatility

SMOKE = os.environ.get("BFC_SMOKE", "0") == "1"
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N = 1500 if SMOKE else int(os.environ.get("BFC_N", 16000))
Cdeg = max(2, round(100 * N / 4000))
SEEDS = 3 if SMOKE else int(os.environ.get("BFC_SEEDS", 10))
NJOBS = int(os.environ.get("BFC_NJOBS", 6))
TMAX, N_EVAL = (100.0, 400) if SMOKE else (250.0, 700)
LATE, DT, NB = ((75.0, 98.0) if SMOKE else (180.0, 240.0)), 0.5, 16
MAXPAIR = 150                                    # cap firms/bin for the pairwise corr matrix


def corr_rows_to_vec(g, v):
    """Pearson corr of each row of g (F x T) with a single vector v (T,)."""
    gc = g - g.mean(1, keepdims=True)
    vc = v - v.mean()
    num = gc @ vc
    den = np.sqrt((gc ** 2).sum(1) * (vc ** 2).sum())
    return np.where(den > 0, num / np.where(den == 0, 1, den), np.nan)


def mean_pairwise_corr(g):
    """Mean off-diagonal Pearson correlation among the rows of g (F x T)."""
    if g.shape[0] < 2:
        return np.nan
    C = np.corrcoef(g)
    iu = np.triu_indices_from(C, k=1)
    c = C[iu]
    c = c[np.isfinite(c)]
    return float(c.mean()) if c.size else np.nan


def binned(x, rho_mkt, g, rng_pairs):
    """Median market-corr and mean same-bin pairwise-corr, binned by the ordering variable x."""
    o = np.argsort(x)
    parts = np.array_split(np.arange(o.size), NB)
    bx, mkt, pw = [], [], []
    for p in parts:
        idx = o[p]
        bx.append(float(np.median(x[idx])))
        mkt.append(float(np.nanmedian(rho_mkt[idx])))
        sub = idx if idx.size <= MAXPAIR else rng_pairs.choice(idx, MAXPAIR, replace=False)
        pw.append(mean_pairwise_corr(g[sub]))
    return np.array(bx), np.array(mkt), np.array(pw)


def one_economy(seed, rng_pairs):
    a = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=Cdeg)
    deg_all = np.diff(a.indptr).astype(float)         # degree per firm (CSR row lengths)
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                  method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT, n_bins=NB)
    g, Sbar = sv["growth"], sv["Sbar"]                # g: (F live) x (T-1) ; Sbar: (F,)
    if g.shape[0] < 8 * NB:
        return None
    win = (r["t"] >= LATE[0]) & (r["t"] <= LATE[1])   # live-firm order matches size_volatility
    deg = deg_all[np.where(r["W"][:, win].min(1) > 1e-6)[0]]
    gbar = g.mean(0)                                  # cross-sectional mean growth per timestep
    rho_mkt = corr_rows_to_vec(g, gbar)               # per-firm market correlation

    binS, mktS, pwS = binned(Sbar, rho_mkt, g, rng_pairs)
    binK, mktK, pwK = binned(deg, rho_mkt, g, rng_pairs)
    return binS, mktS, pwS, binK, mktK, pwK


t0 = time.time()
seeds = range(SEEDS)
rngs = [np.random.default_rng(1000 + s) for s in seeds]
res = [x for x in Parallel(n_jobs=NJOBS, backend="loky")(
    delayed(one_economy)(s, rngs[s]) for s in seeds) if x is not None]
print(f"{'SMOKE ' if SMOKE else ''}between-firm corr  N={N} C={Cdeg}  {len(res)}/{SEEDS} economies"
      f"  in {(time.time()-t0)/60:.1f} min")

# quenched: align each economy's NB bins by rank, average across economies
def agg(i):
    a = np.array([r[i] for r in res])
    return np.nanmedian(a, 0), np.nanstd(a, 0) / np.sqrt(len(res))
binS = np.nanmedian([r[0] for r in res], 0); mktS, mktS_e = agg(1); pwS, pwS_e = agg(2)
binK = np.nanmedian([r[3] for r in res], 0); mktK, mktK_e = agg(4); pwK, pwK_e = agg(5)

for lbl, x, y in [("market vs size", binS, mktS), ("market vs degree", binK, mktK),
                  ("pairwise vs size", binS, pwS), ("pairwise vs degree", binK, pwK)]:
    sl = np.polyfit(np.log10(x), y, 1)[0]
    print(f"  {lbl:20s}:", np.round(y, 3), f" slope/decade={sl:+.3f}")

np.savez("data/between_firm_corr.npz", N=N, C=Cdeg, seeds=len(res),
         binS=binS, mktS=mktS, mktS_e=mktS_e, pwS=pwS, pwS_e=pwS_e,
         binK=binK, mktK=mktK, mktK_e=mktK_e, pwK=pwK, pwK_e=pwK_e)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
panels = [(0, 0, binS, mktS, mktS_e, r"mean relative size $\bar S$", r"corr$(g_i,\bar g)$",
           "market co-movement vs size", "#1d3557"),
          (0, 1, binK, mktK, mktK_e, r"degree $k$", r"corr$(g_i,\bar g)$",
           "market co-movement vs degree", "#1d3557"),
          (1, 0, binS, pwS, pwS_e, r"mean relative size $\bar S$", r"pairwise corr$(g_i,g_j)$",
           "same-bin pairwise vs size", "#e76f51"),
          (1, 1, binK, pwK, pwK_e, r"degree $k$", r"pairwise corr$(g_i,g_j)$",
           "same-bin pairwise vs degree", "#e76f51")]
for i, j, x, y, e, xl, yl, ti, c in panels:
    ax[i, j].errorbar(x, y, yerr=e, marker="o", ms=6, capsize=3, color=c)
    ax[i, j].set_xscale("log")
    ax[i, j].axhline(0, color="0.7", lw=0.8, zorder=0)
    ax[i, j].set(xlabel=xl, ylabel=yl, title=ti)
plt.tight_layout()
fig.savefig("thesis/between_firm_corr.png", dpi=140)
print("saved data/between_firm_corr.npz and thesis/between_firm_corr.png")

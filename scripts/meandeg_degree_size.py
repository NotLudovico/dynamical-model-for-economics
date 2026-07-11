"""Does degree set size in the mean-degree model?

Per-economy (quenched) test: Spearman rho(k_i, Sbar_i), log-log slope
Sbar ~ k^delta, and the degree-binned size curve. sigma(k) reported alongside
for context (the bridge law says hubs are noisier; this asks if they are also
bigger/smaller). Mean-degree pl2.5 at the thesis operating point, C proportional to N.

Run: python3 scripts/meandeg_degree_size.py            (N=8000, 10 seeds)
     MDS_SMOKE=1 python3 scripts/meandeg_degree_size.py (code-path check only)
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate
from relative_glv.msb import size_volatility

SMOKE = os.environ.get("MDS_SMOKE", "0") == "1"
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N0, C0 = 4000, 100                                  # C ∝ N anchor (same as meandeg_characterize)
N = 1500 if SMOKE else int(os.environ.get("MDS_N", 8000))
Cdeg = max(2, round(C0 * N / N0))
SEEDS = 2 if SMOKE else int(os.environ.get("MDS_SEEDS", 10))
TMAX, N_EVAL = (100.0, 400) if SMOKE else (250.0, 700)
LATE, DT = ((75.0, 98.0) if SMOKE else (180.0, 240.0)), 0.5
NJOBS = int(os.environ.get("MDS_NJOBS", 8))
NBIN = 12

print(f"{'SMOKE ' if SMOKE else ''}degree->size  mean-degree pl2.5  N={N} C={Cdeg}  "
      f"{SEEDS} seeds  mu={MU} sigma={SIGMA} lam={LAM}")


def simulate(seed):
    a = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=Cdeg)
    k = np.diff(a.indptr).astype(float)             # per-node degree from the sparse rows
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                  method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    win = (r["t"] >= LATE[0]) & (r["t"] <= LATE[1])
    live = r["W"][:, win].min(1) > 1e-6             # same mask size_volatility applies internally
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT)
    if sv["growth"].size == 0:
        return None
    return dict(k=k[live], Sbar=sv["Sbar"], vol=sv["vol"], seed=seed)


t0 = time.time()
recs = [x for x in Parallel(n_jobs=NJOBS, backend="loky")(
    delayed(simulate)(s) for s in range(SEEDS)) if x is not None]
assert recs, "no economies survived"
print(f"survived {len(recs)}/{SEEDS}  ({(time.time() - t0) / 60:.1f} min)")

# per-economy statistics (NEVER pool economies)
rho_S, rho_v, slope_S = [], [], []
for x in recs:
    ok = (x["Sbar"] > 0) & (x["vol"] > 0) & np.isfinite(x["vol"])
    k, S, v = x["k"][ok], x["Sbar"][ok], x["vol"][ok]
    rho_S.append(spearmanr(k, S).statistic)
    rho_v.append(spearmanr(k, v).statistic)
    slope_S.append(np.polyfit(np.log(k), np.log(S), 1)[0])
rho_S, rho_v, slope_S = map(np.asarray, (rho_S, rho_v, slope_S))

print(f"rho(k, Sbar)  median {np.median(rho_S):+.3f}  range [{rho_S.min():+.3f}, {rho_S.max():+.3f}]")
print(f"rho(k, vol)   median {np.median(rho_v):+.3f}  range [{rho_v.min():+.3f}, {rho_v.max():+.3f}]")
print(f"slope delta in Sbar ~ k^delta  median {np.median(slope_S):+.3f} ± {slope_S.std():.3f}")

# degree-binned size curve, per economy then median across economies
kmin = min(x["k"].min() for x in recs)
kmax = max(x["k"].max() for x in recs)
edges = np.logspace(np.log10(kmin), np.log10(kmax + 1), NBIN + 1)
mids = np.sqrt(edges[:-1] * edges[1:])
curves_S, curves_v = [], []
for x in recs:
    ok = (x["Sbar"] > 0) & (x["vol"] > 0) & np.isfinite(x["vol"])
    k, S, v = x["k"][ok], x["Sbar"][ok], x["vol"][ok]
    S, v = S / np.median(S), v / np.median(v)       # each economy in its own units
    idx = np.digitize(k, edges) - 1
    curves_S.append([np.median(S[idx == b]) if (idx == b).sum() >= 5 else np.nan for b in range(NBIN)])
    curves_v.append([np.median(v[idx == b]) if (idx == b).sum() >= 5 else np.nan for b in range(NBIN)])
bin_S = np.nanmedian(np.array(curves_S), 0)
bin_v = np.nanmedian(np.array(curves_v), 0)

tag = "_smoke" if SMOKE else ("" if N == 8000 else f"_N{N}")
np.savez(f"data/meandeg_degree_size{tag}.npz",
         N=N, C=Cdeg, seeds=len(recs), mu=MU, sigma=SIGMA, lam=LAM,
         rho_S=rho_S, rho_v=rho_v, slope_S=slope_S,
         bin_k=mids, bin_S=bin_S, bin_v=bin_v,
         k=np.concatenate([x["k"] for x in recs]),
         Sbar=np.concatenate([x["Sbar"] for x in recs]),
         vol=np.concatenate([x["vol"] for x in recs]),
         econ=np.concatenate([np.full(x["k"].size, i) for i, x in enumerate(recs)]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
x0 = recs[0]
ok0 = (x0["Sbar"] > 0) & (x0["vol"] > 0) & np.isfinite(x0["vol"])
k0 = x0["k"][ok0]
S0 = x0["Sbar"][ok0] / np.median(x0["Sbar"][ok0])   # same per-economy units as the binned curve
v0 = x0["vol"][ok0] / np.median(x0["vol"][ok0])
ax[0].loglog(k0, S0, ".", ms=2, alpha=0.25, color="gray", label="firms (one economy)")
ax[0].loglog(mids, bin_S, "o-", color="C3", label="binned median (all economies)")
ax[0].set(xlabel="degree $k_i$", ylabel=r"$\bar S_i$ / median", title=rf"size: $\rho$={np.median(rho_S):+.2f}, $\delta$={np.median(slope_S):+.2f}")
ax[0].legend(fontsize=8)
ax[1].loglog(k0, v0, ".", ms=2, alpha=0.25, color="gray")
ax[1].loglog(mids, bin_v, "o-", color="C0")
ax[1].set(xlabel="degree $k_i$", ylabel=r"$\sigma_i$ / median", title=rf"volatility: $\rho$={np.median(rho_v):+.2f}")
fig.savefig(f"data/meandeg_degree_size{tag}.png", dpi=150)
print(f"saved data/meandeg_degree_size{tag}.npz/.png")

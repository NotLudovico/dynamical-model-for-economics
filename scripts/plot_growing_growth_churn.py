"""Thesis Fig 2.2 (growing_growth_churn.png): a single mean-degree run, growth + churn.

One economy of the thesis model (mean-degree normalization, kind="powerlaw",
N=8000, C=N/40=200, mu=-1.76, sigma=1.75, lam=1e-3) at the operating point. Left:
log10 M(t) rising linearly (exponential growth, no blow-up). Right: a 40-firm sample
of relative sizes S_i = N w_i, fluctuating and exchanging rank. Teal-forward palette.

N=8000 is the floor at which every mean-degree realization persists (below it the
standard normalization can collapse); the run is illustrative and normalization-insensitive.

    uv run python scripts/plot_growing_growth_churn.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

from relative_glv import coupling, integrate, growth_rate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
OUT = os.path.join(ROOT, "figures", "growing_growth_churn.png")
plt.rcParams["axes.prop_cycle"] = cycler(color=["#2a9d8f", "#457b9d", "#e76f51", "#1d3557"])

N, C = 8000, 200                                  # mean-degree dilution C = N/40
# mu=+1.76 in the code's coupling convention (alpha = mu/C + ...); the thesis reports
# this operating point as mu=-1.76 under its competitive-is-negative sign convention.
a = coupling(N, mu=1.76, sigma=1.75, kind="powerlaw", mean_degree=C, seed=1)
r = integrate(a, tmax=400, n_eval=1000, lam=1e-3, seed=1,
              method="RK45", rtol=1e-4, atol=1e-7)

t, W, lnM = r["t"], r["W"], r["lnM"]
g_eff = growth_rate(t, lnM)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

# left: aggregate output M(t) growing
ax[0].plot(t, lnM / np.log(10), color="C0")
ax[0].set(xlabel="$t$", ylabel=r"$\log_{10} M(t)$",
          title=f"Total output ($g_{{\\mathrm{{eff}}}}={g_eff:.2f}$)")

# right: relative sizes S_i = N w_i for a random sample of firms (sustained churn)
rng = np.random.default_rng(42)
sample = rng.choice(W.shape[0], size=min(40, W.shape[0]), replace=False)
N_sim = W.shape[0]
for i in sample:
    ax[1].plot(t, np.log10(np.maximum(N_sim * W[i], 1e-8)), lw=0.6, alpha=0.7)
ax[1].set(xlabel="$t$", ylabel=r"$\log_{10} S_i$",
          title=r"Relative firm sizes $S_i = N w_i$ (40-firm sample) -- persistent churn")
ax[1].set_ylim(bottom=-4)   # hide occasional dips to the 1e-8 log-floor clip

plt.tight_layout()
plt.savefig(OUT, dpi=140)
print(f"saved {os.path.normpath(OUT)}  g_eff={g_eff:.3f}  "
      f"survivors={int((W[:, -1] > 1e-6).sum())}/{N_sim}")

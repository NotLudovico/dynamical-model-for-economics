"""Thesis Fig 3.5 (growing_conditional.png): MSB tests D1 and D3 on the own-degree GLV.

Re-plots from data/msb_conditional.npz (written by scripts/msb_conditional.py). Same binning
as before -- smoothness comes from the number of pooled economies in the .npz, not from the bins.

    uv run python scripts/plot_growing_conditional.py
"""
import os
import numpy as np
from scipy.stats import kurtosis, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "msb_conditional.npz")
OUT = os.path.join(ROOT, "..", "glv", "thesis", "growing_conditional.png")
COL = ("#1d3557", "#457b9d", "#e76f51")                        # small, mid, large

d = np.load(DATA)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

# ---- D1: rescaled-volatility collapse + bounded tail --------------------------------------
r, grp = d["d1_r"], d["d1_group"]
for k, nm, col in ((0, "small firms", COL[0]), (1, "mid firms", COL[1]), (2, "large firms", COL[2])):
    h, e = np.histogram(r[grp == k], bins=50, range=(0, 5), density=True); m = h > 0
    ax[0].semilogy(0.5 * (e[:-1] + e[1:])[m], h[m], "o-", ms=3, color=col, lw=1.4, label=nm)
ax[0].set(xlabel=r"$\sigma_i/\bar\sigma(S)$", ylabel="PDF",
          title="D1: volatility distribution collapses with size")
ax[0].legend(fontsize=10)

# ---- D3: un-mixed growth stays fat-tailed and fattens with size ---------------------------
xx = np.linspace(-8, 8, 300)
for nm, key, col in (("small", "ghat_small", COL[0]), ("mid", "ghat_mid", COL[1]),
                     ("large", "ghat_large", COL[2])):
    g = d[key]
    h, e = np.histogram(g, bins=70, range=(-8, 8), density=True); m = h > 0
    ax[1].semilogy(0.5 * (e[:-1] + e[1:])[m], h[m], "o-", ms=3, color=col, lw=1.3,
                   label=f"{nm} (exc. kurt {kurtosis(g):.0f})")
ax[1].semilogy(xx, norm.pdf(xx), "k--", lw=1, label="Gaussian")
ax[1].set(ylim=(1e-4, None), xlabel=r"$(g-\bar g)/\sigma_i$", ylabel="PDF",
          title="D3: large firms stay fat-tailed (no Gaussianization)")
ax[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"saved {os.path.normpath(OUT)}   "
      f"(D1 firms={r.size}, D3 samples/group={d['ghat_small'].size}, "
      f"economies={int(d['seeds'])})")

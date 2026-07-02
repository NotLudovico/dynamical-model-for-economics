"""Thesis Fig 3.4 (growing_multiscaling.png): MSB test D2 on the own-degree GLV.

Re-plots from data/msb_conditional.npz (written by scripts/msb_conditional.py). Left: the
first four size-conditioned volatility moments with their decline-branch power-law fits.
Right: the exponents normalized by the first, against q, vs the MSB data and the granular
(q-independent) prediction.

    uv run python scripts/plot_growing_multiscaling.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "msb_conditional.npz")
OUT = os.path.join(ROOT, "..", "glv", "thesis", "growing_multiscaling.png")
COL = ("#1d3557", "#457b9d", "#e76f51", "#2a9d8f")            # q = 1, 2, 3, 4
MSB = np.array([0.20, 0.39, 0.51, 0.58])                     # Moran-Santos-Bouchaud 2024 zeta_1..4

d = np.load(DATA)
S, M, cq = d["d2_S"], d["d2_M"], d["cq"]
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

# ---- left: the four moments E[sigma^q|S] with power-law fits ------------------------------
for q, col in zip((1, 2, 3, 4), COL):
    y = M[q - 1]; m = (S > 0) & (y > 0)
    ax[0].loglog(S[m], y[m], "o", ms=4, color=col, label=fr"$q={q}$ ($\zeta_q={cq[q-1]:.2f}$)")
    c = np.polyfit(np.log10(S[m]), np.log10(y[m]), 1)
    xx = np.array([S[m].min(), S[m].max()])
    ax[0].loglog(xx, 10 ** np.polyval(c, np.log10(xx)), "-", lw=1.2, color=col)
ax[0].set(xlabel=r"mean size $\bar S$", ylabel=r"$E[\sigma^q\,|\,S]$",
          title="size-conditioned volatility moments")
ax[0].legend(fontsize=10)

# ---- right: normalized exponents vs q (model, data, granular) -----------------------------
q = np.array([1, 2, 3, 4])
ax[1].plot(q, cq / cq[0], "o-", color="#2a9d8f", lw=1.8, ms=7, label="own-degree model")
ax[1].plot(q, MSB / MSB[0], "s--", color="#e76f51", lw=1.8, ms=7, label="MSB data")
ax[1].plot(q, np.ones_like(q), "^:", color="0.5", lw=1.8, ms=7, label="granular prediction")
ax[1].set(xlabel="moment order $q$", ylabel=r"$\zeta_q/\zeta_1$ (normalized exponent)",
          title="multiscaling: $q$-dependence", xticks=q)
ax[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"saved {os.path.normpath(OUT)}   "
      f"(zeta_q={np.round(cq, 2)}, normalized={np.round(cq / cq[0], 2)}, economies={int(d['seeds'])})")

"""Thesis Fig 3.1 (growing_stationary_msb.png): the two MSB facts in one own-degree economy.

Verbatim port of the story.ipynb "two MSB facts LIVE" cell: a single economy
(N=3000, kind="powerlaw_owndeg", mu=1.76, sigma=1.75), size-volatility relation +
rescaled growth tent. Teal-forward palette, matching the notebook.

    uv run python scripts/plot_growing_stationary_msb.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm

from relative_glv import coupling, integrate, rescale
from relative_glv.msb import size_volatility, tent_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "..", "glv", "thesis", "growing_stationary_msb.png")
MODEL, EMPIRICAL, REFERENCE, NEUTRAL = "#2a9d8f", "#e76f51", "black", "0.6"

a = coupling(3000, mu=1.76, sigma=1.75, kind="powerlaw_owndeg", seed=1)
r = integrate(a, tmax=400, n_eval=800, lam=1e-3, seed=1,
              method="RK45", rtol=1e-4, atol=1e-7)
sv = size_volatility(r["W"], r["t"], window=(320.0, 390.0), dt=0.5)
ts = tent_stats(sv["growth"].ravel())
beta_val, r2_val = sv["beta"], sv["r2"]
bowley, exk = ts["bowley"], ts["exkurt"]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# ---- panel 1: size-volatility (Fact 2) ----
ax[0].scatter(sv["Sbar"], sv["vol"], s=5, alpha=0.15, color=NEUTRAL)
bx, by, pk = sv["bin_S"], sv["bin_vol"], int(sv["pk"])
ax[0].plot(bx, by, "o-", color=MODEL, ms=5, label="binned median")
if bx.size > pk + 1:
    c = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)
    xx = np.array([bx[pk], bx[-1]])
    ax[0].plot(xx, 10 ** np.polyval(c, np.log10(xx)), color=EMPIRICAL, ls="--", lw=2,
               label=f"decline fit $\\beta={beta_val:.2f}$ ($R^2={r2_val:.2f}$)")
ax[0].set(xscale="log", yscale="log",
          xlabel=r"firm size $\bar S$", ylabel=r"growth volatility $\sigma(\bar S)$",
          title="Fact 2: size--volatility (own-degree)")
ax[0].legend(fontsize=9)

# ---- panel 2: rescaled growth distribution (Fact 1, the tent) ----
z = rescale(sv["growth"].ravel())
xx = np.linspace(-8, 8, 400)
ax[1].hist(z, bins=160, density=True, color=MODEL, alpha=0.6, label="growth rates")
ax[1].plot(xx, laplace.pdf(xx, scale=np.sqrt(2 / np.pi)), color=EMPIRICAL, ls="--", lw=1.5, label="Laplace")
ax[1].plot(xx, norm.pdf(xx), color=REFERENCE, ls=":", lw=1.2, label="Gaussian")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-4, None)
ax[1].set(xlabel=r"rescaled growth rate $\Delta\ln S$ ($\sqrt{\pi/2}\,$MAD)", ylabel="PDF",
          title=f"Fact 1: tent (Bowley $={bowley:.2f}$, exc.\\ kurt $={exk:.0f}$)")
ax[1].legend(fontsize=9)
ax[1].set_xlim(-8, 8)

plt.tight_layout()
plt.savefig(OUT, dpi=140)
print(f"saved {os.path.normpath(OUT)}  beta={beta_val:.2f} r2={r2_val:.2f} "
      f"bowley={bowley:.2f} exk={exk:.0f}")

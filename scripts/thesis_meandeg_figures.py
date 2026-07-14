"""Committed generator for the three main firm-growth thesis figures.

Reads a characterize .npz (written by notebooks/meandeg_characterize.ipynb) and emits the
thesis-named PNGs, replacing the uncommitted "session script from npz" that FIGURES.md records.
The characterize 2x2 panels are split into two 2-panel thesis figures; the conditional figure
is copied through.

  thesis/assets/figures/meandeg_stationary_msb.png  <- size-volatility decline + growth tent
  thesis/assets/figures/meandeg_multiscaling.png    <- higher moments + quenched multiscaling ratios
  thesis/assets/figures/meandeg_conditional.png     <- copy of the characterize _conditional PNG (D1/D3)

Usage:
  .venv/bin/python scripts/thesis_meandeg_figures.py            # TAG=_N16000 (default)
  MC_TAG=_N8000 .venv/bin/python scripts/thesis_meandeg_figures.py
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = os.environ.get("MC_TAG", "_N16000")
NPZ = os.path.join(ROOT, "data", f"meandeg_characterize{TAG}.npz")
COND_SRC = os.path.join(ROOT, "data", f"meandeg_characterize_conditional{TAG}.png")
OUT = os.environ.get("THESIS_OUT", os.path.join(ROOT, "thesis", "assets", "figures"))
os.makedirs(OUT, exist_ok=True)

QCOL = ["#1d3557", "#457b9d", "#e76f51", "#2a9d8f"]
d = np.load(NPZ)

# ---- Figure 1: stationary MSB facts (size-volatility law + growth tent) ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

bx, by = d["bin_S"], d["bin_vol"]
pk = int(np.argmax(by))                       # decline branch starts at the volatility peak
ax[0].loglog(bx, by, "o", ms=6, color="#457b9d", label="binned median")
ax[0].loglog(bx[pk:], by[pk:], "o", ms=6, color="#1d3557", label="decline branch (fit)")
xf = np.array([bx[pk], bx[-1]])
cf = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)
ax[0].loglog(xf, 10 ** np.polyval(cf, np.log10(xf)), "k--", lw=1.5)
ax[0].axvline(bx[pk], color="grey", ls=":", lw=1)
ax[0].set(xlabel=r"mean relative size $\bar S$", ylabel=r"volatility $\sigma$")
ax[0].legend()

z = d["tent_z"]; zz = np.linspace(-8, 8, 200)
h, e = np.histogram(z, bins=120, range=(-8, 8), density=True); m = h > 0
ax[1].semilogy(0.5 * (e[:-1] + e[1:])[m], h[m], lw=1.3, color="#e76f51", label="model")
ax[1].semilogy(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="Gaussian")
ax[1].semilogy(zz, laplace.pdf(zz, scale=np.sqrt(2 / np.pi)), "0.5", ls="--", lw=1, label="Laplace")
ax[1].set(xlabel=r"rescaled growth $g$", ylabel="PDF", ylim=(1e-4, 1))
ax[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(OUT, "meandeg_stationary_msb.png"), dpi=140)
plt.close(fig)

# ---- Figure 2: multiscaling (higher moments + quenched ratios) --------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
qs = np.array([1, 2, 3, 4])

bxd, D2M = d["d2_S"], d["d2_M"]
cq, cqe = d["cq_quenched"], d["cq_quenched_err"]
for i, q in enumerate(qs):
    ax[0].loglog(bxd, D2M[i], "o", ms=5, color=QCOL[i],
                 label=fr"$q={q}$: $\zeta_{q}={cq[i]:.2f}\pm{cqe[i]:.2f}$")
    cf = np.polyfit(np.log10(bxd), np.log10(D2M[i]), 1)
    ax[0].loglog(bxd, 10 ** np.polyval(cf, np.log10(bxd)), "--", lw=1, color=QCOL[i], alpha=0.7)
ax[0].set(xlabel=r"mean relative size $\bar S$", ylabel=r"$E[\sigma^q\mid S]$")
ax[0].legend()

r, re = d["ratio_quenched"], d["ratio_quenched_err"]
ax[1].plot(qs, [1, 1, 1, 1], "k:", lw=1.5, label="granular (flat)")
ax[1].plot(qs, qs, ls="-.", lw=1.2, color="0.55", label=r"fixed shape ($\zeta_q/\zeta_1=q$)")
ax[1].plot(qs, [1, 2.0, 2.6, 2.9], "k--", lw=1.5, label="MSB data")
ax[1].errorbar(qs, r, yerr=re, marker="s", ms=6, capsize=3, color="#e76f51",
               label="model (quenched)")
ax[1].set(xlabel="moment order $q$", ylabel=r"$\zeta_q/\zeta_1$", xticks=qs)
ax[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(OUT, "meandeg_multiscaling.png"), dpi=140)
plt.close(fig)

# ---- Figure 3: conditional tests (D1/D3) -- copied from the characterize output ----------
shutil.copyfile(COND_SRC, os.path.join(OUT, "meandeg_conditional.png"))

print(f"wrote thesis/assets/figures/meandeg_stationary_msb.png, meandeg_multiscaling.png, "
      f"meandeg_conditional.png  from {os.path.basename(NPZ)}")
print(f"  beta={float(d['beta']):.3f}+-{float(d['beta_sd']):.3f}  "
      f"ratios(quenched)={np.round(d['ratio_quenched'], 3)}  "
      f"zeta1={float(d['cq_quenched'][0]):.3f}  exk={float(d['exk']):.1f}  "
      f"bowley={float(d['bowley']):+.3f}")

"""Committed generator for the appendix finite-size figure (meandeg_scaling_topology.png).

Existing 8k/16k/32k points are the published tab:scaling values (kept fixed); the pl2.5 curve
is extended to N=2000, 4000 from data/scaling_lowN.npz so the reader sees the finite-size
dropdown. No committed generator existed before; this replaces the lost session script.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = {"fc": "#e76f51", "regular": "#6c757d", "pl3.5": "#457b9d",
       "pl2.5": "#1d3557", "pl2.2": "#2a9d8f"}
LBL = {"fc": "fully connected", "regular": "regular", "pl3.5": "power law 3.5",
       "pl2.5": "power law 2.5", "pl2.2": "power law 2.2"}

# published tab:scaling central values (8k, 16k, 32k); errors ~ the table bounds
BETA = {  # topo: {N: (beta, err)}
    "fc":      {8000: (1.02, .003), 16000: (1.02, .003)},
    "regular": {8000: (0.94, .004), 16000: (0.98, .002), 32000: (0.99, .002)},
    "pl3.5":   {8000: (0.73, .009), 16000: (0.77, .006), 32000: (0.80, .002)},
    "pl2.5":   {8000: (0.43, .013), 16000: (0.43, .012), 32000: (0.49, .020)},
    "pl2.2":   {8000: (0.16, .020), 16000: (0.22, .014), 32000: (0.23, .011)},
}
RATIO = {
    "fc":      {8000: (4.00, .010), 16000: (4.01, .008)},
    "regular": {8000: (3.87, .018), 16000: (3.95, .013), 32000: (3.90, .012)},
    "pl3.5":   {8000: (3.49, .028), 16000: (3.42, .026), 32000: (3.43, .022)},
    "pl2.5":   {8000: (3.11, .045), 16000: (3.09, .031), 32000: (3.06, .060)},
    "pl2.2":   {8000: (2.77, .140), 16000: (2.63, .140), 32000: (2.69, .044)},
}

# splice in the fresh low-N points for the topologies stable at small N.
# pl2.2 is EXCLUDED below N=8000: it freezes there (only ~7/20 survive at N=2000), and the
# survivors are biased high (beta 0.44, zeta4/1 4.27 > the fully-connected value) -- a freezing
# artifact, not the same quantity. See caption.
low = np.load(os.path.join(ROOT, "data", "scaling_lowN.npz"))
for topo in ("fc", "regular", "pl3.5", "pl2.5"):
    for N in (2000, 4000):
        k = f"{topo}_N{N}"
        if k in low.files:
            n, b, be, r, re = low[k]
            BETA[topo][N] = (b, be)
            RATIO[topo][N] = (r, re)

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))

ax[0].axhspan(0.15, 0.20, color="0.85", zorder=0, label="empirical")
for topo in ["fc", "regular", "pl3.5", "pl2.5", "pl2.2"]:
    Ns = sorted(BETA[topo]); y = [BETA[topo][N][0] for N in Ns]; e = [BETA[topo][N][1] for N in Ns]
    ax[0].errorbar(Ns, y, yerr=e, marker="o", ms=6, capsize=3, color=COL[topo], label=LBL[topo])
ax[0].set_xscale("log")
ax[0].set(xlabel=r"$N$  ($C=N/40$)", ylabel=r"$\beta$", title=r"size--variance exponent vs $N$")
ax[0].legend(fontsize=8)

ax[1].axhline(4.0, ls="-.", lw=1.2, color="0.55", label="fixed shape")
ax[1].axhline(2.9, ls="--", lw=1.2, color="k", label="MSB data")
for topo in ["fc", "regular", "pl3.5", "pl2.5", "pl2.2"]:
    Ns = sorted(RATIO[topo]); y = [RATIO[topo][N][0] for N in Ns]; e = [RATIO[topo][N][1] for N in Ns]
    ax[1].errorbar(Ns, y, yerr=e, marker="o", ms=6, capsize=3, color=COL[topo], label=LBL[topo])
ax[1].set_xscale("log")
ax[1].set(xlabel=r"$N$  ($C=N/40$)", ylabel=r"$\zeta_4/\zeta_1$",
          title=r"highest multiscaling ratio vs $N$")
ax[1].legend(fontsize=8)

# Panel 3: beta vs 1/N -> finite intercept at 1/N=0 (N->inf) = convergence.
# The graphs with the full 2k-32k range and a clean finite-size approach.
for topo in ["regular", "pl3.5", "pl2.5"]:
    Ns = np.array(sorted(BETA[topo]), float)
    y = np.array([BETA[topo][N][0] for N in Ns])
    e = np.array([BETA[topo][N][1] for N in Ns])
    x = 1.0 / Ns
    a, binf = np.polyfit(x, y, 1)
    xf = np.array([0.0, x.max() * 1.05])
    ax[2].plot(xf, binf + a * xf, "--", lw=1, color=COL[topo], alpha=0.7)
    ax[2].errorbar(x, y, yerr=e, marker="o", ms=6, capsize=3, ls="none",
                   color=COL[topo], label=fr"{LBL[topo]}: $\beta_\infty={binf:.2f}$")
    ax[2].plot(0, binf, marker="*", ms=13, color=COL[topo], zorder=5)
ax[2].axhspan(0.15, 0.20, color="0.85", zorder=0)
ax[2].set_xlim(left=-2e-5)
ax[2].set(xlabel=r"$1/N$", ylabel=r"$\beta$",
          title=r"convergence: $\beta$ vs $1/N$ ($N\to\infty$ at intercept)")
ax[2].legend(fontsize=8, loc="lower right")

plt.tight_layout()
out = os.path.join(ROOT, "thesis", "meandeg_scaling_topology.png")
fig.savefig(out, dpi=140)
print("wrote", out)
print("pl2.5 beta vs N:", {N: round(BETA['pl2.5'][N][0], 3) for N in sorted(BETA['pl2.5'])})
print("pl2.5 z4/1 vs N:", {N: round(RATIO['pl2.5'][N][0], 3) for N in sorted(RATIO['pl2.5'])})

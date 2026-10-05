"""Figures of the demographic-noise paper draft (thesis/assets/figures/demographic_*.png).
Needs: batch.py output for alpha=2.5, sigma=1.75, D=0.1 at N=32000, C=200 (BN=32000 BC=200 BPOINT=2.5,1.75,0.1),
volatility_law.py output, reduction.py and facts_plot.py figures.  Run: uv run python scripts/demographic/paper_figures.py"""
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "demographic")
FIG = os.path.join(ROOT, "thesis", "assets", "figures")
BLUE, LIGHT, ORANGE, INK, GRID = "#104281", "#86b6ef", "#eb6834", "#333333", "#e6e6e3"


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.xaxis.set_minor_formatter(NullFormatter())   # y keeps minor labels: ranges are under a decade
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def curve(x, y, nb=25):
    P = np.array_split(np.argsort(x), nb)
    return np.array([x[p].mean() for p in P]), {q: np.array([np.mean(y[p] ** q) for p in P]) for q in (1, 2, 3, 4)}


# 1. life panel at the reference point
d = np.load(os.path.join(DATA, "batch", "a2.5_s1.75_D0.1_N32000_C200.npz"))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for mg, c, mk in ((2, LIGHT, "o-"), (20, BLUE, "s-")):
    m = d["n"] >= mg
    x, M = curve(d["Sb"][m], d["v"][m])
    z = np.array([-np.polyfit(np.log(x), np.log(M[q]), 1)[0] for q in (1, 2, 3, 4)])
    ax[0].loglog(x, M[1], mk, color=c, lw=1.8, ms=3.5, label=f"lives ≥ {mg} growth rates, β = {z[0]:.2f}")
    ax[2].plot([1, 2, 3, 4], z / z[0], mk, color=c, lw=1.8, label=f"lives ≥ {mg}")
    if mg == 20:
        for q, cq in zip((1, 2, 3, 4), ("#cde2fb", "#86b6ef", "#2a78d6", BLUE)):
            y = M[q] ** (1 / q)
            ax[1].loglog(x, y / y[0], "-", color=cq, lw=1.8, label=f"q = {q}")
xr = np.array([x[0], x[-1]]); xm = np.sqrt(xr[0] * xr[1])
ax[0].loglog(xr, 1.25 * np.interp(np.log(xm), np.log(x), M[1]) * (xr / xm) ** -0.2, "--", color=INK, lw=1, label="slope −0.2")
ax[2].plot([1, 2, 3, 4], [1, 2.0, 2.6, 2.9], "D--", color=ORANGE, lw=1.5, label="Compustat [MSB]")
ax[2].plot([1, 4], [1, 4], ":", color=INK, lw=1, label="no multiscaling")
ax[0].set(xlabel=r"life-mean size $\bar S$", ylabel=r"$E[v\,|\,\bar S]$", title="(a) volatility vs size")
ax[1].set(xticks=[0.2, 0.5, 1, 2, 5], xticklabels=["0.2", "0.5", "1", "2", "5"], xlabel=r"$\bar S$", ylabel=r"$E[v^q|\bar S]^{1/q}$ / first bin", title="(b) moments, lives ≥ 20")
ax[2].set(xlabel="q", ylabel=r"$\zeta_q/\zeta_1$", xticks=[1, 2, 3, 4], ylim=(0.5, 6), title="(c) ratio curve")
for a in ax:
    style(a); a.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "demographic_life_panel.png"), dpi=200); plt.close()

# 2. volatility law
R = np.load(os.path.join(DATA, "volatility_law.npz"))["rows"]
P = np.array_split(np.argsort(R[:, 0]), 10)
sb = np.array([R[p, 0].mean() for p in P]); m = np.array([R[p].mean(0) for p in P])
fig, ax = plt.subplots(figsize=(5, 3.8))
ax.loglog(sb, np.sqrt(m[:, 1]), "o-", color=BLUE, lw=2, label="measured sd(g)")
ax.loglog(sb, np.sqrt(m[:, 2] + m[:, 3]), "s--", color=ORANGE, lw=2, label=r"$\sqrt{V_F + B}$")
ax.loglog(sb, np.sqrt(m[:, 2]), ":", color=INK, lw=1.5, label=r"fitness part $\sqrt{V_F}$")
ax.loglog(sb, np.sqrt(m[:, 3]), "-.", color="#2a78d6", lw=1.5, label=r"noise part $\sqrt{B}$")
ax.set(xticks=[0.2, 0.5, 1, 2], xticklabels=["0.2", "0.5", "1", "2"], xlabel=r"life-mean size $\bar S$", ylabel="growth volatility", title="volatility law, lives ≥ 20 growth rates")
style(ax); ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "demographic_volatility_law.png"), dpi=200); plt.close()

# 3, 4. figures written by reduction.py and facts_plot.py
for src, dst in (("reduction.png", "demographic_reduction.png"), ("facts.png", "demographic_facts.png")):
    if os.path.exists(os.path.join(DATA, src)):
        shutil.copy(os.path.join(DATA, src), os.path.join(FIG, dst))
print("figures written to", FIG)

"""Phase diagram figure, theory first with simulation as evidence.
(a) alpha = 2.5, N = 4000: regions from the HDMFT (unique fixed point below sigma_c, growing or
shrinking side of the zero-growth line), with the simulated g_eff = 0 points, the simulated churn = 0.035
onset per mu column (the old freezing line), cells whose integration failed, and the operating point.
(b) Stability line for alpha = 2.2-6 (60-class HDMFT, SCM kernel, lam = 0) with the exact bilinear
relation (Eq. sigmac of the note, node degrees) and the fully connected limit.
(c) Above the line: fraction of runs chaotic against sigma - sigma_c (ten graphs per point).
Needs data/phase_meandeg_fine_24x20_10s.npz, data/phase_theory_lines.npz,
data/phase_coexistence.npz, thesis/standalone/zero_growth_theory/zero_growth_line.npz.
Run: uv run python scripts/phase/plot_phase_theory.py  -> thesis/assets/figures/phase_theory.png"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from phase_sigma_c_bilinear import sigma_c_bilinear

D = lambda *p: os.path.join(ROOT, *p)
OUT = D("thesis", "assets", "figures", "phase_theory.png")
REG = ["#fde6cf", "#d6e6f2", "#e9965f", "#3d7ea6"]       # FP shrinking, FP growing, chaos shrinking, chaos growing
LINE = {2.2: "#9b2226", 2.5: "#ca6702", 3.5: "#0a9396", 6.0: "#005f73"}


def panel_a(ax):
    fam = np.load(D("data", "phase_theory_lines.npz"))
    zg = np.load(D("thesis", "standalone", "zero_growth_theory", "zero_growth_line.npz"))
    o = np.argsort(-fam["mus"]); mu_sc, s_sc = -fam["mus"][o], fam["sigma_c_a25_lam1e3"][o]
    oz = np.argsort(zg["sigma"]); s_z, mu_z = zg["sigma"][oz], zg["mu_thesis"][oz]
    mu = np.linspace(-2.5, 0.0, 600); sg = np.linspace(0.35, 1.95, 500)
    M, Sg = np.meshgrid(mu, sg)
    above = Sg > np.interp(M, mu_sc, s_sc)
    grow = M > np.interp(Sg, s_z, mu_z)
    ax.pcolormesh(mu, sg, 2 * above + grow, cmap=ListedColormap(REG), vmin=-0.5, vmax=3.5, shading="auto",
                  rasterized=True)
    ax.plot(mu_sc, s_sc, color="black", lw=2.4)
    c = zg["controlled"][oz].astype(bool)
    ax.plot(mu_z[c], s_z[c], color="white", lw=2.2)
    ax.plot(mu_z[~c], s_z[~c], color="white", lw=2.2, ls=(0, (4, 2)))
    ax.scatter(zg["measured_mu_thesis"], zg["measured_sigma"], s=18, facecolor="white", edgecolor="black", lw=0.8,
               zorder=5, label=r"simulated $g_{\rm eff}=0$")
    d = np.load(D("data", "phase_meandeg_fine_24x20_10s.npz"))
    gs, gm, raw = d["sigmas"], -d["mus"], d["raw"]
    og = np.argsort(gm)
    with np.errstate(all="ignore"):
        churn = np.nanmedian(raw[..., 0], 2)[:, og]
    fail = np.isfinite(raw[..., 0]).mean(2)[:, og] < 0.5
    churn = np.where(fail, 1.0, churn)
    first = np.array([gs[np.argmax(col >= 0.035)] if (col >= 0.035).any() else np.nan for col in churn.T])
    edge = first <= gs[0] + 1e-9                       # already above the cutoff at the lowest simulated sigma
    ax.scatter(gm[og][~edge], first[~edge], marker="v", s=22, color="#444444", zorder=5,
               label="simulated churn $>0.035$ from here (old freezing line)")
    ax.scatter(gm[og][edge], first[edge], marker="v", s=22, facecolor="none", edgecolor="#444444", zorder=5,
               label="... already at the lowest simulated $\\sigma$")
    fm, fs = np.meshgrid(gm[og], gs)
    ax.scatter(fm[fail], fs[fail], marker="x", s=14, color="#222222", lw=0.8, zorder=4, label="integration failed")
    ax.scatter([-1.76], [1.75], marker="*", s=330, color="#ffd166", edgecolor="black", lw=0.8, zorder=6,
               label="operating point")
    ax.plot([], [], color="black", lw=2.4, label=r"HDMFT $\sigma_c(\mu)$")
    ax.plot([], [], color="#9aa", lw=2.2, label=r"HDMFT $g^*=0$ (dashed: continuation)")
    t = dict(fontsize=10, ha="center", va="center")
    ax.text(-1.95, 1.18, "unique fixed point\nshrinking", color="#7a4a1f", **t)
    ax.text(-0.42, 0.45, "unique f.p.\ngrowing", color="#1d4e6e", fontsize=9, ha="center", va="center")
    ax.text(-2.22, 1.70, "chaos / trapping\nshrinking", color="white", weight="bold", fontsize=8.5, ha="center", va="center", rotation=-14)
    ax.text(-0.8, 1.83, "chaos / trapping, growing", color="white", weight="bold", **t)
    ax.set_xlim(-2.5, 0.0); ax.set_ylim(0.35, 1.95)
    ax.set_xlabel(r"mean interaction $\mu$  ($\mu<0$ competitive)"); ax.set_ylabel(r"disorder $\sigma$")
    ax.set_title(r"(a) phase diagram, $\alpha=2.5$", fontsize=11, loc="left")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)


def panel_b(ax):
    import relative_glv.model as Mo
    fam = np.load(D("data", "phase_theory_lines.npz"))
    mus = np.linspace(0.0, -2.5, 21)
    for i, al in enumerate(fam["alphas"]):
        al = float(al)
        ax.plot(-fam["mus"], fam["sigma_c"][i], color=LINE[al], lw=2, label=rf"$\alpha={al:g}$")
        if al >= 3.5:                                  # the bilinear kernel is a valid probability only here
            Mo._ALPHA_PL = al
            k = np.diff(Mo.coupling(4000, mu=0.0, sigma=1.0, kind="powerlaw", seed=0, mean_degree=100).indptr).astype(float)
            ax.plot(mus, [sigma_c_bilinear(m, k) for m in mus], color="black", lw=1, ls=(0, (3, 2)))
    ax.plot([], [], color="black", lw=1, ls=(0, (3, 2)), label="closed form, bilinear kernel")
    ax.axhline(np.sqrt(2), color="grey", ls="--", lw=1.2, label=r"regular graph, $\sqrt{2}$")
    ax.set_xlim(-2.5, 0.0); ax.set_ylim(0.7, 1.8)
    ax.set_xlabel(r"mean interaction $\mu$"); ax.set_ylabel(r"$\sigma_c$")
    ax.set_title("(b) stability line by degree exponent", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")


def panel_c(ax):
    d = np.load(D("data", "phase_coexistence.npz"))
    A, ds = d["amp"], np.array(d["ds"])
    for i, (m, col) in enumerate(zip(d["mus"], ("#005f73", "#ca6702"))):
        chaos = (A[i, :, :, 1] >= 1e-2).mean(0)
        err = np.sqrt(chaos * (1 - chaos) / A.shape[1])
        ax.errorbar(ds, chaos, yerr=err, color=col, marker="o", ms=5, capsize=3, lw=1.6, label=rf"$\mu={-m:.2f}$")
    ax.set_xlabel(r"$\sigma-\sigma_c$  (each graph's own HDMFT $\sigma_c$)")
    ax.set_ylabel("fraction of runs chaotic at $T=1600$")
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(-0.01, 0.21)
    ax.set_title("(c) above the line: chaos coexists\n     with trapping equilibria", fontsize=10, loc="left")
    ax.legend(fontsize=8.5, loc="upper left")


def main():
    fig = plt.figure(figsize=(15, 5.0))
    g = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1])
    panel_a(fig.add_subplot(g[0])); panel_b(fig.add_subplot(g[1])); panel_c(fig.add_subplot(g[2]))
    fig.tight_layout(); fig.savefig(OUT, dpi=170); print("wrote", OUT)


if __name__ == "__main__":
    main()

"""Figure: spell-panel statistics of the full model against the empirical facts.
(a) E[v | Sbar] over all spells (equal-count bins, pooled economies), with the fit range of the frozen
estimator (larger half of spells) marked and a slope of -0.2 for reference.  (b) zeta_q / zeta_1 with
the frozen estimator (msb.upper_zeta, mean over economies with spread), the data and the fixed-shape line.
Needs data/headline_spells/*.npz (scripts/headline_spells.py).
Run: uv run python scripts/plot_data_comparison.py  -> thesis/assets/figures/data_comparison.png"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from relative_glv.msb import upper_zeta

OUT = os.path.join(ROOT, "thesis", "assets", "figures", "data_comparison.png")
POINTS = (("a6.0_s1.75_l1e-03", r"$\alpha=6$, $\sigma=1.75$", "#005f73"),
          ("a3.5_s1.9_l1e-03", r"$\alpha=3.5$, $\sigma=1.9$", "#0a9396"),
          ("a3.5_s1.9_l1e-02", r"$\alpha=3.5$, $\sigma=1.9$, $\lambda=10^{-2}$", "#94d2bd"),
          ("a2.5_s1.75_l1e-03", r"$\alpha=2.5$, $\sigma=1.75$", "#ca6702"))
DATA_RATIO = (1.0, 2.0, 2.6, 2.9)


def main():
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    q = np.arange(1, 5)
    for tag, lab, col in POINTS:
        d = np.load(os.path.join(ROOT, "data", "headline_spells", tag + ".npz"))
        S, v, e = d["Sbar"], d["vol"], d["economy"]
        ok = (v > 0) & (S > 0); S, v, e = S[ok], v[ok], e[ok]
        o = np.argsort(S); P = np.array_split(o, 30)
        ax[0].plot([S[p].mean() for p in P], [v[p].mean() for p in P], "o-", color=col, ms=3.5, lw=1.4, label=lab)
        ax[0].axvline(np.median(S), color=col, lw=0.8, ls=":")
        Z = np.array([upper_zeta(S[e == i], v[e == i]) for i in np.unique(e)])
        R = Z / Z[:, :1]
        ax[1].errorbar(q, R.mean(0), yerr=R.std(0), color=col, marker="o", ms=4, capsize=3,
                       label=rf"{lab}: $\beta={np.median(Z[:, 0]):.2f}\pm{Z[:, 0].std():.2f}$")
        print(f"{tag}: beta per economy {np.round(Z[:, 0], 3)}  ratios {np.round(R.mean(0)[1:], 2)}")
    s = np.geomspace(0.5, 20, 10)
    ax[0].plot(s, 0.6 * s ** -0.2, color="black", ls="--", lw=1, label="slope $-0.2$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"mean spell size $\bar S$"); ax[0].set_ylabel(r"$\mathbb{E}[v\mid\bar S]$")
    ax[0].set_title("(a) volatility against size, all spells (dotted: median size)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].plot(q, q, color="grey", ls=":", lw=1.2, label=r"fixed shape, $\zeta_q=q\zeta_1$")
    ax[1].plot(q, DATA_RATIO, "s--", color="black", ms=6, label="Compustat (Moran et al.)")
    ax[1].set_xlabel("$q$"); ax[1].set_ylabel(r"$\zeta_q/\zeta_1$"); ax[1].set_xticks(q)
    ax[1].set_title("(b) multiscaling, larger half of spells", fontsize=10)
    ax[1].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT, dpi=160); print("wrote", OUT)


if __name__ == "__main__":
    main()

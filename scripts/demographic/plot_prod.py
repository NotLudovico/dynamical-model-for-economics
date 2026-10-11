"""Figure: the MSB facts in the production run (robustness.py prod: alpha 2.5, sigma 1.70, D 0.1,
N 32000, C 200, eight economies, those in a collective burst excluded: robustness.kept), from data/demographic/robustness/prod_seed<k>.npz.
(a) E[v | Sbar] for three minimum life lengths, with a slope -0.2 guide.  (b) Ratio curves zeta_q/zeta_1
against Compustat.  (c) Distribution of v / E[v | Sbar] by size fifth (lives >= 10), with the Compustat
tail index.  (d) Excess kurtosis of rescaled growth and the k = 3 persistence ratio by size fifth.
(e) Correlation with the aggregate and pairwise correlation within the fifth.
Run: uv run python scripts/demographic/plot_prod.py [label]   (default label prod)
  -> thesis/assets/figures/demographic_prod.png
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from relative_glv.msb import zeta_bins
from robustness import path, kept

LABEL = sys.argv[1] if len(sys.argv) > 1 else "prod"
OUT = os.path.join(ROOT, "thesis", "assets", "figures", f"demographic_{LABEL}.png")
INK, MUTED, GRID, GREEN = "#0b0b0b", "#52514e", "#e4e3df", "#1baf7a"
NMIN = {2: "#9ec5f0", 5: "#2a78d6", 20: "#14467f"}
FIFTH = ["#c7dcf5", "#8ab8ec", "#4f93e0", "#2a6fc0", "#163f73"]
MSB_RATIO = np.array([1.0, 1.95, 2.55, 2.90])


def style(ax):
    from matplotlib.ticker import NullFormatter, FormatStrFormatter
    if ax.get_xscale() == "log":
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    if ax.get_yscale() == "log" and ax.get_ylim()[0] > 0.05:
        ax.yaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_yticks([0.3, 0.4, 0.5, 0.7, 1, 1.5])
    ax.grid(True, which="major", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ds = kept([np.load(path(LABEL, s)) for s in range(16) if os.path.exists(path(LABEL, s))])
    cat = lambda k: np.concatenate([d[k] for d in ds])
    Sb, v, n = cat("Sbar").astype(float), cat("v").astype(float), cat("n")
    m = lambda k: np.mean([d[k] for d in ds], 0); sdv = lambda k: np.std([d[k] for d in ds], 0)
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10})
    fig, axs = plt.subplots(1, 5, figsize=(21, 4.3))
    a, b, c, d, e = axs
    qq = np.arange(1, 5)
    b.plot(qq, qq, "-", color=MUTED, lw=1, label=r"$\zeta_q=q\zeta_1$")
    for nm, col in NMIN.items():
        sel = (n >= nm) & (v > 0)
        P = np.array_split(np.argsort(Sb[sel]), 25)
        x = np.array([Sb[sel][p].mean() for p in P]); y = np.array([v[sel][p].mean() for p in P])
        z = zeta_bins(Sb[sel], v[sel])[0]
        a.loglog(x, y, "o-", color=col, lw=1.8, ms=4, mec="white", mew=0.5,
                 label=rf"$n_{{\min}}={nm}$: $\zeta_1={z[0]:.2f}$")
        b.plot(qq, z / z[0], "o-", color=col, lw=1.8, ms=5, mec="white", label=rf"$n_{{\min}}={nm}$")
    xg = np.geomspace(0.3, 3, 10)
    a.loglog(xg, 0.55 * xg ** -0.2, ":", color=GREEN, lw=1.5, label=r"slope $-0.2$")
    a.set(xticks=[0.01, 0.1, 1], xlabel=r"life-mean size $\bar S$", ylabel=r"$E[v\mid\bar S]$", title="(a) size--volatility")
    b.plot(qq, MSB_RATIO, "*", color=GREEN, ms=12, label="Compustat")
    b.set(xlabel=r"$q$", ylabel=r"$\zeta_q/\zeta_1$", title="(b) ratio curves", ylim=(0.8, 6.2))
    sel = (n >= 10) & (v > 0)
    P = np.array_split(np.argsort(Sb[sel]), 25); vv = v[sel]
    r = np.empty(vv.size); grp = np.empty(vv.size, int)
    for i, p in enumerate(P):
        r[p] = vv[p] / vv[p].mean(); grp[p] = i // 5
    for f in range(5):
        x = np.sort(r[grp == f])[::-1]
        c.loglog(x, np.arange(1, x.size + 1) / x.size, "-", color=FIFTH[f], lw=1.6, label=f"size fifth {f + 1}")
    xg = np.geomspace(1.5, 4, 10)
    c.loglog(xg, 0.3 * (xg / 1.5) ** -4.6, ":", color=GREEN, lw=1.5, label=r"Compustat tail, $b=4.6$")
    c.set(xlabel=r"$v/E[v\mid\bar S]$", ylabel="complementary CDF", title=r"(c) volatility dispersion, $n_{\min}=10$",
          xlim=(0.3, 6), ylim=(1e-5, 1.5), xticks=[0.3, 0.5, 1, 2, 4])
    f5 = np.arange(1, 6)
    d.errorbar(f5, m("kurt"), sdv("kurt"), fmt="o-", color="#2a78d6", lw=1.8, capsize=3, label="excess kurtosis, 1 year")
    d2 = d.twinx()
    d2.errorbar(f5, m("persist")[:, 1], sdv("persist")[:, 1], fmt="s--", color="#eb6834", lw=1.6, capsize=3,
                label="3-year kurtosis / iid value")
    d2.axhline(1, color="#eb6834", lw=0.8, ls=":")
    d.set(xlabel="size fifth", ylabel="excess kurtosis of rescaled growth", title="(d) tails and persistence")
    d2.set_ylabel("3-year kurtosis over iid value", color="#eb6834")
    h1, l1 = d.get_legend_handles_labels(); h2, l2 = d2.get_legend_handles_labels()
    d.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
    e.errorbar(f5, m("corr_agg"), sdv("corr_agg"), fmt="o-", color="#2a78d6", lw=1.8, capsize=3, label="with the aggregate")
    e.errorbar(f5, m("corr_pair"), sdv("corr_pair"), fmt="s--", color="#14467f", lw=1.6, capsize=3, label="pairwise, same fifth")
    e.set(xlabel="size fifth", ylabel="correlation of growth rates", title="(e) co-movement")
    for ax in axs:
        style(ax)
        if ax is not d:
            ax.legend(frameon=False, fontsize=7.5)
    plt.tight_layout(); plt.savefig(OUT, dpi=200)
    print(f"{len(ds)} economies; wrote {OUT}")


if __name__ == "__main__":
    main()

"""Figure: retuning sigma and D at N = 16000 (scripts/demographic/robustness.py retune).
(a) beta (lives >= 5, all bins) on the (sigma, D) grid.  (b) zeta_q for the old reference point and
the two best points against MSB.  (c) E[v | Sbar] for the same points.  (d) Local exponent by fifths.
Run: uv run python scripts/demographic/plot_retune.py -> thesis/assets/figures/demographic_retune.png"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from robustness import RETUNE, path
from relative_glv.msb import zeta_bins
from plot_robustness import pooled, curve, style, INK, MUTED

OUT = os.path.join(ROOT, "thesis", "assets", "figures", "demographic_retune.png")
MSB_ZETA = np.array([0.20, 0.39, 0.51, 0.58])
SHOW = (("rt_s1.75_D0.1", r"old: $\sigma=1.75$, $D=0.1$", "#9a9890"),
        ("rt_s1.7_D0.1", r"$\sigma=1.70$, $D=0.1$", "#2a78d6"),
        ("rt_s1.72_D0.15", r"$\sigma=1.72$, $D=0.15$", "#eb6834"))


def load(label):
    seeds = next(s for l, _, s in RETUNE if l == label)
    return [np.load(path(label, s)) for s in seeds if os.path.exists(path(label, s))]


def main():
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    a, b, c, d = axs
    for sg, col in ((1.70, "#2a78d6"), (1.72, "#eb6834")):
        Ds, bs = [], []
        for D in (0.1, 0.15, 0.2):
            ds = load(f"rt_s{sg}_D{D:g}")
            z = zeta_bins(*pooled(ds, 5))[0]
            Ds.append(D); bs.append(z[0])
        a.plot(Ds, bs, "o-", color=col, lw=2, ms=9, mec="white", mew=1.5, label=rf"$\sigma={sg:.2f}$")
    z0 = zeta_bins(*pooled(load("rt_s1.75_D0.1"), 5))[0][0]
    a.plot([0.1], [z0], "o", color="#9a9890", ms=9, mec="white", mew=1.5, label=r"$\sigma=1.75$ (old)")
    a.axhline(0.2, color=INK, lw=1.4, ls=(0, (6, 3)), label=r"Compustat $\beta=0.20$")
    a.set_xticks([0.1, 0.15, 0.2]); a.set_xlabel("noise unit $D$"); a.set_ylabel(r"$\beta$ (lives $\geq5$, all bins)")
    a.set_title(r"(a) $\beta$ at $N=16$k", loc="left"); a.legend(fontsize=8, frameon=False)
    q = np.arange(1, 5)
    for lab, name, col in SHOW:
        Sb, v = pooled(load(lab), 5)
        z, _, _, _, _, loc = zeta_bins(Sb, v)
        b.plot(q, z, "o-", color=col, lw=2, ms=8, mec="white", mew=1.5, label=name)
        cs, cv = curve(Sb, v)
        c.plot(cs, cv, "o-", color=col, lw=2, ms=4, mec="white", mew=0.8, label=name)
        P = np.array_split(np.argsort(Sb), 25)
        cen = [np.log10(np.mean([Sb[p].mean() for p in P[i:i + 5]])) for i in range(0, 25, 5)]
        d.plot(cen, loc, "o-", color=col, lw=2, ms=8, mec="white", mew=1.5, label=name)
    b.plot(q, MSB_ZETA, "D-", color=INK, lw=2, ms=8, label="Compustat (MSB 2024)")
    b.set_xticks(q); b.set_xlabel("$q$"); b.set_ylabel(r"$\zeta_q$ (lives $\geq5$)")
    b.set_title(r"(b) $\zeta_q$ against Compustat", loc="left"); b.legend(fontsize=8, frameon=False)
    s = np.geomspace(1e-3, 1e1, 10)
    c.plot(s, 1.218 * (s / 1e-3) ** -0.2, color=INK, lw=1.6, ls=(0, (6, 3)), label=r"Compustat: level, slope $-0.2$")
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlabel(r"life-mean size $\bar S$"); c.set_ylabel(r"$\mathbb{E}[v\mid\bar S]$")
    c.set_title("(c) volatility against size", loc="left"); c.legend(fontsize=8, frameon=False, loc="lower left")
    d.axhline(0.2, color=INK, lw=1.4, ls=(0, (6, 3)), label=r"Compustat $\beta=0.2$")
    d.set_xlabel(r"$\log_{10}\bar S$ (centre of each fifth)"); d.set_ylabel("local exponent")
    d.set_title("(d) the mid-range dip is shallower (0.12 vs 0.07)", loc="left"); d.legend(fontsize=8, frameon=False)
    for ax in axs:
        style(ax)
    fig.tight_layout(); fig.savefig(OUT, dpi=150); print("wrote", OUT)


if __name__ == "__main__":
    main()

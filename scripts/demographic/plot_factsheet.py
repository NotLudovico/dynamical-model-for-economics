"""Figure: the demographic-noise model against the firm-growth facts at the reference point
(alpha 2.5, sigma 1.75, D 0.1, N 8000, step 0.5, 4 economies).
(a) E[v | Sbar] by minimum life length, with the Compustat level and slope.  (b) zeta_q/zeta_1.
(c) Local exponent by fifths of the 25 bins.  (d) Excess kurtosis of rescaled growth by size fifth,
1-step and 3-step against the iid value.  (e) Correlation of a firm's absolute growth with the
aggregate by size fifth.  (f) Size distribution of all firms (density of log10 S) and its CCDF
against a Zipf slope.
Needs data/demographic/factsheet/*.npz (scripts/demographic/factsheet.py).
Run: uv run python scripts/demographic/plot_factsheet.py -> thesis/assets/figures/demographic_factsheet.png"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from factsheet import path, lives, zetas, exkurt, SEEDS, DT

OUT = os.path.join(ROOT, "thesis", "assets", "figures", "demographic_factsheet.png")
P = (2.5, 1.75, 0.1)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = {2: "#9ec5f0", 5: "#5b9be6", 10: "#2a78d6", 20: "#14467f"}       # one hue, light -> dark
DATA_RATIO = (1.0, 2.0, 2.6, 2.9)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def main():
    data = [np.load(path(*P, s)) for s in range(SEEDS)]
    allv = [x for d in data for x in lives(d["S"].astype(float), d["L"], 1)]
    Sb = np.array([x[0] for x in allv]); v = np.array([x[1] for x in allv]); n = np.array([x[2] for x in allv])
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    a, b, c, d, e, f = axs.flat
    q = np.arange(1, 5)
    for nm, col in SEQ.items():
        sel = n >= nm
        z, loc, bs, bv, dec = zetas(Sb[sel], v[sel])
        a.plot(bs, bv, "o-", color=col, lw=2, ms=5, mec="white", mew=1, label=rf"lives $\geq{nm}$: $\beta={z[0]:.2f}$")
        b.plot(q, z / z[0], "o-", color=col, lw=2, ms=8, mec="white", mew=1.5, label=rf"lives $\geq{nm}$")
        P5 = np.array_split(np.argsort(Sb[sel]), 25)
        cen = [np.log10(np.mean([Sb[sel][p].mean() for p in P5[i:i + 5]])) for i in range(0, 25, 5)]
        c.plot(cen, loc, "o-", color=col, lw=2, ms=8, mec="white", mew=1.5, label=rf"lives $\geq{nm}$")
    s = np.geomspace(1e-3, 1e1, 10)
    a.plot(s, 1.218 * (s / 1e-3) ** -0.2, color=INK, lw=1.6, ls=(0, (6, 3)), label=r"Compustat: level, slope $-0.2$")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"life-mean size $\bar S$ (model: relative to all firms; data: listed firms)")
    a.set_ylabel(r"$\mathbb{E}[v\mid\bar S]$"); a.set_title("(a) volatility against size (step 0.5)", loc="left")
    a.legend(fontsize=8, frameon=False, loc="lower left")
    b.plot(q, q, color=MUTED, lw=1.2, ls=":", label=r"fixed shape, $\zeta_q=q\zeta_1$")
    b.plot(q, DATA_RATIO, "D-", color=INK, lw=2, ms=8, label="Compustat (MSB 2024)")
    b.set_xticks(q); b.set_xlabel("$q$"); b.set_ylabel(r"$\zeta_q/\zeta_1$")
    b.set_title("(b) ratio curve, all bins", loc="left"); b.legend(fontsize=8, frameon=False, loc="upper left")
    c.axhline(0.2, color=INK, lw=1.4, ls=(0, (6, 3)), label=r"Compustat $\beta\approx0.2$")
    c.set_xlabel(r"$\log_{10}\bar S$ (centre of each fifth)"); c.set_ylabel("local exponent")
    c.set_title("(c) local slope by fifths", loc="left"); c.legend(fontsize=8, frameon=False)
    # (d) Gaussianisation and persistence, lives >= 20
    L20 = [x for x in allv if x[2] >= 20]
    sb20 = np.array([x[0] for x in L20]); fifth = np.digitize(sb20, np.quantile(sb20, [0.2, 0.4, 0.6, 0.8]))
    k1, k3, cen = [], [], []
    for f5 in range(5):
        res = [x[3] / x[1] for x, ff in zip(L20, fifth) if ff == f5]
        z1 = np.concatenate(res)
        z3 = np.concatenate([np.convolve(rr, np.ones(3), "valid")[::3] for rr in res if rr.size >= 3])
        k1.append(exkurt(z1)); k3.append(exkurt(z3 / z3.std())); cen.append(np.log10(np.median(sb20[fifth == f5])))
    d.plot(cen, k1, "o-", color="#2a78d6", lw=2, ms=8, mec="white", mew=1.5, label="1-step growth")
    d.plot(cen, k3, "s-", color="#eb6834", lw=2, ms=8, mec="white", mew=1.5, label="3-step growth")
    d.plot(cen, np.array(k1) / 3, "s:", color="#eb6834", lw=1.6, ms=7, mfc="white", label="3-step if growth were iid")
    d.set_xlabel(r"$\log_{10}\bar S$ (median of each size fifth, lives $\geq20$)"); d.set_ylabel("excess kurtosis of rescaled growth")
    d.set_title("(d) tails fatten with size; persistence only at the top", loc="left"); d.legend(fontsize=8, frameon=False)
    # (e) correlation with the aggregate
    cS, cc = [], []
    for dd in data:
        Gk = dd["fbar"][:-1] * DT
        for x in lives(dd["S"].astype(float), dd["L"], 1):
            if x[2] < 20:
                continue
            agg = Gk[x[6]:x[6] + x[2]]
            if agg.size == x[2] and agg.std() > 0:
                cS.append(x[0]); cc.append(np.corrcoef(x[4] + agg, agg)[0, 1])
    cS, cc = np.array(cS), np.array(cc)
    qq = np.digitize(cS, np.quantile(cS, [0.2, 0.4, 0.6, 0.8]))
    e.plot([np.log10(np.median(cS[qq == k])) for k in range(5)], [np.nanmean(cc[qq == k]) for k in range(5)],
           "o-", color="#2a78d6", lw=2, ms=8, mec="white", mew=1.5, label="model, lives $\\geq20$")
    e.set_xlabel(r"$\log_{10}\bar S$ (median of each size fifth)"); e.set_ylabel("corr(own growth, aggregate growth)")
    e.set_title("(e) co-movement with the aggregate rises with size", loc="left"); e.legend(fontsize=8, frameon=False)
    # (f) size distribution
    allS = np.concatenate([dd["S"][:, ::5].ravel() for dd in data]).astype(float)
    lx = np.log10(allS[allS > 0])
    h, edges = np.histogram(lx, bins=80, density=True)
    f.plot(0.5 * (edges[1:] + edges[:-1]), h, color="#2a78d6", lw=2, label=r"density of $\log_{10}S$, all firms")
    zt = np.sort(allS)[::-1]; k = int(0.01 * zt.size); hill = 1.0 / np.mean(np.log(zt[:k] / zt[k]))
    f.axvline(np.log10(P[2]), color=MUTED, lw=1, ls=":")
    f.text(np.log10(P[2]) - 0.08, h.max() * 0.55, "entry size D ", color=MUTED, fontsize=8.5, ha="right")
    f.text(0.03, 0.80, f"Hill index of the top 1%: {hill:.1f}\n(Zipf: about 1)", transform=f.transAxes,
           ha="left", va="top", fontsize=8.5, color=INK)
    f.set_xlim(-4, 1.8)
    f.set_xlabel(r"$\log_{10} S$"); f.set_ylabel("density")
    f.set_title("(f) sizes: newborn mode and plateau mode, no Zipf tail", loc="left")
    f.legend(fontsize=8, frameon=False, loc="upper left")
    for ax in axs.flat:
        style(ax)
    fig.tight_layout(); fig.savefig(OUT, dpi=150); print("wrote", OUT)


if __name__ == "__main__":
    main()

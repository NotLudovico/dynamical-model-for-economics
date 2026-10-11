"""Figure: the MSB 2024 facts on the demographic-noise model (reference point), from
data/demographic/msb_facts.npz (scripts/demographic/msb_facts.py).
(a) zeta_q against MSB.  (b) E[v | Sbar] with the Compustat level and slope.  (c) Collapse of
sigma/sigmabar across size fifths.  (d) Tails of rescaled growth by size fifth.  (e) Persistence:
k-step excess kurtosis over its iid value.  (f) Co-movement by size fifth.
Run: uv run python scripts/demographic/plot_msb_facts.py -> thesis/assets/figures/demographic_msb_facts.png"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = os.path.join(ROOT, "data", "demographic", "msb_facts.npz")
OUT = os.path.join(ROOT, "thesis", "assets", "figures", "demographic_msb_facts.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIFTH = ["#c7dcf5", "#8ab8ec", "#4f93e0", "#2a6fc0", "#163f73"]          # one hue, small -> large
CAT = {"MAD 2": "#9ec5f0", "MAD 5": "#2a78d6", "MAD 20": "#14467f", "SD 5": "#eb6834"}
MSB_ZETA = np.array([0.20, 0.39, 0.51, 0.58])


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def main():
    d = np.load(IN)
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    a, b, c, e, f, g = axs.flat
    q = np.arange(1, 5)
    for key, col in CAT.items():
        lab, nm = key.split()
        z = d[f"zeta_{lab}_{nm}"]
        a.plot(q, z, "o-", color=col, lw=2, ms=8, mec="white", mew=1.5, label=f"{lab}, lives $\\geq{nm}$")
        if lab == "MAD" and nm in ("2", "5", "20"):
            bs, bv = d[f"bins_{lab}_{nm}"]
            b.plot(bs, bv, "o-", color=col, lw=2, ms=5, mec="white", mew=1, label=f"lives $\\geq{nm}$")
    a.plot(q, MSB_ZETA, "D-", color=INK, lw=2, ms=8, label="Compustat (MSB 2024)")
    a.set_xticks(q); a.set_xlabel("$q$"); a.set_ylabel(r"$\zeta_q$")
    a.set_title(r"(a) F1: $\zeta_q$, all 25 bins", loc="left"); a.legend(fontsize=8, frameon=False)
    s = np.geomspace(1e-3, 1e1, 10)
    b.plot(s, 1.218 * (s / 1e-3) ** -0.2, color=INK, lw=1.6, ls=(0, (6, 3)), label=r"Compustat: level, slope $-0.2$")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(r"life-mean size $\bar S$ (model: all firms; data: listed firms)"); b.set_ylabel(r"$\mathbb{E}[v\mid\bar S]$ (MAD)")
    b.set_title("(b) F2: level; the largest bin falls below the line", loc="left"); b.legend(fontsize=8, frameon=False, loc="lower left")
    r, grp = d["ratio_20"], d["group_20"]
    edges = np.linspace(-2.5, 1.5, 60)
    for k in range(5):
        h, _ = np.histogram(np.log(r[grp == k]), bins=edges, density=True)
        c.plot(0.5 * (edges[1:] + edges[:-1]), h, color=FIFTH[k], lw=2, label=f"size fifth {k + 1}")
    c.set_xlabel(r"$\ln(\sigma/\bar\sigma(\bar S))$, lives $\geq20$"); c.set_ylabel("density")
    c.set_title(r"(c) F3: $\sigma/\bar\sigma$ shifts with size (KS up to 0.21)", loc="left"); c.legend(fontsize=8, frameon=False)
    for k in range(5):
        z = np.sort(np.abs(d[f"z_{k}"]))[::-1]
        p = np.arange(1, z.size + 1) / z.size
        pick = np.unique(np.geomspace(1, z.size, 400).astype(int)) - 1
        e.plot(z[pick], p[pick], color=FIFTH[k], lw=2, label=f"size fifth {k + 1}")
    xs = np.linspace(0, 6, 100)
    from scipy.stats import norm
    e.plot(xs, 2 * norm.sf(xs * np.sqrt(np.pi / 2)), color=MUTED, lw=1.4, ls=":", label="Gaussian (same MAD)")
    e.set_yscale("log"); e.set_ylim(1e-5, 1.2); e.set_xlim(0, 12)
    e.set_xlabel(r"$|g-\bar g_i|/\sigma_i$, lives $\geq10$"); e.set_ylabel(r"$P(|z|>x)$")
    e.set_title("(d) F4: tails fatten with size", loc="left"); e.legend(fontsize=8, frameon=False)
    pers = d["persistence"]
    for k in range(5):
        f.plot(q, pers[k, :, 0] / pers[k, :, 1], "o-", color=FIFTH[k], lw=2, ms=8, mec="white", mew=1.5, label=f"size fifth {k + 1}")
    f.axhline(1, color=MUTED, lw=1.2, ls=":", label="iid growth")
    f.set_xticks(q); f.set_xlabel("horizon $k$ (steps)"); f.set_ylabel("k-step excess kurtosis / iid value")
    f.set_title("(e) F5: persistence only for the largest firms", loc="left"); f.legend(fontsize=8, frameon=False)
    x5 = np.arange(1, 6)
    g.plot(x5, d["corr_agg"], "o-", color="#2a78d6", lw=2, ms=8, mec="white", mew=1.5, label="with aggregate growth")
    g.plot(x5, d["corr_pair"], "s-", color="#eb6834", lw=2, ms=8, mec="white", mew=1.5, label="between firms of the same size fifth")
    g.set_xticks(x5); g.set_xlabel("size fifth (1 = smallest)"); g.set_ylabel("mean correlation of growth")
    g.set_title("(f) F6: co-movement rises with size", loc="left"); g.legend(fontsize=8, frameon=False)
    for ax in axs.flat:
        style(ax)
    fig.tight_layout(); fig.savefig(OUT, dpi=150); print("wrote", OUT)


if __name__ == "__main__":
    main()

"""Figure: the fitness law against the full model, alpha = 6, 3.5, 2.5.
Top: distribution of the per-firm mean fitness, measured vs sampled from the law with the simulation's
order parameters (law_cavity).  Bottom: amplitude f against relative degree, binned, measured vs law.
Needs data/lifecycle/fit_*.npz and law_cavity_*.npz (fitness_stats.py, mf_law.py).
Run: uv run python scripts/lifecycle/plot_fitness_law.py  -> thesis/assets/figures/fitness_law.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import skew

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "lifecycle")
OUT = os.path.join(ROOT, "thesis", "assets", "figures", "fitness_law.png")
POINTS = (("6.0", "1.75"), ("3.5", "1.9"), ("2.5", "1.75"))


def main():
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for j, (a, s) in enumerate(POINTS):
        fit = np.load(os.path.join(DATA, f"fit_a{a}_s{s}_l1e-03.npz"))
        law = np.load(os.path.join(DATA, f"law_cavity_a{a}_s{s}_l1e-03.npz"))
        m, f, k = fit["m"], fit["f"], fit["k"].astype(float)
        kap = k / k.mean()
        bins = np.linspace(np.percentile(m, 0.2), np.percentile(m, 99.8), 70)
        x = ax[0, j]
        x.hist(m, bins, density=True, color="#e76f51", alpha=0.55, label=f"measured (skew {skew(m):.2f})")
        x.hist(law["m"], bins, density=True, histtype="step", color="#264653", lw=1.8,
               label=f"fitness law (skew {skew(law['m']):.2f})")
        x.set_yscale("log"); x.set_xlabel(r"mean fitness $\bar\Phi$")
        x.set_title(rf"$\alpha={a}$, $\sigma={s}$", fontsize=11); x.legend(fontsize=8.5)
        x = ax[1, j]
        e = np.geomspace(kap.min(), kap.max() * 1.0001, 16)              # log bins, plotted at the median degree
        idx = np.clip(np.digitize(kap, e) - 1, 0, len(e) - 2)
        keep = [i for i in range(len(e) - 1) if (idx == i).sum() >= 50]
        c = np.array([np.median(kap[idx == i]) for i in keep])
        for y, col, lab in ((f, "#e76f51", "measured"), (law["f"], "#264653", "fitness law")):
            med = [np.median(y[idx == i]) for i in keep]
            q1 = [np.quantile(y[idx == i], 0.25) for i in keep]; q3 = [np.quantile(y[idx == i], 0.75) for i in keep]
            x.plot(c, med, "o-" if col == "#e76f51" else "-", color=col, lw=1.6, ms=4, label=lab)
            if col == "#e76f51":
                x.fill_between(c, q1, q3, color=col, alpha=0.2, lw=0)
        x.plot(c, np.median(f[np.abs(np.log(kap)) < 0.1]) * np.sqrt(c), color="grey", ls=":", lw=1.2, label=r"$\propto\sqrt{\kappa}$")
        x.set_xscale("log"); x.set_yscale("log"); x.set_xlabel(r"relative degree $\kappa=k/C$"); x.set_ylabel(r"amplitude $f$")
        for axis in (x.xaxis, x.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        x.legend(fontsize=8.5)
    ax[0, 0].set_ylabel("density")
    fig.tight_layout(); fig.savefig(OUT, dpi=160); print("wrote", OUT)


if __name__ == "__main__":
    main()

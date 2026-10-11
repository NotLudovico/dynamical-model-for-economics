"""Figure: spell-panel statistics at the headline step (dt = 0.5) and at the calibrated year.
(a) Pooled lag-1 autocorrelation of growth within spells against the step.  (b) E[v | Sbar] over all
spells (30 equal-count bins), dashed at dt = 0.5, solid at the calibrated step, with the Compustat
level and slope for reference.  (c) Local beta over fifths of the 25 all-bins bins.  (d) zeta_q/zeta_1
at the calibrated step: larger half of spells (msb.upper_zeta) and all bins.
Needs data/calibrated_year/*.npz (scripts/calibrated_year.py).
Run: uv run python scripts/plot_calibrated_year.py  -> thesis/assets/figures/calibrated_year.png"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from relative_glv.msb import upper_zeta
from calibrated_year import STEPS, all_bins_zeta, curve

OUT = os.path.join(ROOT, "thesis", "assets", "figures", "calibrated_year.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POINTS = (  # tag, label, colour (categorical slots 1-3), calibrated step (lag-1 crosses zero)
    ("a6.0_s1.75_l1e-03", r"$\alpha=6$, $\sigma=1.75$", "#2a78d6", 3.0),
    ("a3.5_s1.9_l1e-03", r"$\alpha=3.5$, $\sigma=1.9$", "#eb6834", 2.0),
    ("a2.5_s1.75_l1e-03", r"$\alpha=2.5$, $\sigma=1.75$", "#1baf7a", 3.0),
)
DATA_RATIO = (1.0, 2.0, 2.6, 2.9)


def load(tag):
    return np.load(os.path.join(ROOT, "data", "calibrated_year", tag + ".npz"))


def local_beta(S, v):
    ok = (v > 0) & (S > 0)
    S, v = S[ok], v[ok]
    P = np.array_split(np.argsort(S), 25)
    ls = np.log10([S[p].mean() for p in P]); lv = np.log10([v[p].mean() for p in P])
    return [ls[i:i + 5].mean() for i in range(0, 25, 5)], [-np.polyfit(ls[i:i + 5], lv[i:i + 5], 1)[0] for i in range(0, 25, 5)]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def main():
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    (a, b), (c, d) = axs
    for tag, lab, col, cal in POINTS:
        D = load(tag)
        k0, kc = "0.5", f"{cal:g}"
        lag = [np.corrcoef(D[f"lagx_{s:g}"], D[f"lagy_{s:g}"])[0, 1] for s in STEPS]
        a.plot(STEPS, lag, "o-", color=col, lw=2, ms=8, label=lab, mec="white", mew=1.5)
        a.plot([cal], [lag[STEPS.index(cal)]], "o", ms=13, mfc="none", mec=col, mew=2)
        for k, ls, alpha in ((k0, "--", 0.55), (kc, "-", 1.0)):
            cs, cv = curve(D[f"Sbar_{k}"], D[f"vol_{k}"])
            b.plot(cs, cv, ls, color=col, lw=2, alpha=alpha,
                   label=f"{lab}, step {k}" if k == kc else None)
            x, y = local_beta(D[f"Sbar_{k}"], D[f"vol_{k}"])
            c.plot(x, y, "o" + ls, color=col, lw=2, ms=8, alpha=alpha, mec="white", mew=1.5,
                   label=f"{lab}, step {k}" if k == kc else None)
        e, S, v = D[f"econ_{kc}"], D[f"Sbar_{kc}"], D[f"vol_{kc}"]
        Zu = np.array([upper_zeta(S[e == i], v[e == i]) for i in np.unique(e)])
        q = np.arange(1, 5)
        d.plot(q, Zu.mean(0) / Zu.mean(0)[0], "o-", color=col, lw=2, ms=8, mec="white", mew=1.5,
               label=rf"{lab}, larger half: $\beta={np.median(Zu[:, 0]):.2f}$")
        if tag.startswith("a6.0"):
            Za = np.array([all_bins_zeta(S[e == i], v[e == i]) for i in np.unique(e)])
            d.plot(q, Za.mean(0) / Za.mean(0)[0], "s:", color=col, lw=2, ms=8, mec="white", mew=1.5,
                   label=rf"{lab}, all bins: $\beta={np.median(Za[:, 0]):.2f}$")
    a.axhline(0, color=MUTED, lw=1); a.axhspan(-0.1, 0.1, color=GRID, alpha=0.6, lw=0)
    a.text(5.9, 0.12, "Compustat-like: |lag-1| < 0.1", ha="right", va="bottom", color=MUTED, fontsize=8.5)
    a.set_xlabel("sampling step called a year (model time units)"); a.set_ylabel("lag-1 autocorrelation of growth within spells")
    a.set_title("(a) the headline step 0.5 is strongly autocorrelated; circles: calibrated step", loc="left")
    a.legend(fontsize=8.5, frameon=False)
    s = np.geomspace(1e-3, 1e1, 10)
    b.plot(s, 1.218 * (s / 1e-3) ** -0.2, color=INK, lw=1.6, ls=(0, (6, 3)),
           label="Compustat: level and slope $-0.2$ (MSB 2024)")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(r"mean spell size $\bar S$ (model: relative to all firms; data: to listed firms)")
    b.set_ylabel(r"$\mathbb{E}[v\mid\bar S]$")
    b.set_title("(b) volatility against size, all spells (dashed: step 0.5)", loc="left")
    b.legend(fontsize=8, frameon=False, loc="lower left")
    c.axhline(0.2, color=INK, lw=1.4, ls=(0, (6, 3)), label=r"Compustat $\beta\approx0.2$")
    c.axhline(0, color=MUTED, lw=1)
    c.set_xlabel(r"$\log_{10}\bar S$ (centre of each fifth of the bins)"); c.set_ylabel("local exponent")
    c.set_title("(c) local slope by fifths of the size range (dashed: step 0.5)", loc="left")
    c.legend(fontsize=8, frameon=False, loc="upper left")
    q = np.arange(1, 5)
    d.plot(q, q, color=MUTED, lw=1.2, ls=":", label=r"fixed shape, $\zeta_q=q\zeta_1$")
    d.plot(q, DATA_RATIO, "D-", color=INK, lw=2, ms=8, label="Compustat (MSB 2024)")
    d.set_xticks(q); d.set_xlabel("$q$"); d.set_ylabel(r"$\zeta_q/\zeta_1$")
    d.set_title("(d) ratio curve at the calibrated step", loc="left")
    d.legend(fontsize=8, frameon=False, loc="upper left")
    for ax in axs.flat:
        style(ax)
    fig.tight_layout(); fig.savefig(OUT, dpi=160); print("wrote", OUT)


if __name__ == "__main__":
    main()

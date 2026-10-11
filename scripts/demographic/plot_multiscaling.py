"""Figure: the multiscaling mechanism of the demographic-noise model (scripts/demographic/multiscaling_theory.py).
(a) Var(ln v | Sbar) and the share explained by the degree, power-law against regular graph.
(b) Fitness floor: mean v^2 against relative degree in the smallest and largest size fifths, with the
fitted law A kappa + B.  (c) Noise share <w>_q by size fifth for q = 1..4.  (d) Ratio curves: measured,
predicted by the two-term law, regular graph, Compustat.
Run: uv run python scripts/demographic/plot_multiscaling.py -> thesis/assets/figures/demographic_multiscaling.png"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiscaling_theory import OUT as DIR, path, lives
from plot_robustness import style, INK, MUTED

OUT = os.path.join(ROOT, "thesis", "assets", "figures", "demographic_multiscaling.png")
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
QCOL = ["#c7dcf5", "#8ab8ec", "#4f93e0", "#163f73"]


def main():
    R = np.load(os.path.join(DIR, "results.npz"))
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    fig, axs = plt.subplots(1, 4, figsize=(18.5, 4.7))
    a, b, c, d = axs
    for kind, col, name in (("powerlaw", BLUE, "power-law graph"), ("regular", ORANGE, "regular graph")):
        lx, tot, expl = R[f"{kind}_lx"], R[f"{kind}_tot"], R[f"{kind}_expl"]
        a.plot(lx / np.log(10), tot, "o-", color=col, lw=2, ms=5, mec="white", mew=1, label=rf"Var$(\ln v\mid\bar S)$, {name}")
        a.plot(lx / np.log(10), tot * expl, "s:", color=col, lw=2, ms=5, mec="white", mew=1, label=f"part explained by degree, {name}")
    a.set_xlabel(r"$\log_{10}\bar S$ (25 bins, lives $\geq20$)"); a.set_ylabel(r"variance of $\ln v$ at fixed size")
    a.set_title("(a) the spread grows with size, through the degree", loc="left"); a.legend(fontsize=7.5, frameon=False)
    ds = [np.load(path("powerlaw", s)) for s in (0, 1)]
    Sb, v, k, n = (np.concatenate(x) for x in zip(*[lives(dd["S"], dd["L"], dd["deg"], 20) for dd in ds]))
    kap = k / k.mean()
    P = np.array_split(np.argsort(Sb), 25)
    A, lB = float(R["powerlaw_A"]), R["powerlaw_lB"]
    for j0, col, name in ((0, "#8ab8ec", "smallest size fifth"), (20, "#163f73", "largest size fifth")):
        idx = np.concatenate(P[j0:j0 + 5])
        kb = np.quantile(kap[idx], np.linspace(0, 1, 11))
        cls = np.clip(np.digitize(kap[idx], kb[1:-1]), 0, 9)
        km = np.array([kap[idx][cls == m].mean() for m in range(10)])
        vm = np.array([np.mean(v[idx][cls == m] ** 2) for m in range(10)])
        b.plot(km, vm, "o", color=col, ms=8, mec="white", mew=1.5, label=f"measured, {name}")
        Bm = np.mean(np.exp(lB[j0:j0 + 5]))
        kk = np.geomspace(km.min(), km.max(), 50)
        b.plot(kk, (A * kk + Bm) * np.mean(v[idx] ** 2) / np.mean(A * kap[idx] + Bm), "-", color=col, lw=1.6,
               label=rf"law $A\kappa+B$, {name}")
    b.set_xscale("log"); b.set_yscale("log"); b.minorticks_off(); b.set_xticks([0.5, 1, 2, 4]); b.set_xticklabels(["0.5", "1", "2", "4"])
    b.set_xlabel(r"relative degree $\kappa=k/C$"); b.set_ylabel(r"mean $v^2$")
    b.set_title(r"(b) fitness floor $\propto\kappa$ (fit: exponent 1.00)", loc="left"); b.legend(fontsize=7.5, frameon=False)
    ws = R["powerlaw_ws"]
    x5 = np.arange(1, 6)
    for q in range(4):
        c.plot(x5, [ws[i:i + 5, q].mean() for i in range(0, 25, 5)], "o-", color=QCOL[q], lw=2, ms=8, mec="white", mew=1.5,
               label=rf"$\langle w\rangle_{q + 1}$")
    c.set_xticks(x5); c.set_xlabel("size fifth (1 = smallest)"); c.set_ylabel(r"noise share $w=B/(v_F^2+B)$, $v^q$-weighted")
    c.set_title(r"(c) $\zeta_q=\frac{q}{2}\langle w\rangle_q$: higher $q$, lower share", loc="left"); c.legend(fontsize=8, frameon=False)
    q = np.arange(1, 5)
    z, zp, zr = R["powerlaw_z"], R["powerlaw_zp"], R["regular_z"]
    d.plot(q, q, color=MUTED, lw=1.2, ls=":", label=r"fixed shape, $\zeta_q=q\zeta_1$")
    d.plot(q, z / z[0], "o-", color=BLUE, lw=2, ms=8, mec="white", mew=1.5, label=rf"power-law graph, measured ($\beta={z[0]:.2f}$)")
    d.plot(q, zp / zp[0], "s--", color=BLUE, lw=2, ms=8, mfc="white", label=r"two-term law $A\kappa+B(\bar S)$")
    d.plot(q, zr / zr[0], "o-", color=ORANGE, lw=2, ms=8, mec="white", mew=1.5, label=rf"regular graph ($\beta={zr[0]:.2f}$)")
    d.plot(q, (1, 1.95, 2.55, 2.90), "D-", color=INK, lw=2, ms=8, label="Compustat (MSB 2024)")
    d.set_xticks(q); d.set_xlabel("$q$"); d.set_ylabel(r"$\zeta_q/\zeta_1$ (lives $\geq20$)")
    d.set_title("(d) equal degrees remove most of the bend", loc="left"); d.legend(fontsize=7.5, frameon=False)
    for ax in axs:
        style(ax)
    fig.tight_layout(); fig.savefig(OUT, dpi=150); print("wrote", OUT)


if __name__ == "__main__":
    main()

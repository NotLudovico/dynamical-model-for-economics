"""Phase diagram with the HDMFT lines.  (a) Simulated regimes of the phase grid (N=4000, alpha=2.5,
10 seeds per cell, frozen = median churn < 0.035) with the HDMFT stability line sigma_c(mu), the
zero-growth line (solid where the fixed point is stable) and the operating point.  (b) The theory
lines for alpha = 2.2, 2.5, 3.5, 6 and the fully connected limit.
Needs data/phase_meandeg_fine_24x20_10s.npz, data/hdmft_sigma_c_line.npz, data/phase_theory_lines.npz,
thesis/standalone/zero_growth_theory/zero_growth_line.npz.
Run: uv run python scripts/plot_phase_theory.py  -> thesis/assets/figures/phase_theory.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(ROOT, *p)
OUT = D("thesis", "assets", "figures", "phase_theory.png")
COLS = ["#e8e8e8", "#457b9d", "#e76f51", "#1d3557"]          # frozen, fluctuating+growing, shrinking, failed
LINE = {2.2: "#9b2226", 2.5: "#ca6702", 3.5: "#0a9396", 6.0: "#005f73"}


def regimes():
    d = np.load(D("data", "phase_meandeg_fine_24x20_10s.npz"))
    sig, mus, raw = d["sigmas"], -d["mus"], d["raw"]
    integ = np.isfinite(raw[..., 0]).mean(2)
    med = np.nanmedian(raw, axis=2)
    churn, geff = med[..., 0], med[..., 1]
    code = np.where(integ < 0.5, 3, np.where(~(churn >= 0.035), 0, np.where(geff < 0, 2, 1)))
    o = np.argsort(mus)
    return sig, mus[o], code[:, o], churn[:, o]


def edges(a):
    m = (a[:-1] + a[1:]) / 2
    return np.concatenate([[2 * a[0] - m[0]], m, [2 * a[-1] - m[-1]]])


def g0_line(sig):
    w1 = lambda x: x * norm.cdf(x) + norm.pdf(x)
    w2 = lambda x: (1 + x * x) * norm.cdf(x) + x * norm.pdf(x)
    out = []
    for s in sig:
        x = brentq(lambda x: s * s * w2(x) - 1, -20, 60)
        out.append(-(1 - x / w1(x)))
    return np.array(out)


def main():
    sig, mus, code, churn = regimes()
    sc = np.load(D("data", "hdmft_sigma_c_line.npz"))
    zg = np.load(D("thesis", "standalone", "zero_growth_theory", "zero_growth_line.npz"))
    fam = np.load(D("data", "phase_theory_lines.npz"))

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
    a = ax[0]
    with np.errstate(invalid="ignore"):
        a.pcolormesh(edges(mus), edges(sig), code, cmap=ListedColormap(COLS), vmin=-0.5, vmax=3.5,
                     shading="flat", edgecolors="white", linewidth=0.3)
    o = np.argsort(-sc["mus"])
    a.plot(-sc["mus"][o], sc["sc_lam1e3"][o], color="black", lw=2.4, label=r"HDMFT $\sigma_c(\mu)$")
    c = zg["controlled"].astype(bool)
    a.plot(zg["mu_thesis"][c], zg["sigma"][c], color="#ffb703", lw=2.4, label=r"HDMFT $g^*=0$ (stable)")
    a.plot(zg["mu_thesis"][~c], zg["sigma"][~c], color="#ffb703", lw=2.4, ls="--", label=r"$g^*=0$ continuation")
    a.scatter(zg["measured_mu_thesis"], zg["measured_sigma"], s=16, facecolor="white", edgecolor="black",
              zorder=5, label=r"simulated $g_{\rm eff}=0$")
    a.scatter([-1.76], [1.75], marker="*", s=380, color="black", zorder=6)
    a.set_xlim(edges(mus)[0], edges(mus)[-1]); a.set_ylim(0.9, edges(sig)[-1])
    a.set_xlabel(r"mean interaction $\mu$ ($\mu<0$ competitive)"); a.set_ylabel(r"disorder $\sigma$")
    a.set_title(r"(a) simulation ($\alpha=2.5$, $N=4000$) and HDMFT", fontsize=11)
    a.text(-2.05, 1.38, "unique fixed point", fontsize=10, ha="center")
    a.text(-1.25, 1.18, "not simulated", fontsize=9, color="grey", ha="center", style="italic")
    a.text(-1.0, 1.85, "fluctuating, growing", fontsize=10, color="white", ha="center", weight="bold")
    a.text(-2.12, 1.77, "shrinking", fontsize=10, color="white", ha="center", weight="bold")
    a.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    b = ax[1]
    for i, al in enumerate(fam["alphas"]):
        b.plot(-fam["mus"], fam["sigma_c"][i], color=LINE[float(al)], lw=2.2, label=rf"$\sigma_c$, $\alpha={al:g}$")
        z, cc = fam["mu0"][i], fam["controlled"][i].astype(bool)
        b.plot(z[cc], fam["sigmas"][cc], color=LINE[float(al)], lw=1.4, ls=":")
    b.axhline(np.sqrt(2), color="grey", lw=1.5, ls="--", label=r"fully connected $\sqrt{2}$")
    s = np.linspace(0.3, 1.414, 60)
    b.plot(g0_line(s), s, color="grey", lw=1.2, ls=":", label=r"$g^*=0$ (dotted, stable part)")
    b.scatter([-1.76], [1.75], marker="*", s=300, color="black", zorder=6)
    b.set_xlim(-2.5, 0.0); b.set_ylim(0.3, 1.95)
    b.set_xlabel(r"mean interaction $\mu$"); b.set_ylabel(r"disorder $\sigma$")
    b.set_title(r"(b) HDMFT lines by degree exponent ($\lambda=0$)", fontsize=11)
    b.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

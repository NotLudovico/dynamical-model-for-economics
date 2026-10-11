"""Figure: robustness of the demographic-noise model's MSB facts (data/demographic/robustness/,
scripts/demographic/robustness.py).
(a) beta against N (pooled lives, error bars: spread between economies).  (b) zeta_q/zeta_1 against N.
(c) E[v | Sbar] against N with the Compustat level and slope.  (d) beta and zeta_4/zeta_1 at every
point of the neighbourhood.  (e) Tail index b of sigma/sigmabar at every point.  (f) Aggregate
volatility per step against N.
Run: uv run python scripts/demographic/plot_robustness.py -> thesis/assets/figures/demographic_robustness.png"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from robustness import JOBS, path, zeta_all
from factsheet import hill

OUT = os.path.join(ROOT, "thesis", "assets", "figures", "demographic_robustness.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NSEQ = {"ref": ("#9ec5f0", 8000), "N16k": ("#5b9be6", 16000), "N32k": ("#2a78d6", 32000), "N64k": ("#14467f", 64000)}
NMIN = {5: "#2a78d6", 10: "#eb6834", 20: "#1baf7a"}
DATA_RATIO = (1.0, 2.0, 2.6, 2.9)


def load(label):
    seeds = next(s for l, _, s in JOBS if l == label)
    return [np.load(path(label, s)) for s in seeds if os.path.exists(path(label, s))]


def pooled(ds, nm, key="v"):
    Sb = np.concatenate([d["Sbar"][d["n"] >= nm] for d in ds]).astype(float)
    v = np.concatenate([d[key][d["n"] >= nm] for d in ds]).astype(float)
    ok = v > 0
    return Sb[ok], v[ok]


def curve(Sb, v, nb=25):
    P = np.array_split(np.argsort(Sb), nb)
    return np.array([Sb[p].mean() for p in P]), np.array([v[p].mean() for p in P])


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def main():
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    a, b, c, d, e, f = axs.flat
    q = np.arange(1, 5)
    data = {l: load(l) for l, _, _ in JOBS}
    for nm, col in NMIN.items():
        xs, ys, es = [], [], []
        for lab, (_, N) in NSEQ.items():
            ds = data[lab]
            if not ds:
                continue
            z = zeta_all(*pooled(ds, nm))[0]
            per = [zeta_all(d["Sbar"][d["n"] >= nm].astype(float), d["v"][d["n"] >= nm].astype(float))[0][0] for d in ds]
            xs.append(N); ys.append(z[0]); es.append(np.std(per))
        a.errorbar(xs, ys, yerr=es, fmt="o-", color=col, lw=2, ms=8, mec="white", mew=1.5, capsize=3,
                   label=rf"lives $\geq{nm}$")
    a.axhline(0.2, color=INK, lw=1.4, ls=(0, (6, 3)), label=r"Compustat $\beta=0.20$")
    a.set_xscale("log"); a.minorticks_off(); a.set_xticks([8000, 16000, 32000, 64000]); a.set_xticklabels(["8k", "16k", "32k", "64k"])
    a.set_xlabel("number of firms $N$"); a.set_ylabel(r"$\beta=\zeta_1$ (all bins)")
    a.set_title(r"(a) $\beta$ converges by $N\approx16$k, to about 0.13", loc="left"); a.legend(fontsize=8, frameon=False)
    for lab, (col, N) in NSEQ.items():
        ds = data[lab]
        if not ds:
            continue
        Sb, v = pooled(ds, 5)
        z = zeta_all(Sb, v)[0]
        b.plot(q, z / z[0], "o-", color=col, lw=2, ms=8, mec="white", mew=1.5, label=f"N = {N // 1000}k")
        cs, cv = curve(Sb, v)
        c.plot(cs, cv, "o-", color=col, lw=2, ms=4, mec="white", mew=0.8, label=f"N = {N // 1000}k")
    b.plot(q, q, color=MUTED, lw=1.2, ls=":", label=r"fixed shape, $\zeta_q=q\zeta_1$")
    b.plot(q, DATA_RATIO, "D-", color=INK, lw=2, ms=8, label="Compustat (MSB 2024)")
    b.set_xticks(q); b.set_xlabel("$q$"); b.set_ylabel(r"$\zeta_q/\zeta_1$ (lives $\geq5$)")
    b.set_title("(b) the ratio curve stays below $q$ at every N", loc="left"); b.legend(fontsize=8, frameon=False)
    s = np.geomspace(1e-3, 1e1, 10)
    c.plot(s, 1.218 * (s / 1e-3) ** -0.2, color=INK, lw=1.6, ls=(0, (6, 3)), label=r"Compustat: level, slope $-0.2$")
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel(r"life-mean size $\bar S$ (model: all firms; data: listed firms)"); c.set_ylabel(r"$\mathbb{E}[v\mid\bar S]$, lives $\geq5$")
    c.set_title("(c) no hump; a mid-range flat stretch at large N", loc="left"); c.legend(fontsize=8, frameon=False, loc="lower left")
    names = [l for l, _, _ in JOBS if data[l]]
    betas, r4, bs = [], [], []
    for lab in names:
        Sb, v = pooled(data[lab], 5); z = zeta_all(Sb, v)[0]; betas.append(z[0]); r4.append(z[3] / z[0])
        Sb2, v2 = pooled(data[lab], 2)
        P = np.array_split(np.argsort(Sb2), 25); r = np.empty(v2.size)
        for p in P:
            r[p] = v2[p] / v2[p].mean()
        bs.append(hill(r, 0.05))
    y = np.arange(len(names))[::-1]
    d.plot(betas, y, "o", color="#2a78d6", ms=9, mec="white", mew=1.5, label=r"model, lives $\geq5$")
    for yy, bb, rr in zip(y, betas, r4):
        d.text(bb + 0.008, yy, rf"$\zeta_4/\zeta_1$={rr:.2f}", va="center", fontsize=7.5, color=MUTED)
    d.axvline(0.2, color=INK, lw=1.4, ls=(0, (6, 3)), label=r"Compustat $\beta=0.20$ ($\zeta_4/\zeta_1$=2.9)")
    d.set_xlim(0.05, 0.5); d.set_yticks(y); d.set_yticklabels(names); d.set_xlabel(r"$\beta$ (all bins)")
    d.set_title(r"(d) neighbourhood: $\sigma$, $D$ and $C$ set the level of $\beta$", loc="left"); d.legend(fontsize=8, frameon=False, loc="lower right")
    e.plot(bs, y, "o", color="#1baf7a", ms=9, mec="white", mew=1.5, label=r"model, lives $\geq2$")
    e.axvline(4.6, color=INK, lw=1.4, ls=(0, (6, 3)), label="Compustat $b=4.6$")
    e.set_yticks(y); e.set_yticklabels(names); e.set_xlim(3.5, 5.5); e.set_xlabel(r"tail index $b$ of $\sigma/\bar\sigma$ (Hill, top 5%)")
    e.set_title(r"(e) the volatility-spread tail is the same everywhere", loc="left"); e.legend(fontsize=8, frameon=False, loc="upper right")
    xs, ys, es = [], [], []
    for lab, (_, N) in NSEQ.items():
        ds = data[lab]
        if ds:
            av = [float(x["aggvol"]) for x in ds]; xs.append(N); ys.append(np.mean(av)); es.append(np.std(av))
            f.plot([N] * len(av), av, "o", color="#9ec5f0", ms=6, mec="white", mew=1)
    f.errorbar(xs, ys, yerr=es, fmt="o-", color="#2a78d6", lw=2, ms=9, mec="white", mew=1.5, capsize=3, label="mean over economies")
    f.plot(xs, ys[0] * np.sqrt(xs[0] / np.array(xs)), color=MUTED, lw=1.4, ls=":", label=r"if granular: $\propto N^{-1/2}$")
    f.set_xscale("log"); f.minorticks_off(); f.set_xticks([8000, 16000, 32000, 64000]); f.set_xticklabels(["8k", "16k", "32k", "64k"])
    f.set_xlabel("number of firms $N$"); f.set_ylabel("aggregate growth volatility per step")
    f.set_title("(f) aggregate volatility: no $N^{-1/2}$ decline", loc="left"); f.legend(fontsize=8, frameon=False)
    for ax in axs.flat:
        style(ax)
    fig.tight_layout(); fig.savefig(OUT, dpi=150); print("wrote", OUT)


if __name__ == "__main__":
    main()

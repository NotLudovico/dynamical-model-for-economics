"""Multiscaling from the physical decomposition of the volatility law (volatility_law.py).

Each life with >= 20 growth rates has var g ~ V_F + B: V_F from the noise-free one-firm equation along the
life's measured field (the fitness floor), B the filtered demographic part.  With B replaced by its
mean in the life's size bin, B(Sbar), the identity
    zeta_q = (q/2) b <w>_q (+ composition term),   w = B / (V_F + B),   b = -dln B / dln Sbar,
says the ratio curve bends below q iff V_F differs between lives of equal size.  Counterfactuals on the
same lives, 25 equal-count bins of Sbar, sd of growth as the volatility:
  measured          var g
  floor             V_F (own) + B(Sbar)          -> heterogeneity of the floor only
  floor + noise     V_F (own) + B (own)          -> adds the dispersion of the per-life noise estimate
  no floor spread   V_F(Sbar) + B (own)          -> floor replaced by its bin mean
On the power-law graph and on the random regular graph (every degree = C) of multiscaling_theory.py.
Run: uv run python scripts/demographic/multiscaling_law.py
  -> data/demographic/multiscaling_law.npz, thesis/assets/figures/demographic_multiscaling_law.png
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import reference as ref
from volatility_law import rows_of, OUT as ROWS_POWERLAW

DATA = os.path.join(ROOT, "data", "demographic")
OUT = os.path.join(DATA, "multiscaling_law.npz")
FIG = os.path.join(ROOT, "thesis", "assets", "figures", "demographic_multiscaling_law.png")
QS = (1, 2, 3, 4)
MSB_RATIO = np.array([1.95, 2.55, 2.90])


def rows(kind):
    f = ROWS_POWERLAW if kind == "powerlaw" else os.path.join(DATA, f"volatility_law_{kind}.npz")
    if not os.path.exists(f):
        np.savez(f, rows=np.array([r for s in ref.SEEDS for r in rows_of(s, kind)]))
    return np.load(f)["rows"].T                                  # Sbar, var g, V_F, B, degree


def binmean(x, P):
    y = np.empty_like(x)
    for p in P:
        y[p] = x[p].mean()
    return y


def zeta(Y, P, lx):
    return np.array([-np.polyfit(lx, np.log([np.mean(Y[p] ** (q / 2)) for p in P]), 1)[0] for q in QS])


def analyse(kind):
    Sb, vg, VF, B, deg = rows(kind)
    kap = deg / deg.mean()
    P = np.array_split(np.argsort(Sb), 25)
    lx = np.log([Sb[p].mean() for p in P])
    Bs, VFs = binmean(B, P), binmean(VF, P)
    cases = {"measured": vg, "floor": VF + Bs, "floor + noise": VF + B, "no floor spread": VFs + B}
    out = {"lx": lx}
    print(f"\n=== {kind} graph: {Sb.size} lives with >= 20 growth rates, degree CV {kap.std():.2f}")
    for name, Y in cases.items():
        z = zeta(Y, P, lx)
        out[f"zeta {name}"] = z
        print(f"  {name:16s} zeta {' '.join(f'{x:.3f}' for x in z)}   ratios {' '.join(f'{x:.2f}' for x in z[1:] / z[0])}")
    # by fifths of the 25 bins
    F = [np.concatenate(P[i:i + 5]) for i in range(0, 25, 5)]
    lB = np.log([B[p].mean() for p in P])
    b = np.array([-np.polyfit(lx[i:i + 5], lB[i:i + 5], 1)[0] for i in range(0, 25, 5)])
    w = Bs / (VF + Bs)
    wq = np.array([[np.sum(w[p] * (VF[p] + Bs[p]) ** (q / 2)) / np.sum((VF[p] + Bs[p]) ** (q / 2)) for q in QS] for p in F])
    var_lvf = np.array([np.log(VF[p]).var() for p in F])
    s2 = np.array([np.mean([np.log(VF[p] + Bs[p]).var() for p in P[i:i + 5]]) for i in range(0, 25, 5)])
    w41 = np.array([np.mean([np.sum(w[p] * (VF[p] + Bs[p]) ** 2) / np.sum((VF[p] + Bs[p]) ** 2) /
                             (np.sum(w[p] * (VF[p] + Bs[p]) ** 0.5) / np.sum((VF[p] + Bs[p]) ** 0.5)) for p in P[i:i + 5]])
                    for i in range(0, 25, 5)])
    if kap.std() > 0.05:
        r2 = np.array([np.corrcoef(np.log(VF[p]), np.log(kap[p]))[0, 1] ** 2 for p in F])
        h = np.polyfit(np.log(kap), np.log(VF), 1)[0]
    else:
        r2, h = np.zeros(5), np.nan
    # identity check: local zeta_q of the "floor" case against (q/2) b <w>_q, per fifth
    Yf = VF + Bs
    loc = np.array([[-np.polyfit(lx[i:i + 5], np.log([np.mean(Yf[p] ** (q / 2)) for p in P[i:i + 5]]), 1)[0]
                     for q in QS] for i in range(0, 25, 5)])
    ident = 0.5 * np.array(QS)[None, :] * b[:, None] * wq
    f5 = lambda a: " ".join(f"{x:.2f}" for x in a)
    print(f"  by size fifth (Sbar {f5([Sb[p].mean() for p in F])}):")
    print(f"    mean relative degree {f5([kap[p].mean() for p in F])}")
    print(f"    Var ln V_F {f5(var_lvf)};  share explained by ln kappa {f5(r2)};  V_F ~ kappa^{h:.2f} overall")
    print(f"    slope b of B(Sbar) {f5(b)};  noise share <w>_1 {f5(wq[:, 0])}")
    print(f"    <w>_q / <w>_1, q = 4: {f5(wq[:, 3] / wq[:, 0])}")
    print(f"    within-bin s^2 = Var ln(V_F + B(Sbar)) {' '.join(f'{x:.3f}' for x in s2)};  mean V_F {f5([VF[p].mean() for p in F])}")
    print(f"    within-bin <w>_4/<w>_1 {f5(w41)}  vs lognormal exp(-3 s^2/2) {f5(np.exp(-1.5 * s2))}")
    print("    local zeta_q of 'floor' vs (q/2) b <w>_q:")
    for i in range(5):
        print(f"      fifth {i}: {f5(loc[i])}   vs   {f5(ident[i])}")
    out.update(s2=s2, w41=w41, var_lvf=var_lvf, r2=r2, h=h, b=b, wq=wq, loc=loc, ident=ident,
               Sb5=np.array([Sb[p].mean() for p in F]), kap5=np.array([kap[p].mean() for p in F]))
    # floor against degree in the smallest and largest fifth (binned medians)
    if kap.std() > 0.05:
        for j, p in ((0, F[0]), (4, F[4])):
            e = np.quantile(kap[p], np.linspace(0, 1, 11)); bi = np.clip(np.digitize(kap[p], e[1:-1]), 0, 9)
            out[f"vf_kappa_{j}"] = np.array([[np.median(kap[p][bi == k]), np.median(VF[p][bi == k])] for k in range(10)])
    return out


def figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#9a9890", "#0b0b0b"
    QCOL = ["#c7dcf5", "#8ab8ec", "#4f93e0", "#163f73"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    a, b, c, d = axs
    for kind, col, name in (("powerlaw", BLUE, "power-law graph"), ("regular", ORANGE, "regular graph")):
        r = R[kind]
        a.plot(r["Sb5"], r["s2"], "o-", color=col, lw=2, ms=6, mec="white", label=rf"$s^2$, {name}")
    a.set(xscale="log", xlabel=r"life-mean size $\bar S$", ylabel=r"$s^2=$ Var$(\ln(V_F+B)\mid\bar S)$",
          title="(a) dispersion of the total variance")
    r = R["powerlaw"]
    for j, mk, lab in ((0, "o", "smallest fifth"), (4, "s", "largest fifth")):
        x = r[f"vf_kappa_{j}"]
        b.loglog(x[:, 0], x[:, 1], mk + "-", color=BLUE if j else GREY, lw=1.8, ms=5, mec="white", label=lab)
    kk = np.geomspace(0.3, 6, 10)
    b.loglog(kk, np.median(r["vf_kappa_4"][:, 1]) * (kk / np.median(r["vf_kappa_4"][:, 0])) ** r["h"], ":", color=INK, lw=1,
             label=rf"$\propto\kappa^{{{r['h']:.2f}}}$")
    b.set(xlabel=r"relative degree $\kappa$", ylabel=r"floor $V_F$ (median)", title="(b) floor against degree")
    for k, q in enumerate(QS):
        c.plot(r["Sb5"], r["wq"][:, k], "o-", color=QCOL[k], lw=2, ms=5, mec="white", label=rf"$q={q}$")
    rr = R["regular"]
    c.plot(rr["Sb5"], rr["wq"][:, 0], "D--", color=ORANGE, lw=1.2, ms=4, mfc="white", label=r"$q=1$, regular graph")
    c.plot(rr["Sb5"], rr["wq"][:, 3], "D:", color=ORANGE, lw=1.2, ms=4, label=r"$q=4$, regular graph")
    c.set(xscale="log", xlabel=r"life-mean size $\bar S$", ylabel=r"$\langle w\rangle_q$", title="(c) noise share")
    qq = np.array(QS)
    d.plot(qq, qq, "-", color=GREY, lw=1, label=r"$\zeta_q=q\zeta_1$")
    for key, col, mk, ls, lab in (("measured", INK, "o", "-", "measured"), ("floor", BLUE, "s", "--", r"own $V_F$ + $B(\bar S)$")):
        z = r[f"zeta {key}"]
        d.plot(qq, z / z[0], mk + ls, color=col, lw=1.8, ms=6, mec="white", label=lab)
    for key, mk, ls, lab in (("measured", "D", "-", "regular graph, measured"), ("floor", "D", "--", r"regular, own $V_F$ + $B(\bar S)$")):
        z = R["regular"][f"zeta {key}"]
        d.plot(qq, z / z[0], mk + ls, color=ORANGE, lw=1.4, ms=5, mfc="white" if ls == "--" else ORANGE, label=lab)
    d.plot(qq, np.r_[1, MSB_RATIO], "*", color="#1baf7a", ms=12, label="Compustat")
    d.set(xlabel=r"$q$", ylabel=r"$\zeta_q/\zeta_1$", title="(d) ratio curves", ylim=(0.8, 4.3))
    from matplotlib.ticker import NullFormatter, FormatStrFormatter
    for ax in axs:
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.set_xticks([0.3, 0.5, 1, 2, 3] if ax is not b else [0.3, 1, 3, 10])
        if ax.get_yscale() == "log":
            ax.yaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.set_yticks([0.03, 0.05, 0.1, 0.2, 0.3])
        ax.grid(True, which="both", color="#e4e3df", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5)
    plt.tight_layout(); plt.savefig(FIG, dpi=200)
    print("wrote", FIG)


if __name__ == "__main__":
    R = {k: analyse(k) for k in ("powerlaw", "regular")}
    np.savez(OUT, **{f"{k}|{kk}": v for k, r in R.items() for kk, v in r.items()})
    figure(R)

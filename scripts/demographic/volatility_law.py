"""Volatility law of the demographic model: per-life growth variance against
    var g  ~  V_F + B,
V_F : growth variance of the NOISE-FREE one-firm equation dS = S(Phi - S) dt driven by the life's measured
      field Phi(t) from its actual starting size (fitness forcing, filtered and nonlinear, transients included);
B   : demographic part, an OU process in ln S with local relaxation rate S: mean_k (D/S_k^2)(1 - exp(-S_k dt)).
Lives with >= 20 growth rates of the two reference economies (reference.py), by deciles of life-mean size.
Run: uv run python scripts/demographic/volatility_law.py   (add "plot" to redraw from volatility_law.npz)
  -> data/demographic/volatility_law.npz, thesis/assets/figures/demographic_volatility_law.png
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import reference as ref

dt, NMIN, CHUNK = 0.01, 20, 20000
OUT = os.path.join(ROOT, "data", "demographic", "volatility_law.npz")
FIG = os.path.join(ROOT, "thesis", "assets", "figures", "demographic_volatility_law.png")


def noise_free(ph, lnS0):
    """ln S of dS = S(Phi - S)dt along padded fields ph (lives, n), Phi linear between samples."""
    per = int(round(ref.DT / dt))
    lnS = lnS0.copy(); det = [lnS.copy()]
    for k in range(ph.shape[1] - 1):
        p0, p1 = ph[:, k], ph[:, k + 1]
        for j in range(per):
            lnS = lnS + (p0 + (p1 - p0) * j / per - np.exp(lnS)) * dt
        det.append(lnS.copy())
    return np.array(det).T


def rows_of(seed, kind="powerlaw"):
    """(Sbar, var g, V_F, B, degree) for every life with >= NMIN growth rates."""
    S, L, deg, _, Phi = ref.economy(seed, kind)
    T = S.shape[1]
    lv = [(i, seg) for i in range(S.shape[0]) for seg in np.split(np.arange(T), np.flatnonzero(np.diff(L[i])) + 1)
          if seg.size >= NMIN + 1]
    lv.sort(key=lambda x: x[1].size)
    rows = []
    for c0 in range(0, len(lv), CHUNK):                       # chunks of similar length keep the padding small
        part = lv[c0:c0 + CHUNK]
        n = part[-1][1].size
        ph = np.zeros((len(part), n)); x0 = np.empty(len(part))
        for k, (i, seg) in enumerate(part):
            ph[k, :seg.size] = Phi[i, seg]; ph[k, seg.size:] = Phi[i, seg[-1]]; x0[k] = S[i, seg[0]]
        det = noise_free(ph, np.log(x0))
        for k, (i, seg) in enumerate(part):
            x = S[i, seg]; g = np.diff(np.log(x)); gd = np.diff(det[k, :seg.size])
            B = np.mean(ref.D / x[:-1] ** 2 * (1 - np.exp(-x[:-1] * ref.DT)))
            rows.append((x.mean(), g.var(), gd.var(), B, deg[i]))
        print(f"  seed {seed}: {min(c0 + CHUNK, len(lv))}/{len(lv)} lives", flush=True)
    return rows


def report(R):
    P = np.array_split(np.argsort(R[:, 0]), 10)
    print(f"lives >= {NMIN}: {len(R)}")
    print(f"{'Sbar':>6} | {'sd g':>6} | {'sqrt(V_F+B)':>11} | ratio | {'sqrt V_F':>8} {'sqrt B':>7}")
    tab = []
    for p in P:
        m = R[p].mean(0); pr = np.sqrt(m[2] + m[3])
        tab.append((m[0], np.sqrt(m[1]), pr, np.sqrt(m[2]), np.sqrt(m[3])))
        print(f"{m[0]:6.3f} | {np.sqrt(m[1]):6.3f} | {pr:11.3f} | {pr / np.sqrt(m[1]):5.2f} | {np.sqrt(m[2]):8.3f} {np.sqrt(m[3]):7.3f}")
    tab = np.array(tab)
    x = np.log(tab[:, 0])
    for name, col in (("measured", 1), ("predicted", 2)):
        print(f"slope of sd g against Sbar, {name}: {-np.polyfit(x, np.log(tab[:, col]), 1)[0]:.3f}")
    cross = np.flatnonzero(np.diff(np.sign(tab[:, 4] - tab[:, 3])))
    if cross.size:
        print(f"crossover V_F = B between Sbar {tab[cross[0], 0]:.2f} and {tab[cross[0] + 1, 0]:.2f}")
    return tab


def figure(tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.loglog(tab[:, 0], tab[:, 1], "o", color="#0b0b0b", ms=6, label="measured sd $g$")
    ax.loglog(tab[:, 0], tab[:, 2], "-", color="#2a78d6", lw=2, label=r"$\sqrt{V_F+B}$")
    ax.loglog(tab[:, 0], tab[:, 3], "--", color="#eb6834", lw=1.6, label=r"fitness $\sqrt{V_F}$")
    ax.loglog(tab[:, 0], tab[:, 4], ":", color="#1baf7a", lw=1.8, label=r"demographic $\sqrt{B}$")
    ax.set(xlabel=r"life-mean size $\bar S$", ylabel="growth volatility")
    ref.tidy_log_axes(ax, [0.3, 0.5, 1, 2, 3], [0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(FIG, dpi=200)
    print("wrote", FIG)


if __name__ == "__main__":
    if sys.argv[1:] != ["plot"]:
        np.savez(OUT, rows=np.array([r for s in ref.SEEDS for r in rows_of(s)]))   # columns: Sbar, var g, V_F, B, degree
    figure(report(np.load(OUT)["rows"]))

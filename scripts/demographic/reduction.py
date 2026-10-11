"""One-firm reduction: is a firm's life determined by its field Phi_i(t) plus its own noise?

Full model: the two reference economies (reference.py: alpha 2.5, sigma 1.70, D 0.1, N 16000, C 200).
Field: Phi_i = 1 - (a S)_i - fbar, so that the full equation is dS_i = S_i (Phi_i - S_i) dt + sqrt(D S_i) dW
up to the O(N^-1/2) term that keeps sum S = N.
Surrogate: each node integrates that one-firm equation with its measured Phi_i(t) (linear interpolation
between samples), fresh noise, exit at 0 and a newborn at D, starting from the full model's sizes at the
first sample.  Compare the life panels (relative_glv.msb: lives, zeta_bins) and the exit rates.
Run: uv run python scripts/demographic/reduction.py   (add "plot" to redraw the figure from reduction.npz)
  -> data/demographic/reduction.npz, thesis/assets/figures/demographic_reduction.png
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import reference as ref
from relative_glv.msb import lives, zeta_bins

dt = 0.01
NMIN = (2, 5, 10, 20)
OUT = os.path.join(ROOT, "data", "demographic", "reduction.npz")
FIG = os.path.join(ROOT, "thesis", "assets", "figures", "demographic_reduction.png")


def surrogate(S_full, Phi, seed):
    rng = np.random.default_rng(100 + seed)
    N, T = S_full.shape
    per = int(round(ref.DT / dt))
    S = S_full[:, 0].copy(); life = np.zeros(N, np.int64)
    out_S, out_L = [S.copy()], [life.copy()]
    for k in range(T - 1):
        for j in range(per):
            ph = Phi[:, k] + (Phi[:, k + 1] - Phi[:, k]) * j / per
            S = S + S * (ph - S) * dt + np.sqrt(D_ * np.maximum(S, 0) * dt) * rng.standard_normal(N)
            dead = S <= 0
            if dead.any():
                S[dead] = D_; life[dead] += 1
        out_S.append(S.copy()); out_L.append(life.copy())
    return np.array(out_S).T, np.array(out_L).T


def panel(S, L):
    lv = lives(S, L)
    return np.array([x[0] for x in lv]), np.array([x[1] for x in lv]), np.array([x[2] for x in lv])


def figure(save):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for k, nm in enumerate((2, 20)):
        for which, c, mk, lab in (("full", "#2a78d6", "o-", "full model"),
                                  ("sur", "#eb6834", "s--", "one firm in its measured field")):
            x, y = save[f"curve_{which}_{nm}"]
            ax[k].loglog(x, y, mk, color=c, lw=1.8, ms=4, mec="white", mew=0.6, label=lab)
        ax[k].set(title=rf"lives with $\geq{nm}$ growth rates", xlabel=r"life-mean size $\bar S$",
                  ylabel=r"$E[v\mid\bar S]$")
        ref.tidy_log_axes(ax[k], [0.01, 0.03, 0.1, 0.3, 1, 3], [0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2])
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(FIG, dpi=200)
    print("wrote", FIG)


D_ = ref.D
if __name__ == "__main__" and sys.argv[1:] == ["plot"]:
    figure(dict(np.load(OUT)))
elif __name__ == "__main__":
    res = {}
    for seed in ref.SEEDS:
        S_full, L_full, _, _, Phi = ref.economy(seed)
        L_full = L_full - L_full[:, :1]
        S_sur, L_sur = surrogate(S_full, Phi, seed)
        res[seed] = {"full": panel(S_full, L_full), "sur": panel(S_sur, L_sur),
                     "exits": (L_full[:, -1].mean() / ref.TOBS, L_sur[:, -1].mean() / ref.TOBS)}
        print(f"seed {seed}: surrogate mean size {S_sur.mean():.3f} (full 1); exits per node per unit time "
              f"full {res[seed]['exits'][0]:.3f}, surrogate {res[seed]['exits'][1]:.3f}", flush=True)

    save = {}
    print(f"\n{'nmin':>4} {'':4} {'beta':>6}  ratios            local exponents (fifths)")
    for nm in NMIN:
        for which in ("full", "sur"):
            Sb, v, n = (np.concatenate(x) for x in zip(*(res[s][which] for s in ref.SEEDS)))
            sel = n >= nm
            z, _, _, _, dec, loc = zeta_bins(Sb[sel], v[sel])
            save[f"zeta_{which}_{nm}"] = z
            print(f"{nm:4d} {which:4s} {z[0]:6.3f}  {' '.join(f'{x:.2f}' for x in z[1:] / z[0])}  "
                  f"{' '.join(f'{x:.2f}' for x in loc)}  ({sel.sum()} lives, {dec:.1f} dec)")
    ex = np.array([res[s]["exits"] for s in ref.SEEDS])
    print(f"exit rate full {ex[:, 0].mean():.3f}, surrogate {ex[:, 1].mean():.3f}, "
          f"relative difference {(ex[:, 1] / ex[:, 0] - 1).mean():+.3f}")
    save["exits"] = ex

    for nm in (2, 20):
        for which in ("full", "sur"):
            Sb, v, n = (np.concatenate(x) for x in zip(*(res[s][which] for s in ref.SEEDS)))
            sel = n >= nm
            P = np.array_split(np.argsort(Sb[sel]), 25)
            save[f"curve_{which}_{nm}"] = np.array([[Sb[sel][p].mean() for p in P], [v[sel][p].mean() for p in P]])
    np.savez(OUT, **save)
    figure(save)

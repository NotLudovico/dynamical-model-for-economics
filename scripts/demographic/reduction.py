"""Step 1: is a firm's life determined by its field Phi_i(t) plus its own noise?

Full model: facts.npz (alpha 2.5, sigma 1.75, D 0.1, N 8000, C 200, seeds 0 and 1, samples every 0.5 from t=100).
Field: Phi_i = 1 - (a S)_i - fbar, so that the full equation is dS_i = S_i (Phi_i - S_i) dt + sqrt(D S_i) dW.
Surrogate: each node integrates that one-firm equation with its measured Phi_i(t) (linear interpolation),
fresh noise, death at 0 and rebirth at D, starting from the full model's sizes at t = 100.
Compare the life panels (same estimator as demo_batch)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "demographic")
import numpy as np
from relative_glv import model

D, DT, dt, N = 0.1, 0.5, 0.01, 8000
model._ALPHA_PL = 2.5
d = np.load(os.path.join(DATA, "facts.npz"))


def lives(S, L, ming):
    Sb, v = [], []
    T = S.shape[1]; lnS = np.log(S)
    for i in range(S.shape[0]):
        for seg in np.split(np.arange(T), np.flatnonzero(np.diff(L[i])) + 1):
            if seg.size < ming + 1:
                continue
            g = np.diff(lnS[i, seg])
            Sb.append(S[i, seg].mean()); v.append(np.sqrt(np.pi / 2) * np.abs(g - g.mean()).mean())
    return np.array(Sb), np.array(v)


def panel(Sb, v, nb=25):
    P = np.array_split(np.argsort(Sb), nb)
    x = np.log([Sb[p].mean() for p in P])
    M = {q: np.log([np.mean(v[p] ** q) for p in P]) for q in (1, 2, 3, 4)}
    z = np.array([-np.polyfit(x, M[q], 1)[0] for q in M])
    fifths = [-np.polyfit(x[i:i + 5], M[1][i:i + 5], 1)[0] for i in range(0, 25, 5)]
    return z, np.array(fifths), np.exp(x), np.exp(M[1])


res = {}
for seed in (0, 1):
    S_full = d["S"][seed].astype(float); L_full = d["L"][seed]
    a = model.coupling(N, 1.76, 1.75, kind="powerlaw", seed=seed, mean_degree=200)
    f = 1.0 - S_full - (a @ S_full)
    fbar = (S_full * f).sum(0) / N
    Phi = 1.0 - (a @ S_full) - fbar                   # (N, T)
    rng = np.random.default_rng(100 + seed)
    T = S_full.shape[1]; per = int(round(DT / dt))
    S = S_full[:, 0].copy(); life = np.zeros(N, np.int64)
    out_S, out_L = [S.copy()], [life.copy()]
    for k in range(T - 1):
        for j in range(per):
            ph = Phi[:, k] + (Phi[:, k + 1] - Phi[:, k]) * j / per
            S = S + S * (ph - S) * dt + np.sqrt(D * S * dt) * rng.standard_normal(N)
            dead = S <= 0
            if dead.any():
                S[dead] = D; life[dead] += 1
        out_S.append(S.copy()); out_L.append(life.copy())
    S_sur, L_sur = np.array(out_S).T, np.array(out_L).T
    res[seed] = dict(full=(S_full, L_full), sur=(S_sur, L_sur), phi_sd=Phi.std(1), phi_mean=Phi.mean(1))
    print(f"seed {seed}: surrogate mean size {S_sur.mean():.3f} (full 1), deaths per node per unit time "
          f"full {(L_full[:, -1] - L_full[:, 0]).mean() / 480:.3f}, surrogate {L_sur[:, -1].mean() / 480:.3f}")

for ming in (2, 10, 20):
    for which in ("full", "sur"):
        Z, F = [], []
        for seed in (0, 1):
            Sb, v = lives(*res[seed][which], ming)
            z, fi, *_ = panel(Sb, v)
            Z.append(z); F.append(fi)
        z = np.mean(Z, 0); fi = np.mean(F, 0)
        print(f"min g {ming:2d} {which:4s}: beta {z[0]:.3f}, ratios {np.round(z[1:] / z[0], 2)}, fifths {np.round(fi, 2)}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for k, ming in enumerate((2, 20)):
    for which, c, lab in (("full", "#104281", "full model"), ("sur", "#eb6834", "one firm in its measured field")):
        Sb, v = lives(*res[0][which], ming)
        *_, x, y = panel(Sb, v)
        ax[k].loglog(x, y, "o-" if which == "full" else "s--", color=c, lw=2, ms=4, label=lab)
    ax[k].set(title=f"lives with ≥ {ming} growth rates (seed 0)", xlabel="$\\bar S$", ylabel="E[v | $\\bar S$]")
    ax[k].legend(frameon=False, fontsize=8); ax[k].grid(True, color="#e6e6e3", lw=0.6)
fig.suptitle("Step 1: one-firm reduction test, α = 2.5, σ = 1.75, D = 0.1", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(DATA, "reduction.png"), dpi=115)

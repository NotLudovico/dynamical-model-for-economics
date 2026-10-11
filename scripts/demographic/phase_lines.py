"""Phase lines of the demographic-noise model from the static renewal theory (relative_glv.renewal), mu = 1.76:
instability line sigma_c(D) and zero-growth line sigma_0(D) for alpha 2.5 and 6 (N = 8000, C = 200 classes),
the fully connected check (sigma_c -> sqrt 2 as D -> 0), and a simulation check at alpha = 6, D = 0.1
(N = 4000, C = 100): temporal share of the field variance and turnover against sigma.
Run: uv run python scripts/demographic/phase_lines.py   -> data/demographic/phase_lines.npz, figure in the draft."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "demographic")
FIG = os.path.join(ROOT, "thesis", "assets", "figures")
import numpy as np
from joblib import Parallel, delayed

MU = 1.76
DS = np.array([0.003, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3])
ALPHAS = (2.5, 6.0)
SIGMAS_SIM = np.array([1.2, 1.3, 1.4, 1.45, 1.5, 1.6, 1.7])
OUT = os.path.join(DATA, "phase_lines.npz")


def classes(alpha, N, C):
    import relative_glv.model as M
    from relative_glv.hdmft import bin_degree_distribution, soft_configuration_kernel
    M._ALPHA_PL = alpha
    a = M.coupling(N, mu=0.0, sigma=1.0, kind="powerlaw", seed=0, mean_degree=C)
    cl = bin_degree_distribution(np.diff(a.indptr), n_bins=60)
    K = soft_configuration_kernel(cl["degree_fractions"], cl["weights"], cl["connectance"])["kernel"]
    return cl["weights"], K * cl["weights"][None, :] / cl["connectance"]


def lines(alpha):
    from relative_glv.renewal import renewal_sigma_c, renewal_zero_growth
    w, Kc = classes(alpha, 8000, 200)
    return [(renewal_sigma_c(MU, w, Kc, D), renewal_zero_growth(MU, w, Kc, D)) for D in DS]


def sim(sigma, seed, alpha=6.0, D=0.1, N=4000, C=100):
    from relative_glv import model
    model._ALPHA_PL = alpha
    a = model.coupling(N, MU, sigma, kind="powerlaw", seed=seed, mean_degree=C)
    r = model.integrate_demographic(a, D=D, tmax=500.0, t_start=200.0, dt_save=0.5, seed=seed)
    S, L = r["S"], r["life"]
    Phi = 1.0 - a @ S - (S * (1.0 - S - a @ S)).sum(0) / N
    return Phi.var(1).mean() / Phi.var(), (L[:, -1] - L[:, 0]).mean() / 300.0


if __name__ == "__main__" and not os.path.exists(OUT):
    from relative_glv.renewal import renewal_sigma_c, solve_renewal_fixed_point
    L = np.array(Parallel(n_jobs=2)(delayed(lines)(a) for a in ALPHAS))          # (alpha, D, 2)
    fc = np.array([renewal_sigma_c(MU, np.ones(1), np.ones((1, 1)), D) for D in DS])
    S = np.array(Parallel(n_jobs=8)(delayed(sim)(s, k) for s in SIGMAS_SIM for k in range(2))).reshape(-1, 2, 2).mean(1)
    w, Kc = classes(6.0, 4000, 100)
    th = [solve_renewal_fixed_point(MU, s, w, Kc, 0.1) for s in SIGMAS_SIM]
    np.savez(OUT, D=DS, alphas=ALPHAS, sigma_c=L[..., 0], sigma_0=L[..., 1], fc_sigma_c=fc,
             sigmas_sim=SIGMAS_SIM, share_sim=S[:, 0], turnover_sim=S[:, 1],
             share_th=[t["temporal_share"] for t in th], turnover_th=[t["turnover"] for t in th],
             radius_th=[t["stability_radius"] for t in th], sigma_c_check=renewal_sigma_c(MU, w, Kc, 0.1))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    d = np.load(OUT)
    print("D:", d["D"])
    print("fully connected sigma_c(D):", np.round(d["fc_sigma_c"], 4), " (deterministic sqrt 2 = 1.4142)")
    for k, a in enumerate(d["alphas"]):
        print(f"alpha {a}: sigma_c {np.round(d['sigma_c'][k], 4)}\n           sigma_0 {np.round(d['sigma_0'][k], 4)}")
    print(f"check alpha 6, D 0.1, N 4000: sigma_c {float(d['sigma_c_check']):.3f}")
    for s, a, b, c, e in zip(d["sigmas_sim"], d["share_sim"], d["share_th"], d["turnover_sim"], d["turnover_th"]):
        print(f"   sigma {s:.2f}: temporal share sim {a:.3f} static {b:.3f} | turnover sim {c:.3f} theory {e:.3f}")

    BLUE, LIGHT, ORANGE, INK = "#104281", "#86b6ef", "#eb6834", "#333333"
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for k, (a, c) in enumerate(zip(d["alphas"], (BLUE, LIGHT))):
        ax[0].plot(d["D"], d["sigma_c"][k], "o-", color=c, lw=2, ms=4, label=f"$\\sigma_c(D)$, α = {a:g}")
        ax[0].plot(d["D"], d["sigma_0"][k], "s--", color=c, lw=1.5, ms=4, label=f"zero growth, α = {a:g}")
    ax[0].plot(d["D"], d["fc_sigma_c"], ":", color=INK, lw=1.5, label="$\\sigma_c(D)$, fully connected")
    ax[0].plot([0.1], [1.70], "*", color=ORANGE, ms=12, label="reference point")
    ax[0].set(xlabel="demographic noise D", ylabel="disorder σ", title="(a) static theory, μ = 1.76")
    ax[1].plot(d["sigmas_sim"], d["share_sim"], "o-", color=BLUE, lw=2, label="simulation")
    ax[1].plot(d["sigmas_sim"], d["share_th"], "s--", color=LIGHT, lw=2, label="static theory (no feedback)")
    ax[1].axvline(float(d["sigma_c_check"]), color=ORANGE, lw=1.5, ls=":", label="theory $\\sigma_c$")
    ax[1].set(xlabel="disorder σ", ylabel="temporal share of field variance",
              title="(b) α = 6, D = 0.1, N = 4000")
    for a in ax:
        a.grid(True, color="#e6e6e3", lw=0.6); a.legend(fontsize=7, frameon=False)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "demographic_phase_lines.png"), dpi=200)

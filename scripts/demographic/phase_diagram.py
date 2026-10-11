"""Phase diagram of the demographic-noise model in the (mu, sigma) plane at D = 0.1, from the static renewal
theory (relative_glv.renewal): instability line sigma_c(mu), above which the composition keeps fluctuating,
and zero-growth line sigma_0(mu), above which the economy grows.  Degree classes from the reference graph
(alpha 2.5, N 32000, C 200, numpy configuration model), from an alpha = 6 graph, and the complete graph.
Paper sign: mu < 0 is competition (the code uses the opposite sign).
Run: uv run python scripts/demographic/phase_diagram.py   -> data/demographic/phase_diagram.npz,
     thesis/assets/figures/demographic_phase_diagram.png   (add "plot" to redraw from the npz)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

D = 0.1
MUS = np.round(np.linspace(0.0, 2.5, 11), 3)            # code sign: competition > 0
GRAPHS = (("alpha 2.5", 2.5), ("alpha 6", 6.0), ("complete", None))
REF = (-1.76, 1.70)                                     # paper sign
OUT = os.path.join(ROOT, "data", "demographic", "phase_diagram.npz")
FIG = os.path.join(ROOT, "thesis", "assets", "figures", "demographic_phase_diagram.png")


def classes(alpha, N=32000, C=200):
    if alpha is None:
        return np.ones(1), np.ones((1, 1))
    from relative_glv.model import configuration_coupling
    from relative_glv.hdmft import bin_degree_distribution, soft_configuration_kernel
    a = configuration_coupling(N, 0.0, 1.0, alpha=alpha, mean_degree=C, seed=0)
    cl = bin_degree_distribution(np.diff(a.indptr), n_bins=60)
    K = soft_configuration_kernel(cl["degree_fractions"], cl["weights"], cl["connectance"])["kernel"]
    return cl["weights"], K * cl["weights"][None, :] / cl["connectance"]


def point(alpha, mu):
    from relative_glv.renewal import renewal_sigma_c, renewal_zero_growth
    w, K = classes(alpha)
    out = []
    for f in (renewal_sigma_c, renewal_zero_growth):
        try:
            out.append(f(mu, w, K, D))
        except ValueError:                               # no root in the bracket
            out.append(np.nan)
    return out


def figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mu = -R["mus"]
    sc, s0 = R["sigma_c"], R["sigma_0"]                   # (graph, mu)
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    top = 2.2
    ax.fill_between(mu, sc[0], top, color="#dbe8f8", lw=0)
    ax.fill_between(mu, 0.9, sc[0], color="#f3f2ee", lw=0)
    for k, (name, col, ls) in enumerate((("$\\alpha=2.5$", "#14467f", "-"), ("$\\alpha=6$", "#4f93e0", "--"),
                                         ("complete graph", "#9a9890", ":"))):
        ax.plot(mu, sc[k], ls, color=col, lw=2, label=rf"$\sigma_c$, {name}")
    ok = np.isfinite(s0[0])
    ax.plot(mu[ok], s0[0][ok], "-", color="#eb6834", lw=2, label=r"zero growth, $\alpha=2.5$")
    ax.plot(*REF, "*", color="#eb6834", ms=14, mec="white", mew=0.8, label="reference point")
    ax.text(-2.42, 2.08, "fluctuating", fontsize=9, color="#14467f")
    ax.text(-2.42, 0.98, "relaxing", fontsize=9, color="#52514e")
    ax.text(-1.05, 1.62, "economy grows", fontsize=8, color="#eb6834", rotation=-38)
    ax.text(-2.42, 1.80, "shrinks", fontsize=8, color="#eb6834")
    ax.set(xlabel=r"mean interaction $\mu$ (competition $<0$)", ylabel=r"disorder $\sigma$",
           xlim=(mu.min(), mu.max()), ylim=(0.9, top), title=rf"$D={D}$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.5, loc="upper right")
    plt.tight_layout(); plt.savefig(FIG, dpi=200)
    print("wrote", FIG)


if __name__ == "__main__":
    if sys.argv[1:] != ["plot"]:
        from joblib import Parallel, delayed
        jobs = [(a, m) for _, a in GRAPHS for m in MUS]
        res = Parallel(n_jobs=11, verbose=5)(delayed(point)(a, m) for a, m in jobs)
        res = np.array(res).reshape(len(GRAPHS), len(MUS), 2)
        np.savez(OUT, mus=MUS, graphs=[g for g, _ in GRAPHS], sigma_c=res[..., 0], sigma_0=res[..., 1])
    R = np.load(OUT)
    for g, sc, s0 in zip(R["graphs"], R["sigma_c"], R["sigma_0"]):
        print(f"{g:10s} sigma_c {' '.join(f'{x:.3f}' for x in sc)}\n{'':10s} sigma_0 {' '.join(f'{x:.3f}' for x in s0)}")
    print("mu (paper sign):", " ".join(f"{-m:.2f}" for m in R["mus"]))
    figure(R)

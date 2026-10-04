"""Figure: the kill rule and the lifetime law of the one-firm process (alpha = 6 fitness).
(a) a fitness path Phi(t) with its dips, coloured by whether their area exceeds ell_c(s), and ln S with
the listing threshold; (b) P_kill(u): ODE, Slepian model, large-u asymptote; (c) mean lifetime: ODE
against Phi(u) / (nu(u) P_kill(u)) with the Slepian P_kill.
Run: uv run python scripts/lifecycle/plot_lifetime_law.py  -> thesis/assets/figures/lifetime_law.png"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, HERE]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from joblib import Parallel, delayed
from kill_rule import simulate, dips, spells_ode, lc, pkill_slepian, pkill_asymptotic, F, TAU, LAM, THR, DTI

OUT = os.path.join(ROOT, "thesis", "assets", "figures", "lifetime_law.png")


def ode_point(u, seed):
    T = 1500.0 if u < 1.5 else 6000.0
    n = 300 if u < 1.5 else 120
    P, X = simulate(np.full(n, u * F), F, TAU, LAM, 31 + seed, t_end=T + 50.0)
    L, frac = spells_ode(X)
    return dips(P, X)[:, 5].mean(), L.mean(), T


def main():
    u_ode = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    R = Parallel(n_jobs=6)(delayed(ode_point)(u, i) for i, u in enumerate(u_ode))
    pk_ode = np.array([r[0] for r in R]); L_ode = np.array([r[1] for r in R]); T_ode = np.array([r[2] for r in R])
    u_th = np.linspace(-0.5, 5.0, 23)
    pk_sl = np.array(Parallel(n_jobs=6)(delayed(pkill_slepian)(u) for u in u_th))
    u_as = u_th[u_th > 0.4]; pk_as = np.array([pkill_asymptotic(u) for u in u_as])
    nu = lambda u: np.exp(-u ** 2 / 2) / (2 * np.pi * TAU)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    P, X = simulate(np.array([0.5 * F]), F, TAU, LAM, 3, t_end=260.0)
    t = DTI * np.arange(P.shape[1]); sel = t < 200
    p, x = P[0, sel], X[0, sel]; t = t[sel]
    a = ax[0]
    a.plot(t, p, color="#264653", lw=1.2, label=r"fitness $\Phi(t)$")
    neg = p < 0
    d = np.diff(neg.astype(int)); st, en = np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1
    if en.size and st.size and en[0] < st[0]: en = en[1:]
    for s0, e0 in zip(st, en):
        s = (p[s0 - 1] - p[s0 + 1]) / (2 * DTI); area = -p[s0:e0].sum() * DTI
        a.fill_between(t[s0:e0], p[s0:e0], 0, color="#e63946" if area > lc(s) else "#a8dadc", lw=0)
    a.axhline(0, color="grey", lw=0.6)
    b2 = a.twinx()
    b2.plot(t, np.log(x), color="#e76f51", lw=1.4, label=r"$\ln S$")
    b2.axhline(np.log(THR), color="#e76f51", ls=":", lw=1)
    b2.set_ylabel(r"$\ln S$", color="#e76f51")
    a.set_xlabel("time"); a.set_ylabel(r"$\Phi$")
    a.set_title(r"(a) dips that kill (red, area $>\ell_c$) and that merge (light)", fontsize=10)

    a = ax[1]
    a.plot(u_th, pk_sl, color="#264653", lw=2, label="Slepian model")
    a.plot(u_as, pk_as, color="#264653", ls="--", lw=1.2, label=r"large-$u$, $\propto e^{-c u^{4/3}}$")
    a.scatter(u_ode, pk_ode, color="#e76f51", zorder=5, label="one-firm ODE")
    a.set_yscale("log"); a.set_xlabel(r"$u=\bar\Phi/f$"); a.set_ylabel(r"$P_{\rm kill}$")
    a.set_title("(b) probability that a dip ends the spell", fontsize=10); a.legend(fontsize=8.5)

    a = ax[2]
    a.plot(u_th, norm.cdf(u_th) / (nu(u_th) * pk_sl), color="#264653", lw=2, label=r"$\Phi_N(u)/\nu(u)P_{\rm kill}(u)$")
    a.plot(u_th, norm.cdf(u_th) / nu(u_th), color="grey", lw=1, ls=":", label="single Rice excursion")
    ok = L_ode < 0.1 * T_ode                     # mean spell over a tenth of the run: biased short
    a.scatter(u_ode[ok], L_ode[ok], color="#e76f51", zorder=5, label="one-firm ODE")
    a.scatter(u_ode[~ok], L_ode[~ok], facecolor="white", edgecolor="#e76f51", zorder=5, label="ODE, run-limited")
    a.set_yscale("log"); a.set_xlim(-0.7, 3.2); a.set_ylim(3, 3e4)
    a.set_xlabel(r"$u$"); a.set_ylabel(r"mean lifetime $\bar L$")
    a.set_title("(c) lifetime law", fontsize=10); a.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT, dpi=160); print("wrote", OUT)


if __name__ == "__main__":
    main()

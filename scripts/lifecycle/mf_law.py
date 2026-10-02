"""The (m, f) law of the per-firm fitness Phi_i = m_i + f_i eta_i from mean-field theory.

HDMFT at gamma = 0: Phi = 1 - g - mu kappa Bbar + sigma sqrt(kappa) xi, <xi(t) xi(t')> = Q(t - t'), with
Bbar and Q degree-weighted neighbour averages of S and S(t)S(t'). Splitting Q into its static part Q_T
(window-mean sizes) and the fluctuating rest gives, per degree class kappa = k / C,
    m | kappa ~ N(1 - gbar - mu kappa Bbar, sigma^2 kappa Q_T),   f(kappa) = sigma sqrt(kappa (Q_0 - Q_T)),
    rho(L) = (Q(L) - Q_T) / (Q_0 - Q_T).
A  cavity check: order parameters from the simulation, law against the measured (m_i, f_i);
B  closure: order parameters from the two-time HDMFT solver, no simulation input;
C  single-firm surrogate run with (m, f) sampled from each law, against the full model.
Run: uv run python scripts/lifecycle/mf_law.py   (needs data/lifecycle/fit_*.npz from fitness_stats.py)
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
from beta_spells import model, MU, N, C, START, DT
from relative_glv.hdmft import bin_degree_distribution, soft_configuration_kernel, solve_twotime_heterogeneous
from surrogate import surrogate

T, LAM, LAGS = 480.0, 1e-3, 61
POINTS = ((6.0, 1.75), (2.5, 1.75), (3.5, 1.9))


def cavity(alpha, sigma, seed):
    """Order parameters from one simulated economy and the per-firm predicted law."""
    model._ALPHA_PL = alpha
    a = model.coupling(N, MU, sigma, kind="powerlaw", seed=seed, mean_degree=C)
    r = model.integrate(a, tmax=START + T, n_eval=int((START + T) / DT) + 1, lam=LAM, seed=seed,
                        method="RK45", rtol=1e-4, atol=1e-7)
    t, W = r["t"], r["W"]
    win = t >= START - 1e-9
    S = N * W[:, win]
    k = np.diff(a.indptr).astype(float)
    kap = k / k.mean()                                         # C_eff = realised mean degree
    w = k / k.sum()
    g = (W[:, win] * (1 - S - np.asarray(a @ S))).sum(0)
    Sb = S.mean(1)
    Q = np.array([np.sum(w * np.mean(S[:, :S.shape[1] - L] * S[:, L:], 1)) for L in range(LAGS)])
    QT, B = np.sum(w * Sb ** 2), np.sum(w * Sb)
    acf = (Q - QT) / (Q[0] - QT)
    return dict(r=1 - g.mean() - MU * kap * B, s=sigma * np.sqrt(kap * QT), f=sigma * np.sqrt(kap * (Q[0] - QT)),
                kap=kap, acf=acf, B=B, Q0=Q[0], QT=QT, gbar=g.mean())


def hdmft(alpha, sigma, k0):
    """Two-time HDMFT on the degree sequence k0; per-class r, static sd, f and the pooled ACF."""
    b = bin_degree_distribution(k0, 25)
    ker = soft_configuration_kernel(b["degree_fractions"], b["weights"], b["connectance"])["kernel"]
    res = solve_twotime_heterogeneous(MU, sigma, ker, b["connectance"], lam=LAM, n_per_class=200,
                                      Nt=1440, dt=0.125, iters=20, burn=10)
    t, q, dt = res["t"], res["q"], res["t"][1] - res["t"][0]
    il = np.where(t >= 120)[0]
    Ql = [q[p][np.ix_(il, il)] for p in range(q.shape[0])]
    q0 = np.array([np.mean(np.diag(X)) for X in Ql])
    qinf = np.array([np.mean(X[np.triu_indices(il.size, 200)]) for X in Ql])      # lags > 25
    step = int(round(DT / dt))
    acf = np.mean([np.array([np.mean(np.diag(X, l)) for l in range(0, step * LAGS, step)]) - qi
                   for X, qi in zip(Ql, qinf)], 0)
    late = t >= 120
    return dict(kap=b["representative_degrees"] / k0.mean(), r=1 - res["g_t"][late].mean() - MU * res["B"][:, late].mean(1),
                s=sigma * np.sqrt(qinf), f=sigma * np.sqrt(np.maximum(q0 - qinf, 0)), acf=acf / acf[0], err=res["err"])


def law_file(name, m, f, acf):
    path = f"{DATA}/law_{name}_l{LAM:.0e}.npz"
    np.savez(path, m=m, f=f, acf=acf)
    return path


def fmt(z):
    return " ".join(f"{x:.2f}" for x in z)


if __name__ == "__main__":
    for alpha, sigma in POINTS:
        fit = np.load(f"{DATA}/fit_a{alpha}_s{sigma}_l{LAM:.0e}.npz")
        m_meas, f_meas = fit["m"], fit["f"]
        cav = Parallel(n_jobs=8)(delayed(cavity)(alpha, sigma, s) for s in range(8))
        r = np.concatenate([c["r"] for c in cav]); s = np.concatenate([c["s"] for c in cav])
        fp = np.concatenate([c["f"] for c in cav]); kap = np.concatenate([c["kap"] for c in cav])
        z = (m_meas - r) / s
        print(f"\nalpha={alpha} sigma={sigma}")
        print(f"  A cavity: gbar {np.mean([c['gbar'] for c in cav]):+.3f}, Bbar {np.mean([c['B'] for c in cav]):.3f}, "
              f"Q0 {np.mean([c['Q0'] for c in cav]):.3f}, QT {np.mean([c['QT'] for c in cav]):.3f}")
        print(f"    standardised static field z = (m - r)/s: mean {z.mean():+.3f} sd {z.std():.3f} "
              f"skew {skew(z):+.2f} exkurt {kurtosis(z):+.2f}   (law: 0, 1, 0, 0)")
        print(f"    f_measured / f_law: median {np.median(f_meas / fp):.3f}, IQR "
              f"{np.percentile(f_meas / fp, 25):.3f}-{np.percentile(f_meas / fp, 75):.3f}")
        rng = np.random.default_rng(1)
        m_cav = r + s * rng.standard_normal(r.size)
        print(f"    m: sd {m_meas.std():.2f} vs law {m_cav.std():.2f}; skew {skew(m_meas):+.2f} vs {skew(m_cav):+.2f}; "
              f"corr(m,f) {np.corrcoef(m_meas, f_meas)[0, 1]:+.2f} vs {np.corrcoef(m_cav, fp)[0, 1]:+.2f}")
        acf_cav = np.mean([c["acf"] for c in cav], 0)
        model._ALPHA_PL = alpha                     # read at call time; workers set their own copy in cavity()
        hd = hdmft(alpha, sigma, np.diff(model.coupling(N, MU, sigma, kind="powerlaw", seed=0, mean_degree=C).indptr).astype(float))
        lk = np.log(kap)
        r_h, s_h, f_h = (np.interp(lk, np.log(hd["kap"]), hd[x]) for x in ("r", "s", "f"))
        print(f"  B HDMFT (rel.err {hd['err']:.3f}): per firm, law HDMFT / law cavity: "
              f"mean {np.median(r_h - r):+.3f} (shift), static sd {np.median(s_h / s):.3f}, f {np.median(f_h / fp):.3f}")
        print(f"    ACF at lag 0.5, 2, 8: sim {fmt(fit['acf'][[1, 4, 16]])} | cavity {fmt(acf_cav[[1, 4, 16]])} | "
              f"HDMFT {fmt(hd['acf'][[1, 4, 16]])}")
        m_h = r_h + s_h * rng.standard_normal(r.size)
        full = fit["Z"].mean(0)
        print(f"  C spells, T=480: full model beta {np.median(fit['beta']):.3f} ratios {fmt(full[1:] / full[0])}")
        for name, m, f, acf in (("measured", m_meas, f_meas, fit["acf"]), ("cavity", m_cav, fp, acf_cav),
                                ("hdmft", m_h, f_h, hd["acf"])):
            b, zr, tau = surrogate(law_file(f"{name}_a{alpha}_s{sigma}", m, f, acf))
            print(f"    surrogate, {name:8s} law: beta {b:.3f} ratios {fmt(zr)} (tau {tau:.2f})", flush=True)

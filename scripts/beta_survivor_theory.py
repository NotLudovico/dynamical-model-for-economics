"""Survivor-ensemble theory of beta (thesis/standalone/beta_theory, Secs. 4 and 7).

Leading order in 1/C. The degree-resolved two-time HDMFT is solved on the seed-0 graph of
each alpha (as in beta_ad_hdmft.py). From it we read the order parameters of the static
survivor ensemble:
  r_p = 1 - g - mu B_p             deterministic load of class p
  a_p = sigma sqrt(q_T,p)          static (window-mean) field amplitude
  f_p = sigma sqrt(q_0,p - q_T,p)  fluctuating field amplitude
  n   = T / (2 tau)                independent blocks of the window, tau = correlation time
The model: Sbar = r + a x (x ~ N(0,1)); survive with prob Phi(Sbar/f)^n (extreme value).
It is compared with HDMFT's own survivors (survival, eps_k, V_S). Also reported:
  - the universality of the survival curve P(live | Phibar/f) across degrees (alpha = 3.5)
  - beta_ad on realized field increments and the fluctuation selection eps_chi
  - beta_inf = beta_ad - eps_chi - eps_k^EV, and beta_8000 = beta_inf - eps_n - eps_z
    (closed forms from data/beta_within_degree_N8000.npz)
  - the window-length test: simulated survival / eps_k / beta at T = 30, 60, 120 against
    the model's prediction with the same order parameters.

Run: uv run python scripts/beta_survivor_theory.py   (-> data/beta_survivor_theory.npz, ~12 min)
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from relative_glv import model
from relative_glv.hdmft import (bin_degree_distribution, soft_configuration_kernel,
                                solve_twotime_heterogeneous)
from relative_glv.msb import _decline_beta

MU, SIGMA, LAM, N = 1.76, 1.75, 1e-3, 8000
C = N // 40
ALPHAS = (2.2, 2.5, 3.5, 6.0)
DELTA, T0, FLOOR = 0.5, 60.0, 1e-6 * N
OUT = os.path.join(ROOT, "data", "beta_survivor_theory.npz")


def mad(g):
    return np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)


def decline_bins(Sb, v, live):
    beta, _, bx, _, pk = _decline_beta(Sb[live], v[live])
    idx = np.where(live)[0]
    return beta, np.array_split(idx[np.argsort(Sb[live])], 20)[pk:]


def slope(Sb, y, P, agg=np.mean):
    bS = np.array([Sb[p].mean() for p in P])
    return np.polyfit(np.log(bS), np.log([agg(y[p]) for p in P]), 1)[0]


def ensemble_stats(kap, S, live, Speak=1.1):
    """Survival, eps_k and V_S of a survivor ensemble, on the decline range S >= Speak."""
    dec = live & (S >= Speak)
    o = np.where(dec)[0][np.argsort(S[dec])]
    P = np.array_split(o, 20)
    return dict(surv=live.mean(), eps_k=slope(S, np.sqrt(kap), P), V_S=np.log(S[dec]).var())


def graph(alpha, seed):
    model._ALPHA_PL = alpha                         # read at call time by _powerlaw_adjacency
    return model.coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)


def solve_hdmft(alpha):
    k0 = np.diff(graph(alpha, 0).indptr).astype(float)
    b = bin_degree_distribution(k0, 25)
    ker = soft_configuration_kernel(b["degree_fractions"], b["weights"], b["connectance"])["kernel"]
    r = solve_twotime_heterogeneous(MU, SIGMA, ker, b["connectance"], lam=LAM, n_per_class=200,
                                    Nt=1440, dt=0.125, iters=20, burn=10)
    r["kap_c"] = b["representative_degrees"] / C
    return r


def order_parameters(r):
    t, q = r["t"], r["q"]
    late = t >= 120
    il = np.where(late)[0]
    dt = t[1] - t[0]
    Ql = [q[p][np.ix_(il, il)] for p in range(q.shape[0])]
    q0 = np.array([np.mean(np.diag(Q)) for Q in Ql])
    qT = np.array([Q.mean() for Q in Ql])
    qinf = np.array([np.mean(Q[np.triu_indices(il.size, 200)]) for Q in Ql])   # lags > 25
    taus = []
    for Q in Ql:
        c = np.array([np.mean(np.diag(Q, l)) for l in range(320)]) - Q.mean()
        taus.append(dt * (0.5 + np.clip(c[1:] / c[0], 0, None).sum()))
    return dict(r=1 - r["g_t"][late].mean() - MU * r["B"][:, late].mean(1),
                q0=q0, qT=qT, qinf=qinf, tau=float(np.median(taus)), kap_c=r["kap_c"], Ql=Ql,
                late=late, dt=dt)


def ev_model(op, T, rng, M=400_000):
    """Static survivor ensemble with extreme-value survival Phi(S/f)^(T/2tau)."""
    qT = op["qinf"] + (op["qT"] - op["qinf"]) * T0 / T            # window-mean variance ~ 1/T
    a, f = SIGMA * np.sqrt(qT), SIGMA * np.sqrt(np.maximum(op["q0"] - qT, 1e-9))
    cls = rng.integers(0, op["kap_c"].size, M)
    S = op["r"][cls] + a[cls] * rng.standard_normal(M)
    live = (S > 0) & (rng.uniform(size=M) < norm.cdf(S / f[cls]) ** (T / (2 * op["tau"])))
    return ensemble_stats(op["kap_c"][cls], np.where(live, S, 1e-9), live)


def hdmft_side(r, op):
    t, U = r["t"], r["U"].astype(float)
    late, dt = op["late"], op["dt"]
    st = int(round(DELTA / dt))
    ncls, npc, _ = U.shape
    cls = np.repeat(np.arange(ncls), npc)
    U = U.reshape(-1, t.size)[:, late]
    live = U.min(1) > FLOOR
    Sb = U.mean(1)
    v = mad(np.diff(np.log(np.maximum(U[:, ::st], 1e-300)), axis=1))
    Phi = np.gradient(np.log(U), dt, axis=1) + U - LAM * (1 / U - 1)   # exact inversion of the process
    Dr = np.var(np.diff(Phi[:, ::st], axis=1), axis=1)
    Dc = np.array([SIGMA ** 2 * np.mean(np.diag(Q)[:-st] + np.diag(Q)[st:] - 2 * np.diag(Q, st))
                   for Q in op["Ql"]])[cls]
    f = SIGMA * np.sqrt(op["q0"] - op["qT"])[cls]
    beta, P = decline_bins(Sb, v, live)
    kap = op["kap_c"][cls]
    own = ensemble_stats(kap, np.where(live, Sb, 1e-9), live)
    u = Phi.mean(1) / f
    edges = np.arange(0.0, 2.01, 0.25)
    half = cls < ncls // 2
    curve = np.array([[live[m & (u >= edges[i]) & (u < edges[i + 1])].mean() for i in range(edges.size - 1)]
                      for m in (half, ~half)])
    return dict(beta_HD=beta, surv=own["surv"], eps_k=own["eps_k"], V_S=own["V_S"],
                beta_ad=1 - slope(Sb, v * Sb / np.sqrt(Dr), P, np.median),
                eps_chi=slope(Sb, np.sqrt(Dr / Dc), P), curve=curve, edges=edges)


def window_sim(alpha, seed, windows=(30.0, 60.0, 120.0), end=360.0):
    a = graph(alpha, seed)
    r = model.integrate(a, tmax=end, n_eval=1441, lam=LAM, seed=seed, method="RK45", rtol=1e-4, atol=1e-7)
    t, kap = r["t"], np.diff(a.indptr) / C
    out = {}
    for T in windows:
        S = N * r["W"][:, t >= end - T - 1e-9][:, ::2]
        live = S.min(1) > FLOOR
        Sb = S.mean(1)
        v = mad(np.diff(np.log(np.maximum(S, 1e-300)), axis=1))
        beta, P = decline_bins(Sb, v, live)
        out[T] = (live.mean(), slope(Sb, np.sqrt(kap), P), beta)
    return out


if __name__ == "__main__":
    t0, rng = time.time(), np.random.default_rng(0)
    within = np.load(os.path.join(ROOT, "data", "beta_within_degree_N8000.npz"))
    wk = list(within["keys"])
    rows, ops = [], {}
    for alpha in ALPHAS:
        r = solve_hdmft(alpha)
        op = order_parameters(r)
        ops[alpha] = op
        hd, ev = hdmft_side(r, op), ev_model(op, T0, rng)
        wrow = within["table"][list(within["alphas"]).index(alpha)]
        eps_fin = wrow[wk.index("eps_n_th")] + wrow[wk.index("eps_z_th")]
        beta_inf = hd["beta_ad"] - hd["eps_chi"] - ev["eps_k"]
        rows.append(dict(alpha=alpha, tau=op["tau"], n=T0 / (2 * op["tau"]),
                         surv_HD=hd["surv"], surv_EV=ev["surv"], eps_k_HD=hd["eps_k"], eps_k_EV=ev["eps_k"],
                         V_S_HD=hd["V_S"], V_S_EV=ev["V_S"], beta_ad=hd["beta_ad"], eps_chi=hd["eps_chi"],
                         beta_inf=beta_inf, beta_HD=hd["beta_HD"], beta_8000=beta_inf - eps_fin,
                         beta_sim=wrow[wk.index("beta")]))
        if alpha == 3.5:
            print("survival curve P(live | Phibar/f), alpha=3.5, low / high degree half:")
            print("  u   " + " ".join(f"{e:5.2f}" for e in hd["edges"][:-1]))
            for name, c in zip(("low ", "high"), hd["curve"]):
                print(f"  {name}" + " ".join(f"{x:5.2f}" for x in c))
        print(f"alpha={alpha}: HDMFT done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    keys = list(rows[0])
    print("\n" + " ".join(f"{k:>9}" for k in keys))
    for row in rows:
        print(" ".join(f"{row[k]:9.3f}" for k in keys))

    print("\nwindow-length test (simulation medians over 8 economies vs EV model):")
    wtab = []
    for alpha in (2.5, 3.5):
        sims = Parallel(n_jobs=4)(delayed(window_sim)(alpha, s) for s in range(8))
        for T in (30.0, 60.0, 120.0):
            s_surv, s_eps, s_beta = np.median(np.array([x[T] for x in sims]), 0)
            ev = ev_model(ops[alpha], T, rng)
            wtab.append((alpha, T, s_surv, ev["surv"], s_eps, ev["eps_k"], s_beta))
            print(f"  alpha={alpha} T={T:5.0f}: surv sim {s_surv:.3f} / EV {ev['surv']:.3f} | "
                  f"eps_k sim {s_eps:.3f} / EV {ev['eps_k']:.3f} | beta sim {s_beta:.3f}", flush=True)
    np.savez(OUT, keys=np.array(keys), table=np.array([[row[k] for k in keys] for row in rows]),
             window=np.array(wtab))

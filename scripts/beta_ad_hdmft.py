"""Closing beta_ad with the two-time HDMFT (thesis/standalone/beta_theory, Sec. 6).

HDMFT side, per degree exponent alpha: solve the degree-resolved two-time HDMFT on the
realized degree sequence of the seed-0 graph (25 classes, SCM kernel, 200 paths per class,
dt=0.125, 180 time units, lambda=1e-3), then on its surviving paths measure
  beta_HD, beta_ad_HD (slope of v S / sqrt(D_class)), eps_k_HD, the field roughness
  D/Var of the class field, and the fully theoretical
  beta_th = beta_HD - eps_n - eps_z, with c_ds, V_S (+ static composition spread V_Rs),
  E[1/k] and the static z-score x all taken from HDMFT paths.
Simulation side (4 economies, output every 0.05 in the late window):
  beta_ad_sim, and the field roughness D/Var, measured and as the neighbour-weighted
  mixture E_w[d_j] / E_w[Var_t S_j].

Run: uv run python scripts/beta_ad_hdmft.py   (-> data/beta_ad_hdmft.npz, ~8 min)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from relative_glv import model
from relative_glv.hdmft import (bin_degree_distribution, soft_configuration_kernel,
                                solve_twotime_heterogeneous)
from relative_glv.msb import _decline_beta

MU, SIGMA, LAM, N = 1.76, 1.75, 1e-3, 8000
C = N // 40
ALPHAS, SEEDS = (2.2, 2.5, 3.5, 6.0), 4
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "beta_ad_hdmft.npz")
KEYS = ("beta_HD", "beta_ad_HD", "eps_k_HD", "rough_HD", "c_ds_HD", "eps_n_HD", "eps_z_HD", "beta_th",
        "beta_ad_sim", "rough_sim", "rough_mix")


def mad(g):
    return np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)


def decline(Sb, v, live):
    beta, _, bx, _, pk = _decline_beta(Sb[live], v[live])
    idx = np.where(live)[0]
    P = np.array_split(idx[np.argsort(Sb[live])], 20)[pk:]
    return beta, live & (Sb >= bx[pk]), P


def bin_slope(Sb, y, P, agg=np.median):
    bS = np.array([Sb[p].mean() for p in P])
    return np.polyfit(np.log(bS), np.log([agg(y[p]) for p in P]), 1)[0]


def graph(alpha, seed):
    model._ALPHA_PL = alpha                         # read at call time by _powerlaw_adjacency
    return model.coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)


def hdmft(alpha):
    k0 = np.diff(graph(alpha, 0).indptr).astype(float)
    b = bin_degree_distribution(k0, 25)
    ker = soft_configuration_kernel(b["degree_fractions"], b["weights"], b["connectance"])["kernel"]
    r = solve_twotime_heterogeneous(MU, SIGMA, ker, b["connectance"], lam=LAM, n_per_class=200,
                                    Nt=1440, dt=0.125, iters=20, burn=10)
    t, q = r["t"], r["q"]
    st = int(round(0.5 / (t[1] - t[0])))
    late = t >= 120
    il = np.where(late)[0]
    ncls, npc, _ = r["U"].shape
    cls = np.repeat(np.arange(ncls), npc)
    U = r["U"].reshape(-1, t.size)[:, late].astype(float)
    k = np.repeat(b["representative_degrees"], npc)
    live = U.min(1) > 1e-6 * N
    Sb = U.mean(1)
    v = mad(np.diff(np.log(np.maximum(U[:, ::st], 1e-300)), axis=1))
    dS = np.var(np.diff(U[:, ::st], axis=1), axis=1)
    Ql = [q[p][np.ix_(il, il)] for p in range(ncls)]
    Dc = np.array([SIGMA ** 2 * np.mean(np.diag(Q)[:-st] + np.diag(Q)[st:] - 2 * np.diag(Q, st)) for Q in Ql])
    Vc = np.array([SIGMA ** 2 * (np.mean(np.diag(Q)) - Q.mean()) for Q in Ql])
    qinf = np.array([np.mean(Q[np.triu_indices(il.size, 200)]) for Q in Ql])   # frozen part, lags > 25
    beta, dec, P = decline(Sb, v, live)
    w = k / k.sum()
    inv_k = np.mean(1 / k[dec])
    V_S = np.log(Sb[dec]).var() + 0.25 * inv_k * ((w @ Sb ** 4) * w.sum() / (w @ Sb ** 2) ** 2 - 1)
    c_ds = (w @ (dS * Sb ** 2)) / ((w @ dS) * (w @ Sb ** 2))
    r_p = 1 - r["g_t"][late].mean() - MU * r["B"][:, late].mean(1)
    x = (Sb - r_p[cls]) / (SIGMA * np.sqrt(np.maximum(qinf[cls], 1e-12)))
    eps_n = (c_ds - 1) * inv_k / (4 * V_S)
    eps_z = bin_slope(Sb, 1 + np.minimum(c_ds / k, 0.9) * (x ** 2 - 1), P, np.mean) / 2
    return dict(beta_HD=beta, beta_ad_HD=1 - bin_slope(Sb, v * Sb / np.sqrt(Dc[cls]), P),
                eps_k_HD=bin_slope(Sb, np.sqrt(k), P, np.mean), rough_HD=np.median(Dc / Vc),
                c_ds_HD=c_ds, eps_n_HD=eps_n, eps_z_HD=eps_z, beta_th=beta - eps_n - eps_z)


def simulation(alpha, seed):
    a = graph(alpha, seed)
    r = model.integrate(a, tmax=240.0, n_eval=4801, lam=LAM, seed=seed,
                        method="RK45", rtol=1e-6, atol=1e-9)
    S = N * r["W"][:, r["t"] >= 180 - 1e-9][:, ::10]                # Delta = 0.5
    k = np.diff(a.indptr).astype(float)
    live = S.min(1) > 1e-6 * N
    Sb = S.mean(1)
    v = mad(np.diff(np.log(np.maximum(S, 1e-300)), axis=1))
    aS = a @ S
    Phi = 1 - np.mean(S * (1 - S - aS), axis=0)[None, :] - aS          # fitness excluding self
    D = np.var(np.diff(Phi, axis=1), axis=1)
    beta, dec, P = decline(Sb, v, live)
    if not np.isfinite(beta):
        return None
    az = a.copy(); az.data -= MU / (a.nnz / N)
    h = az @ S
    d = np.var(np.diff(S, axis=1), axis=1)
    return dict(beta_ad_sim=1 - bin_slope(Sb, v * Sb / np.sqrt(D), P),
                rough_sim=np.median(np.var(np.diff(h[dec], axis=1), axis=1) / np.var(h[dec], axis=1)),
                rough_mix=(k @ d) / (k @ S.var(1)))


if __name__ == "__main__":
    t0, table = time.time(), []
    for alpha in ALPHAS:
        row = hdmft(alpha)
        sims = [s for s in Parallel(n_jobs=3)(delayed(simulation)(alpha, s) for s in range(SEEDS)) if s]
        row.update({q: np.median([s[q] for s in sims]) for q in sims[0]})
        table.append([row[q] for q in KEYS])
        print(f"alpha={alpha}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    table = np.array(table)
    np.savez(OUT, alphas=np.array(ALPHAS), keys=np.array(KEYS), table=table)
    print("\n" + " ".join(f"{q:>11}" for q in ("alpha",) + KEYS))
    for a, row in zip(ALPHAS, table):
        print(f"{a:11.1f} " + " ".join(f"{v:11.3f}" for v in row))

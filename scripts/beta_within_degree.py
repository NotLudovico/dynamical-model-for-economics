"""Closure of the within-degree channel of beta (thesis/standalone/beta_theory, Sec. 5).

Per degree exponent alpha (N=8000, C=N/40, thesis operating point, eight economies,
never pooled), on each economy's decline branch:
  ratio     D_i / sum_j z_ij^2 d_j              field increments are a neighbour sum
  n_eff     (sum c_j)^2 / sum c_j^2, c_j = z_ij^2 d_j   effective number of neighbours
  f_s       dlnS / dln a_s,  a_s^2 = sum_j A_ij Sbar_j^2 / C    size tracks static amplitude
  c_ds      E_w[d S^2] / (E_w[d] E_w[S^2]),  w ~ k   neighbour cross-moment
  eps_k, eps_n, eps_z   measured degree / neighbour-composition / coupling-selection channels
  *_th      their closed forms (Sec. 5 of the note)
  beta_ad   adiabatic slope, from v S / sqrt(D)
  beta, beta_pred = beta_ad - eps_k^th - eps_n^th - eps_z^th

Run: uv run python scripts/beta_within_degree.py   (-> data/beta_within_degree_N8000.npz, ~4 min;
     BWD_N=16000 BWD_ALPHAS=2.2,2.5 for the finite-size check)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from relative_glv import model
from relative_glv.msb import _decline_beta

MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N = int(os.environ.get("BWD_N", 8000))
SEEDS, ALPHAS = 8, tuple(float(a) for a in os.environ.get("BWD_ALPHAS", "2.2,2.5,3.5,6.0").split(","))
C = N // 40
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                   f"beta_within_degree_N{N}.npz")
KEYS = ("beta", "beta_ad", "ratio", "n_eff", "f_s", "c_ds", "inv_k",
        "eps_k", "eps_k_th", "eps_n", "eps_n_th", "eps_z", "eps_z_th", "beta_pred")


def bin_slope(S, y, P, agg=np.mean):
    bS = np.array([S[p].mean() for p in P])
    return np.polyfit(np.log(bS), np.log([agg(y[p]) for p in P]), 1)[0]


def economy(alpha, seed):
    model._ALPHA_PL = alpha                         # read at call time by _powerlaw_adjacency
    a = model.coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)
    r = model.integrate(a, tmax=250.0, n_eval=1001, lam=LAM, seed=seed,
                        method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    t = r["t"]
    S = N * r["W"][:, (t >= 180) & (t <= 240.0001)][:, ::2]          # Delta = 0.5 grid
    k = np.diff(a.indptr).astype(float)
    live = S.min(1) > 1e-6 * N
    g = np.diff(np.log(np.maximum(S, 1e-300)), axis=1)
    v = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    Sb = S.mean(1)
    beta, _, bx, _, pk = _decline_beta(Sb[live], v[live])
    if not np.isfinite(beta):
        return None
    dec = live & (Sb >= bx[pk])
    idx = np.where(live)[0]
    P = np.array_split(idx[np.argsort(Sb[live])], 20)[pk:]           # estimator's decline bins

    A = a.copy(); A.data[:] = 1.0
    az = a.copy(); az.data -= MU / (a.nnz / N)                        # random couplings
    az2 = az.copy(); az2.data **= 2
    d = np.var(np.diff(S, axis=1), axis=1)                            # Var dS_j
    D = np.var(np.diff(az @ S, axis=1), axis=1)                       # field increment variance
    Dp = az2 @ d
    s2 = Sb ** 2
    ss, Ad, Ads = A @ s2, A @ d, A @ (d * s2)
    x = -(az @ Sb) / (SIGMA / np.sqrt(C) * np.sqrt(ss))              # favourable static z-score

    rows = np.where(dec)[0]
    neff = [np.sum(c) ** 2 / np.sum(c ** 2) for c in
            (az.data[az.indptr[i]:az.indptr[i + 1]] ** 2 * d[az.indices[az.indptr[i]:az.indptr[i + 1]]]
             for i in rows)]
    lS = np.log(Sb[dec]); VS = lS.var()
    w = k / k.sum()
    c_ds = (w @ (d * s2)) / ((w @ d) * (w @ s2))
    inv_k = np.mean(1 / k[dec])
    eps_k = bin_slope(Sb, np.sqrt(k), P)
    eps_n = bin_slope(Sb, np.sqrt(Ad), P) - eps_k
    eps_z = bin_slope(Sb, np.sqrt(Dp), P) - eps_k - eps_n
    eps_k_th = np.log(k[dec]).var() / (4 * VS)
    eps_n_th = (c_ds - 1) * inv_k / (4 * VS)
    eps_z_th = bin_slope(Sb, 1 + np.minimum(c_ds / k, 0.9) * (x ** 2 - 1), P) / 2
    beta_ad = 1 - bin_slope(Sb, v * Sb / np.sqrt(Dp), P, np.median)
    return dict(beta=beta, beta_ad=beta_ad, ratio=np.median(D[dec] / Dp[dec]), n_eff=np.median(neff),
                f_s=np.polyfit(0.5 * np.log(ss[dec] / C), lS, 1)[0], c_ds=c_ds, inv_k=inv_k,
                eps_k=eps_k, eps_k_th=eps_k_th, eps_n=eps_n, eps_n_th=eps_n_th,
                eps_z=eps_z, eps_z_th=eps_z_th,
                beta_pred=beta_ad - eps_k_th - eps_n_th - eps_z_th)


if __name__ == "__main__":
    t0, table = time.time(), []
    for alpha in ALPHAS:
        res = [e for e in Parallel(n_jobs=4)(delayed(economy)(alpha, s) for s in range(SEEDS)) if e]
        table.append([np.median([e[q] for e in res]) for q in KEYS])
        print(f"alpha={alpha}: {len(res)}/{SEEDS} economies ({(time.time() - t0) / 60:.1f} min)", flush=True)
    table = np.array(table)
    np.savez(OUT, alphas=np.array(ALPHAS), keys=np.array(KEYS), table=table)
    print("\n" + " ".join(f"{q:>9}" for q in ("alpha",) + KEYS))
    for a, row in zip(ALPHAS, table):
        print(f"{a:9.1f} " + " ".join(f"{v:9.3f}" for v in row))

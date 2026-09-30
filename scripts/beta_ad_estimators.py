"""Which amplitude defines beta_ad? (thesis/standalone/beta_theory, Secs. 5-6)

The note used three denominators in beta_ad = 1 - slope of v Sbar / sqrt(D):
  Phi   D = Var dPhi_i, full fitness Phi_i = 1 - g(t) - (alpha S)_i  (eq. adiabatic; Sec. 6)
  h     D = Var dh_i, random field h_i = (z-part of alpha) S           (Table 1, beta_D)
  p     D = sum_j z_ij^2 d_j, diagonal neighbour sum                   (eq. nbrsum; Sec. 5)
Here all three are measured on the same economies, under the two integrator settings the
note's scripts used, so the definition and the numerics can be told apart. Also reported:
the degree and within-degree channels relative to D_p (Sec. 5) and the extra channel
eps_r = slope of E[sqrt(D_Phi/D_p) | S], which closes beta = beta_ad^Phi - eps_k - eps_n - eps_z - eps_r.

Run: uv run python scripts/beta_ad_estimators.py   (-> data/beta_ad_estimators.npz, ~20 min)
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from joblib import Parallel, delayed
from relative_glv import model
from relative_glv.msb import _decline_beta

MU, SIGMA, LAM, N = 1.76, 1.75, 1e-3, 8000
C = N // 40
SEEDS, ALPHAS = 8, (2.2, 2.5, 3.5, 6.0)
# (tmax, n_eval, rtol, atol, stride to Delta = 0.5): Sec. 5 script, Sec. 6 script
SETTINGS = {"coarse": (250.0, 1001, 1e-4, 1e-7, 2), "fine": (240.0, 4801, 1e-6, 1e-9, 10)}
KEYS = ("beta", "bad_Phi", "bad_h", "bad_p", "eps_k", "eps_n", "eps_z", "eps_r", "closure")
OUT = os.path.join(ROOT, "data", "beta_ad_estimators.npz")


def mad(g):
    return np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)


def bin_slope(S, y, P, agg=np.mean):
    bS = np.array([S[p].mean() for p in P])
    return np.polyfit(np.log(bS), np.log([agg(y[p]) for p in P]), 1)[0]


def economy(alpha, seed, setting):
    tmax, n_eval, rtol, atol, st = SETTINGS[setting]
    model._ALPHA_PL = alpha                         # read at call time by _powerlaw_adjacency
    a = model.coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)
    r = model.integrate(a, tmax=tmax, n_eval=n_eval, lam=LAM, seed=seed,
                        method="RK45", rtol=rtol, atol=atol)
    if not r["success"]:
        return None
    t = r["t"]
    S = N * r["W"][:, (t >= 180 - 1e-9) & (t <= 240 + 1e-9)][:, ::st]
    k = np.diff(a.indptr).astype(float)
    live = S.min(1) > 1e-6 * N
    Sb = S.mean(1)
    v = mad(np.diff(np.log(np.maximum(S, 1e-300)), axis=1))
    beta, _, bx, _, pk = _decline_beta(Sb[live], v[live])
    if not np.isfinite(beta):
        return None
    dec = live & (Sb >= bx[pk])
    idx = np.where(live)[0]
    P = np.array_split(idx[np.argsort(Sb[live])], 20)[pk:]           # estimator's decline bins

    aS = a @ S
    Phi = 1 - np.mean(S * (1 - S - aS), axis=0)[None, :] - aS       # fitness excluding self
    az = a.copy(); az.data -= MU / (a.nnz / N)                        # random couplings
    az2 = az.copy(); az2.data **= 2
    A = a.copy(); A.data[:] = 1.0
    d = np.var(np.diff(S, axis=1), axis=1)
    D = dict(Phi=np.var(np.diff(Phi, axis=1), axis=1),
             h=np.var(np.diff(az @ S, axis=1), axis=1) * C / SIGMA ** 2,
             p=az2 @ d)
    bad = {f"bad_{n}": 1 - bin_slope(Sb, v * Sb / np.sqrt(Dn), P, np.median) for n, Dn in D.items()}
    eps_k = bin_slope(Sb, np.sqrt(k), P)
    eps_n = bin_slope(Sb, np.sqrt(A @ d), P) - eps_k
    eps_z = bin_slope(Sb, np.sqrt(D["p"]), P) - eps_k - eps_n
    eps_r = bin_slope(Sb, np.sqrt(D["Phi"] / D["p"]), P)
    return dict(beta=beta, **bad, eps_k=eps_k, eps_n=eps_n, eps_z=eps_z, eps_r=eps_r,
                closure=bad["bad_Phi"] - eps_k - eps_n - eps_z - eps_r)


if __name__ == "__main__":
    t0, table = time.time(), np.full((len(SETTINGS), len(ALPHAS), len(KEYS)), np.nan)
    for i, setting in enumerate(SETTINGS):
        for j, alpha in enumerate(ALPHAS):
            res = [e for e in Parallel(n_jobs=4)(delayed(economy)(alpha, s, setting)
                                                 for s in range(SEEDS)) if e]
            table[i, j] = [np.median([e[q] for e in res]) for q in KEYS]
            print(f"{setting:6s} alpha={alpha}: {len(res)}/{SEEDS} economies "
                  f"({(time.time() - t0) / 60:.1f} min)  "
                  + " ".join(f"{q}={x:.3f}" for q, x in zip(KEYS, table[i, j])), flush=True)
    np.savez(OUT, settings=np.array(list(SETTINGS)), alphas=np.array(ALPHAS),
             keys=np.array(KEYS), table=table)

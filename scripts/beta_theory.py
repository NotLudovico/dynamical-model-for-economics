"""Numerical checks behind thesis/standalone/beta_theory: the decomposition of beta.

Per degree exponent alpha (N=8000, C=N/40, thesis operating point), per economy
(never pooled), on the decline branch of the repo estimator:
  beta      measured size-variance exponent (msb._decline_beta)
  beta_A    fixed-degree slope: -dln[v S/sqrt(k)]/dlnS + 1, i.e. slope of A(S)/S
  eps_k     dln E[sqrt k | S]/dlnS                       (sum rule: beta = beta_A - eps_k)
  b_k       exponent of k in ln v = c - a lnS + b lnk    (Prop. 1: b = 1/2)
  f_k       forward elasticity dlnS/dln sqrt(k)          (Prop. 3: f = 1)
  s         survivor degree tail exponent minus alpha    (Hill)
  V_k, V_S  Var ln k, Var ln S                           (closure: eps_k = V_k / (4 V_S))
  beta_D, eps_D, V_R  same with the realized field-increment amplitude sqrt(D_i)
                      in place of sqrt(k), and R_i^2 = D_i / k_i

Run: uv run python scripts/beta_theory.py   (-> data/beta_theory.npz, ~10 min)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from relative_glv import model
from relative_glv.msb import size_volatility, _decline_beta

MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N, SEEDS, ALPHAS = 8000, 8, (2.2, 2.5, 3.0, 3.5, 6.0)
C = N // 40
TMAX, N_EVAL, LATE, DT = 250.0, 1001, (180.0, 240.0), 0.5
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "beta_theory.npz")


def simulate(alpha, seed):
    model._ALPHA_PL = alpha                         # read at call time by _powerlaw_adjacency
    a = model.coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)
    k = np.diff(a.indptr).astype(float)
    r = model.integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                        method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    t, W = r["t"], r["W"]
    win = (t >= LATE[0]) & (t <= LATE[1])
    live = W[:, win].min(1) > 1e-6
    sv = size_volatility(W, t, window=LATE, dt=DT)
    tg = np.arange(LATE[0], LATE[1] + 1e-9, DT)
    S = np.array([np.interp(tg, t, N * w) for w in W])
    az = a.copy()
    az.data -= MU / (a.nnz / N)                     # random part of the couplings only
    h = (az @ S)[live]
    return dict(k=k[live], S=sv["Sbar"], v=sv["vol"], D=np.var(np.diff(h, axis=1), axis=1))


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def economy_stats(k, S, v, D, alpha):
    beta, _, bx, _, pk = _decline_beta(S, v)
    if not np.isfinite(beta):
        return None
    o = np.argsort(S)
    P = np.array_split(o, 20)[pk:]                  # same bins as the estimator, decline branch
    bS = np.array([S[p].mean() for p in P])
    binned = lambda f, agg: np.array([agg(f(p)) for p in P])
    A_k = binned(lambda p: v[p] * S[p] / np.sqrt(k[p]), np.median)
    A_D = binned(lambda p: v[p] * S[p] / np.sqrt(D[p]), np.median)
    dec = S >= bx[pk]
    lk, lS, lD = np.log(k[dec]), np.log(S[dec]), np.log(D[dec])
    X = np.c_[np.ones(dec.sum()), lS, lk]
    b_k = np.linalg.lstsq(X, np.log(v[dec]), rcond=None)[0][2]
    kmin = C * (alpha - 2) / (alpha - 1)
    x = k[k >= kmin]
    return dict(
        beta=beta,
        beta_A=-slope(bS, A_k / bS), eps_k=slope(bS, binned(lambda p: np.sqrt(k[p]), np.mean)),
        beta_D=-slope(bS, A_D / bS), eps_D=slope(bS, binned(lambda p: np.sqrt(D[p]), np.mean)),
        b_k=b_k, f_k=np.polyfit(lk / 2, lS, 1)[0],
        s=1 + x.size / np.log(x / kmin).sum() - alpha,
        V_k=lk.var(), V_S=lS.var(), V_R=np.var(0.5 * (lD - lk)))


if __name__ == "__main__":
    t0, rows = time.time(), {}
    for alpha in ALPHAS:
        runs = [x for x in Parallel(n_jobs=4)(delayed(simulate)(alpha, s) for s in range(SEEDS)) if x]
        st = [e for e in (economy_stats(x["k"], x["S"], x["v"], x["D"], alpha) for x in runs) if e]
        rows[alpha] = {key: np.median([e[key] for e in st]) for key in st[0]}
        rows[alpha]["n"] = len(st)
        print(f"alpha={alpha}: {len(st)}/{SEEDS} economies ({(time.time() - t0) / 60:.1f} min)", flush=True)

    keys = list(next(iter(rows.values())))
    np.savez(OUT, alphas=np.array(ALPHAS), keys=np.array(keys),
             table=np.array([[rows[a][q] for q in keys] for a in ALPHAS]))
    print("\n alpha   beta  beta_A  eps_k  bA-eps | eps_k^th  b_k   f_k    s    V_k  1/(a-1+s)^2  V_S  |"
          " beta_D eps_D  V_R")
    for a in ALPHAS:
        r = rows[a]
        th = 1 / (4 * r["V_S"] * (a - 1 + r["s"]) ** 2)
        print(f" {a:4.1f}  {r['beta']:.3f}  {r['beta_A']:.3f}  {r['eps_k']:.3f}  {r['beta_A'] - r['eps_k']:.3f} |"
              f"  {th:.3f}   {r['b_k']:.2f}  {r['f_k']:.2f}  {r['s']:.2f}  {r['V_k']:.3f}   {1 / (a - 1 + r['s']) ** 2:.3f}"
              f"    {r['V_S']:.3f} |  {r['beta_D']:.3f} {r['eps_D']:.3f} {r['V_R']:.3f}")

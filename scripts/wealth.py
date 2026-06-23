"""Stationary firm-size ('wealth') distribution of the own-degree relative GLV.

Relative size S_i = N w_i (mean 1). Pool over survivors, over the late window, and
over economies -> the stationary cross-sectional size distribution. Contrast the
RELAXED phase (sigma<sqrt2, DMFT clipped-Gaussian, light tail) with the
FLUCTUATING operating point (sigma>sqrt2, Bouchaud-Mezard condensation, heavier
tail). Stores plot-ready arrays (computed on the full pool) -> data/wealth.npz.

    uv run python scripts/wealth.py [--smoke]
"""
import os
import sys
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relative_glv.model import coupling, integrate

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SMOKE = "--smoke" in sys.argv
N, LAM, LATE = (800, 1e-3, (95.0, 118.0)) if SMOKE else (4000, 1e-3, (320.0, 390.0))
TMAX, N_EVAL = (120.0, 600) if SMOKE else (400.0, 800)
SEEDS = 2 if SMOKE else 4
ACTIVE_MIN = 0.1                       # listing cut (same as the MSB tests): drops floor "zombies"


def pool_sizes(mu, sigma):
    def one(s):
        a = coupling(N, mu, sigma, kind="powerlaw_owndeg", seed=s)
        r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=s, method="RK45", rtol=1e-4, atol=1e-7)
        if not r["success"]:
            return np.array([])
        t, W = r["t"], r["W"]
        win = (t >= LATE[0]) & (t <= LATE[1])
        S = (N * W[:, win]).ravel()
        return S[S > 1e-9].astype(np.float32)
    out = Parallel(n_jobs=6, backend="loky")(delayed(one)(s) for s in range(SEEDS))
    return np.concatenate([o for o in out if o.size])


def summarize(S, tag):
    n = S.size
    Ss = np.sort(S)[::-1]
    P = np.arange(1, n + 1) / n
    idx = np.unique(np.geomspace(1, n, 2500).astype(int)) - 1     # thin CCDF in log-rank
    ccdf_S, ccdf_P = Ss[idx], P[idx]
    pdf_y, e = np.histogram(np.log10(S), bins=90, range=(-5.0, 2.2), density=True)
    pdf_x = 0.5 * (e[:-1] + e[1:])
    act = S[S > ACTIVE_MIN]
    ay, ae = np.histogram(np.log10(act), bins=70, range=(np.log10(ACTIVE_MIN), 2.2), density=True)
    ax_ = 0.5 * (ae[:-1] + ae[1:])
    acts = np.sort(act)[::-1]                                     # active-firm CCDF (normalised to active)
    aP = np.arange(1, act.size + 1) / act.size
    aidx = np.unique(np.geomspace(1, act.size, 1500).astype(int)) - 1
    k = max(50, int(n * 0.02))                                    # Hill on the top 2%
    tail = np.sort(S)[-k:]
    mu_hill = 1.0 / np.mean(np.log(tail / tail[0]))
    return {f"{tag}_ccdf_S": ccdf_S, f"{tag}_ccdf_P": ccdf_P,
            f"{tag}_pdf_x": pdf_x, f"{tag}_pdf_y": pdf_y,
            f"{tag}_act_x": ax_, f"{tag}_act_y": ay,
            f"{tag}_act_ccdf_S": acts[aidx], f"{tag}_act_ccdf_P": aP[aidx],
            f"{tag}_mu_hill": float(mu_hill), f"{tag}_xmin": float(tail[0]),
            f"{tag}_max": float(S.max()), f"{tag}_pgt1": float((S > 1).mean()),
            f"{tag}_n": int(n)}


if __name__ == "__main__":
    out = dict(N=N, lam=LAM, active_min=ACTIVE_MIN, hill_frac=0.02, op_mu=1.76, op_sig=1.75)
    for tag, (mu, sigma) in (("op", (1.76, 1.75)),):
        S = pool_sizes(mu, sigma)
        out.update(summarize(S, tag))
        print(f"{tag:>8}: n={S.size}  mean={S.mean():.3f}  max={S.max():.1f}x  "
              f"P(S>1)={(S>1).mean():.3f}  Hill mu={out[f'{tag}_mu_hill']:.2f} "
              f"(PDF exp {1+out[f'{tag}_mu_hill']:.2f})", flush=True)
    np.savez(os.path.join(DATA, "wealth.npz"), **out)
    print(f"saved {os.path.join(DATA, 'wealth.npz')}  (empirical firm-size Zipf: alpha~2 i.e. mu~1)")

"""Unique fixed point, multiple attractors or chaos?  The gap between the HDMFT stability line
sigma_c(mu) and the simulated freezing line of the phase grid (N=4000, C=100, power law 2.5).

For each (mu, sigma): one graph + couplings, three initial conditions (narrow uniform, two broad
log-normal).  Per run: late drift  max_i |ln S_i(T) - ln S_i(T-40)|  (frozen if small) and the
late growth-rate std (the churn of the phase grid).  Between frozen runs: mean |ln S_a - ln S_b|.
UFP: all frozen, same state.  MA: all frozen, different states.  Chaos: not frozen.

Run: uv run python scripts/phase_attractors.py   -> data/phase_attractors.npz
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
from relative_glv.model import coupling

N, C, LAM, TMAX, SMIN = 4000, 100, 1e-3, 400.0, 1e-2
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "phase_attractors.npz")
# mu in code convention (competition > 0); HDMFT sigma_c (lam=1e-3) and simulated freezing line
CUTS = {2.50: (1.620, 1.747), 1.97: (1.509, 1.632), 1.45: (1.346, 1.458), 1.05: (1.205, 1.400), 0.66: (1.085, 1.371)}


def w0_of(kind, seed):
    rng = np.random.default_rng(1000 + seed)
    w = rng.uniform(0.5, 1.5, N) if kind == 0 else rng.lognormal(0.0, 2.0, N)
    return w / w.sum()


def run(a, w0):
    def rhs(t, s):
        w = np.clip(s, 0.0, None); w = w / w.sum()
        f = 1.0 - N * w - N * (a @ w)
        return w * (f - w @ f) + LAM * (1.0 / N - w)
    t_eval = np.concatenate([np.arange(320.0, TMAX - 39.9, 0.5), [TMAX]])
    t_eval = np.unique(np.concatenate([t_eval, np.arange(360.0, TMAX + 1e-9, 0.5)]))
    r = solve_ivp(rhs, (0.0, TMAX), w0, method="RK45", t_eval=t_eval, rtol=1e-4, atol=1e-7)
    if not r.success or not np.isfinite(r.y).all():
        return None
    # firms at the floor sit at w ~ lam/N ~ atol, so their logs are solver noise: clip at S = 1e-2
    lnS = np.log(np.clip(r.y * N, SMIN, None))
    k = np.searchsorted(r.t, TMAX - 40)
    up = (lnS[:, -1] > np.log(SMIN)) | (lnS[:, k] > np.log(SMIN))
    drift = np.abs(lnS[up, -1] - lnS[up, k]).max()
    live = (lnS > np.log(SMIN)).all(1)
    churn = np.diff(lnS[live], axis=1).std() if live.any() else np.nan
    return lnS[:, -1], drift, churn


def point(mu, sigma, seed=0):
    a = coupling(N, mu, sigma, kind="powerlaw", seed=seed, mean_degree=C)
    R = [run(a, w0_of(k, seed + k)) for k in range(3)]
    if any(r is None for r in R):
        return dict(mu=mu, sigma=sigma, ok=False)
    fin = [r[0] for r in R]
    lo = np.log(SMIN)
    dist = [np.abs(fin[i] - fin[j])[(fin[i] > lo) | (fin[j] > lo)].mean() for i, j in ((0, 1), (0, 2), (1, 2))]
    return dict(mu=mu, sigma=sigma, ok=True, drift=[r[1] for r in R], churn=[r[2] for r in R], dist=dist,
                final=np.array(fin, np.float32))


if __name__ == "__main__":
    jobs = []
    for mu, (sc, sf) in CUTS.items():
        for s in (sc - 0.10, sc + 0.03, 0.5 * (sc + sf), sf, sf + 0.10):
            jobs.append((mu, round(s, 3)))
    R = Parallel(n_jobs=10, verbose=5)(delayed(point)(mu, s) for mu, s in jobs)
    np.savez(OUT, results=np.array(R, dtype=object), cuts=np.array(list(CUTS.items()), dtype=object))
    print(f"{'mu':>6} {'sigma':>6} | {'sc_HD':>6} {'s_freeze':>8} | {'max drift':>9} {'churn':>7} | {'state dist':>10}  verdict")
    for r in R:
        sc, sf = CUTS[r["mu"]]
        if not r["ok"]:
            print(f"{-r['mu']:6.2f} {r['sigma']:6.3f} | {sc:6.3f} {sf:8.3f} | integration failed"); continue
        frozen = max(r["drift"]) < 1e-2
        same = max(r["dist"]) < 1e-2
        verdict = ("UFP" if same else "MA") if frozen else "chaos/slow"
        print(f"{-r['mu']:6.2f} {r['sigma']:6.3f} | {sc:6.3f} {sf:8.3f} | {max(r['drift']):9.2e} {max(r['churn']):7.4f} | "
              f"{max(r['dist']):10.2e}  {verdict}")

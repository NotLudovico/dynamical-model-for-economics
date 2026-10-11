"""Mean-field reduction of the partner model (variants3.py), for predictions before full runs.

Fast-shock limit (tau << year): each firm is a log random walk
    d ln x_i = nu dt + sqrt(2 D H_i) dW_i,     H_i = sum_{j in P_i} x_j^2 / (sum_{j in P_i} x_j)^2,
with exit at rate delta and entry at s0 m; nu is fixed by keeping m constant (the normalisation).
Mean-field step: the n_i = ceil((x_i/(s0 m))^gamma) partners are random members of the current
population.  H_i is held between partner events and redrawn from the current population when the
firm's partner count changes by more than 25% or at rate delta (turnover of the largest partner).
Cost O(N) per step instead of O(sum n_i).  Approximation: the partners' sizes are frozen between
redraws (in the full model they drift).

Run:  python scripts/screen/theory_partner.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from score import score, comove


def reduced(*, gamma=0.75, D=0.05, N=4000, delta=0.05, s0=0.05, T=1000.0, burn=500.0, dt=0.05,
            seed=0, redraw=1.25, alpha_k=None, exit="hazard", x_exit=1e-3):
    """alpha_k: if set, each firm has a breadth k_i ~ Pareto(alpha_k), k >= 1, drawn at birth, and
    v_i^2 = 2 D H_i k_i / <k>  (hybrid: share-weighted field times sqrt of the number of channels).
    exit: "hazard" (random exit at rate delta, Reed) or "decline" (a firm exits when x < x_exit m and a
    newborn of size s0 m takes its node; delta unused, the exit rate is an output)."""
    rng = np.random.default_rng(seed + 4000)
    kap = lambda n: (1 - rng.random(n)) ** (-1 / alpha_k) if alpha_k else np.ones(n)
    kbar = (alpha_k - 1) / (alpha_k - 2) if alpha_k else 1.0
    k = kap(N)
    x = s0 / (1 - rng.random(N)); x /= x.mean()
    fid, nxt = np.arange(N), N
    n_at = np.ones(N); H = np.ones(N)
    X, Fid, LM, lm = [], [], [], 0.0
    nstep, nburn, every = int(T / dt), int(burn / dt), int(round(1 / dt))

    def draw(idx):
        n = np.minimum(N - 1, np.ceil(np.maximum(x[idx] / s0, 1.0) ** gamma)).astype(int)
        for k, i in enumerate(idx):
            p = x[rng.integers(N, size=n[k])]
            H[i] = (p ** 2).sum() / p.sum() ** 2
        n_at[idx] = n

    draw(np.arange(N))
    for s in range(nstep):
        n = np.minimum(N - 1, np.ceil(np.maximum(x / s0, 1.0) ** gamma))
        stale = (n > redraw * n_at) | (n < n_at / redraw) | (rng.random(N) < delta * dt)
        if stale.any():
            draw(np.flatnonzero(stale))
        x = x * np.exp(np.sqrt(2 * D * H * k / kbar * dt) * rng.standard_normal(N))
        dead = (rng.random(N) < delta * dt) if exit == "hazard" else (x < x_exit)
        if dead.any():
            nd = int(dead.sum())
            x[dead] = s0; fid[dead] = np.arange(nxt, nxt + nd); nxt += nd; H[dead] = 1.0; n_at[dead] = 1
            k[dead] = kap(nd)
        mx = x.mean(); x /= mx; lm += np.log(mx)
        if s >= nburn and (s - nburn) % every == 0:
            X.append(x.astype(np.float32)); Fid.append(fid.copy()); LM.append(lm)
    return np.array(X), np.array(Fid, dtype=np.int32), np.array(LM)


def report(tag, X, fid, lm):
    s = score(X, fid); c = comove(X, fid, lm)
    print(f"{tag:28} top {np.mean(X.max(1) / X.sum(1)):.3f}  hill1 {s['hill1']:.2f} "
          f"(halves {s['hill1_halves'][0]:.2f} {s['hill1_halves'][1]:.2f})  beta {s.get('beta', np.nan):6.3f}  "
          f"ratios {np.round(s.get('ratios', [np.nan] * 3), 2)}  loc.beta {np.round(s.get('local_beta', []), 2)}  "
          f"b {s.get('tail_b', np.nan):.1f}  agg vol {c['agg_vol']:.3f}", flush=True)
    return s


if __name__ == "__main__":
    from joblib import Parallel, delayed
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":        # same settings as the full-model smoke test (N=1000, T=300, burn 100)
        jobs = [(g, 0.05, dict(N=1000, T=300.0, burn=100.0)) for g in (0.0, 1.0)]
    elif mode == "hybrid":        # breadth heterogeneity, alpha_k = 3.3 from b = 2(alpha - 1) = 4.6
        jobs = [(g, s0, dict(alpha_k=3.3)) for s0 in (0.1, 0.2) for g in (0.0, 1.0, 1.5, 2.0)]
    else:
        jobs = [(g, s0, {}) for s0 in (0.05, 0.2) for g in (0.0, 0.5, 1.0, 1.5, 2.0)]
    res = Parallel(n_jobs=12)(delayed(reduced)(gamma=g, s0=s0, **kw) for g, s0, kw in jobs)
    for (g, s0, kw), r in zip(jobs, res):
        report(f"reduced g{g} s0 {s0}" + (f" ak{kw['alpha_k']}" if kw.get("alpha_k") else ""), *r)

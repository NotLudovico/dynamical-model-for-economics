"""Round 2 of the model screen: couplings that change in time.  All in absolute sizes x_i, m = <x>.

Round 1 showed that with quenched alpha_ij every brake-free variant condenses onto the best niche.
Here each coupling is redrawn at rate 1/tau (correlation e^{-|t-t'|/tau}); tau = inf is quenched.
Every variant except `ann` and `rebirth_base` has no brake and uses random exit at rate delta
(a newborn of size s0 m takes the node), the Reed mechanism for a Zipf tail.

Same power-law graph as round 1 (model.coupling); the graph fixes who can interact with whom.
z_ij(t) ~ N(0,1) is refreshed at rate 1/tau.  Two normalisations of a coupling:
  repo     alpha_ij = mu/C + (sigma/sqrt C) z_ij        (as in the repo; field sums over neighbours)
  avg      a_ij     = mu + sigma z_ij                   (field averages over the partners)

  ann           x_i' = x_i [1 - x_i/m - sum_j alpha_ij(t) x_j/m] + lam (m - x_i)
  ann_nb        x_i' = x_i [1 - sum_j alpha_ij(t) x_j/m] + lam (m - x_i)
  ann_minflow   x_i' = x_i - sum_j alpha_ij(t) min(x_i, x_j) + lam (m - x_i)
  rebirth_nb    ann_nb with tau = inf, but a newborn draws fresh couplings (its row and column)
  rebirth_base  the same with the brake x_i/m
  share         x_i' = x_i [1 - sum_j a_ij(t) x_j / sum_j x_j] + lam (m - x_i)    (j over all neighbours)
  scope_share   share with only n_i(x_i) partners, n_i = min(k_i, ceil((x_i/(s0 m))^gs)),
                partners drawn among the k_i neighbours and redrawn with the coupling
  scope         x_i' = x_i [1 - (1/n_i) sum_{j in P_i} a_ij(t) x_j/m] + lam (m - x_i)

share/scope_share: the field is a size-weighted mean of the partners' couplings, so its variance
is sigma^2 times the Herfindahl index of the partners' sizes (granular volatility; ratios < q when
partner sizes are Zipf).  scope: the linear GLV field averaged over n_i partners.
"""
import numpy as np
from relative_glv import model

VARIANTS = ("ann", "ann_nb", "ann_minflow", "rebirth_nb", "rebirth_base", "share", "scope_share", "scope")


def simulate(variant, *, alpha=6.0, sigma=1.75, mu=1.76, N=4000, lam=1e-4, delta=0.05, s0=2e-3,
             tau=1.0, gs=0.5, T=500.0, burn=150.0, dt=0.01, rec=1.0, seed=0):
    """Same contract as variants.simulate (plus `clipped`, the fraction of firm-steps with |r| > 50)."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    model._ALPHA_PL = alpha
    A = model.coupling(N, mu, sigma, kind="powerlaw", seed=seed, mean_degree=N // 40).tocsr()
    ip, nb = A.indptr, A.indices
    deg = np.diff(ip); C = deg.mean(); E = nb.size
    row = np.repeat(np.arange(N), deg)
    rank = np.arange(E) - ip[row]                       # position of a slot within its row
    p = nb.copy()                                       # current partner of each slot
    rng = np.random.default_rng(seed + 2000)
    z = rng.standard_normal(E)
    avg = variant in ("share", "scope_share", "scope")
    c0, c1 = (mu, sigma) if avg else (mu / C, sigma / np.sqrt(C))
    brake = variant in ("ann", "rebirth_base")
    if variant.startswith("rebirth"):
        tau = np.inf
    x = rng.uniform(0.5, 1.5, N); x /= x.mean()
    fid, nxt = np.arange(N), N
    every, nstep, nburn = int(round(rec / dt)), int(round(T / dt)), int(round(burn / dt))
    X, Fid, tt, clipped = [], [], [], 0
    for s in range(nstep):
        if np.isfinite(tau):                            # redraw a Poisson number of couplings
            idx = rng.integers(E, size=rng.binomial(E, dt / tau))
            z[idx] = rng.standard_normal(idx.size)
            if variant.startswith("scope"):
                r_ = row[idx]
                p[idx] = nb[ip[r_] + rng.integers(deg[r_])]
        w = c0 + c1 * z
        xp = x[p]
        if variant == "ann_minflow":
            r = 1.0 - np.bincount(row, w * np.minimum(x[row], xp), N) / x
        elif variant in ("share", "scope_share", "scope"):
            if variant == "share":
                act = 1.0
            else:
                n = np.minimum(deg, np.ceil(np.maximum(x / s0, 1.0) ** gs))
                act = rank < n[row]
            num = np.bincount(row, act * w * xp, N)
            r = 1.0 - (num / n if variant == "scope" else num / np.bincount(row, act * xp, N))
        else:
            r = 1.0 - np.bincount(row, w * xp, N) - (x if brake else 0.0)
        clipped += int((np.abs(r) > 50.0).sum())
        x = x * np.exp(np.clip(r, -50.0, 50.0) * dt)    # guard only; the clipped count is returned
        x = x + lam * (1.0 - x) * dt
        if delta > 0:
            dead = rng.random(N) < delta * dt
            if dead.any():
                x[dead] = s0; fid[dead] = np.arange(nxt, nxt + dead.sum()); nxt += dead.sum()
                if variant.startswith("rebirth"):
                    m_ = dead[row] | dead[p]
                    z[m_] = rng.standard_normal(m_.sum())
        mx = x.mean()
        if not np.isfinite(mx):
            return None
        x /= mx
        if s >= nburn and (s - nburn) % every == 0:
            X.append(x.astype(np.float32)); Fid.append(fid.copy()); tt.append(s * dt)
    return dict(X=np.array(X), fid=np.array(Fid, dtype=np.int32), deg=deg, t=np.array(tt),
                births=(nxt - N) / N / T, clipped=clipped / (N * nstep))

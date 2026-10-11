"""Round 3: the partner model.  Absolute sizes x_i, m = <x>.

  x_i' = x_i [1 - sum_{j in P_i} (mu + sigma z_j(t)) x_j / X_i],   X_i = sum_{j in P_i} x_j

  - P_i: firm i's n_i = min(N-1, ceil((x_i / (s0 m))^gamma)) partners;
  - z_j(t): partner j's shock, Ornstein-Uhlenbeck with unit variance and correlation time tau;
  - exit at rate delta, newborn of size s0 m with a fresh partner order.

Mass balance: zeta - 1 = s0 delta / (delta + v^2/2) for the size tail, with v the volatility.

mu is common to every firm (weights sum to one) and cancels from x/m; it is set to 0.  No floor.
In the fast limit (tau << year), d ln x_i = a dt + sqrt(2 D H_i) dW_i with D = sigma^2 tau and H_i
the Herfindahl index of i's partners' sizes.

Partners are sticky.  Each node draws a random order of the other nodes at birth, and its firm's
partners are the first n_i nodes in that order; a firm adds partners as it grows and drops them if
it shrinks.  When a partner exits, the newborn at that node takes the link, so a link is a
relationship with a market position that outlives its current occupant.  In the stationary state
a partner's size is a draw from the population.
"""
import numpy as np


def simulate(*, gamma=0.75, D=0.05, tau=0.1, N=4000, delta=0.05, s0=0.05, T=1000.0, burn=500.0,
             dt=0.01, rec=1.0, seed=0, **_):
    """dict(X, fid, lm, t, births, npart): X[r, i] = x_i/m, lm = ln m(t) up to a constant (aggregate
    growth), npart = mean number of partners per firm."""
    rng = np.random.default_rng(seed + 3000)
    sigma = np.sqrt(D / tau)
    K = N - 1
    order = np.argsort(rng.random((N, K)), axis=1).astype(np.int16)      # random order of the others
    order += (order >= np.arange(N, dtype=np.int16)[:, None])            # skip self
    flat = order.ravel()
    x = s0 / (1 - rng.random(N)); x /= x.mean()           # Zipf start: the founders-only start relaxes
    z = rng.standard_normal(N)                             # on a time ~ 1/(delta s0) >> T
    a_, b_ = np.exp(-dt / tau), np.sqrt(1 - np.exp(-2 * dt / tau))
    fid, nxt, lm = np.arange(N), N, 0.0
    every, nstep, nburn = int(round(rec / dt)), int(round(T / dt)), int(round(burn / dt))
    X, Fid, LM, tt, NP = [], [], [], [], []
    ar = np.arange(N)
    for s in range(nstep):
        n = np.minimum(K, np.ceil(np.maximum(x / s0, 1.0) ** gamma)).astype(np.int64)
        rows = np.repeat(ar, n)
        start = np.cumsum(n) - n
        cols = flat[rows * K + (np.arange(rows.size) - start[rows])]
        xp = x[cols]
        phi = sigma * np.bincount(rows, z[cols] * xp, N) / np.bincount(rows, xp, N)
        x = x * np.exp(-phi * dt)
        z = a_ * z + b_ * rng.standard_normal(N)
        dead = rng.random(N) < delta * dt
        if dead.any():
            k = int(dead.sum())
            x[dead] = s0; fid[dead] = np.arange(nxt, nxt + k); nxt += k
            o = np.argsort(rng.random((k, K)), axis=1).astype(np.int16)
            o += (o >= np.flatnonzero(dead).astype(np.int16)[:, None])
            order[dead] = o
        mx = x.mean()
        if not np.isfinite(mx):
            return None
        x /= mx; lm += np.log(mx)
        if s >= nburn and (s - nburn) % every == 0:
            X.append(x.astype(np.float32)); Fid.append(fid.copy()); LM.append(lm); tt.append(s * dt)
            NP.append(n.mean())
    return dict(X=np.array(X), fid=np.array(Fid, dtype=np.int32), lm=np.array(LM), t=np.array(tt),
                births=(nxt - N) / N / T, npart=float(np.mean(NP)), deg=np.zeros(N, int))

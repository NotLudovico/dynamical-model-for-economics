"""Round 5: a second variable per firm, the carrying capacity K_i.  Absolute sizes x_i, m = <x>.

  lvK      x_i' = x_i [1 - (x_i + Phi_i) / K_i] + lam (m - x_i)     (standard LV competition)
  lvK_rel  x_i' = x_i [1 - x_i / K_i - Phi_i / m] + lam (m - x_i)   (control: field relative to m)
  lvK_sat  x_i' = x_i [1 - x_i / K_i - sum_j alpha_ij x_j / (K_i + K_j)] + lam (m - x_i)
           (saturating pressure: a much bigger competitor can take at most i's own market; with
           all K equal this is the thesis model with alpha/2)

  Phi_i = sum_j alpha_ij x_j (absolute), alpha from model.coupling on the power-law graph.
  d ln K_i = s dW_i - s^2/2 dt (level martingale); exit at rate delta resets x = K = K0 <K>.
  The thesis model is lvK with every K_i = m.

Both right-hand sides are homogeneous of degree zero in a common rescaling of (x, K), so the
integrator divides x and K by <K> after every step: survivors drift up at the exit rate (Reed), and
the capacity tail is Zipf, zeta = 1/(1 - phi) with phi ~ K0.  A firm is active while K_i exceeds
the competition Phi_i it faces; it then sits near x = K - Phi (lvK) and falls back to the floor
lam when K drifts down or the field rises, so lives keep the rise-plateau-fall shape.
"""
import numpy as np
from relative_glv import model

VARIANTS = ("lvK", "lvK_rel", "lvK_sat")


def simulate(variant, *, alpha=6.0, sigma=1.75, mu=1.76, N=4000, lam=1e-4, delta=0.05, K0=0.05,
             s=0.3, T=500.0, burn=200.0, dt=0.01, rec=1.0, seed=0, **_):
    """dict(X, fid, deg, t, births, K): X[r, i] = x_i/m, K[r, i] = K_i/m at each record."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    model._ALPHA_PL = alpha
    A = model.coupling(N, mu, sigma, kind="powerlaw", seed=seed, mean_degree=N // 40).tocsr()
    Ao = A.tocoo(); Ar, Ac, Ad = Ao.row, Ao.col, Ao.data
    rng = np.random.default_rng(seed + 5000)
    K = K0 / (1 - rng.random(N)); K /= K.mean()          # Zipf start for the capacities
    x = 0.5 * K
    fid, nxt = np.arange(N), N
    every, nstep, nburn = int(round(rec / dt)), int(round(T / dt)), int(round(burn / dt))
    X, Fid, tt, Ks = [], [], [], []
    sq = s * np.sqrt(dt)
    for st in range(nstep + nburn):
        if variant == "lvK_sat":
            r = 1.0 - x / K - np.bincount(Ar, Ad * x[Ac] / (K[Ar] + K[Ac]), N)
        elif variant == "lvK":
            phi = A @ x
            r = 1.0 - (x + phi) / K
        else:
            r = 1.0 - x / K - (A @ x) / x.mean()
        x = x * np.exp(np.clip(r, -50, 50) * dt)
        x = x + lam * (x.mean() - x) * dt
        K = K * np.exp(sq * rng.standard_normal(N) - 0.5 * sq * sq)
        dead = rng.random(N) < delta * dt
        if dead.any():
            K[dead] = K0 * K.mean(); x[dead] = K0 * K.mean() * 0.5
            fid[dead] = np.arange(nxt, nxt + dead.sum()); nxt += dead.sum()
        mk = K.mean()
        x /= mk; K /= mk
        if not np.all(np.isfinite(x)):
            return None
        if st >= nburn and (st - nburn) % every == 0:
            m = x.mean()
            X.append((x / m).astype(np.float32)); Ks.append((K / m).astype(np.float32))
            Fid.append(fid.copy()); tt.append(st * dt)
    return dict(X=np.array(X), fid=np.array(Fid, dtype=np.int32), deg=np.diff(A.indptr), t=np.array(tt),
                births=(nxt - N) / N / (T + burn), K=np.array(Ks))

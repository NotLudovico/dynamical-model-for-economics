"""Variants of the relative GLV for the model screen, all in absolute sizes x_i with m = <x>.

Notation: Phi_i = sum_j alpha_ij x_j / m (interaction field), alpha from model.coupling on the
power-law graph (per-edge mean mu/C, std sigma/sqrt C).  lam (m - x_i) is the immigration floor of
the repo model.  Every right-hand side is homogeneous of degree one in x, so dividing all x by a
common factor changes nothing; the integrator divides by m after every step and records x/m.

  base          x_i' = x_i [1 - x_i/m - Phi_i] + lam (m - x_i)                 (repo model)
  nobrake       x_i' = x_i [1 - Phi_i] + lam (m - x_i)
  capacity      x_i' = x_i [1 - x_i/K_i - Phi_i] + lam (m - x_i),  K_i' = eps (x_i - K_i)
  sublin        x_i' = x_i [1 - <Phi> - (x_i/m)^(a-1) (Phi_i - <Phi>)] + lam (m - x_i)
  minflow       x_i' = x_i [1 - x_i/m] - sum_j alpha_ij min(x_i, x_j) + lam (m - x_i)
  minflow_nb    x_i' = x_i - sum_j alpha_ij min(x_i, x_j) + lam (m - x_i)
  inertia       x_i' = x_i g_i,  g_i' = (m / (tau0 x_i)) (1 - Phi_i - g_i),  + lam (m - x_i)
  demog         dx_i = x_i [1 - Phi_i] dt + sqrt(D m x_i) dW_i, no floor
  base_demog    dx_i = x_i [1 - x_i/m - Phi_i] dt + sqrt(D m x_i) dW_i, no floor   (repo demographic)

demog variants: a firm reaching x_i <= 0 dies and a newborn of size D m takes its node.
Turnover option delta > 0 (any variant): each firm exits at hazard delta and a newborn of size s0 m
takes its node.  A newborn keeps the node's couplings (the node is the market niche).
"""
import numpy as np
from relative_glv import model

VARIANTS = ("base", "nobrake", "capacity", "sublin", "minflow", "minflow_nb", "inertia", "demog",
            "base_demog")


def simulate(variant, *, alpha=6.0, sigma=1.75, mu=1.76, N=4000, lam=1e-4, delta=0.0, s0=2e-3,
             eps=1.0, a=0.8, tau0=1.0, D=1e-2, T=500.0, burn=150.0, dt=0.01, rec=1.0, seed=0):
    """Euler integration (multiplicative part exact in ln x).  Returns dict(X, fid, deg, t):
    X[r, i] = x_i/m at record r (every `rec` after `burn`), fid the firm id at node i (changes at
    each death/birth)."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    model._ALPHA_PL = alpha                       # module global, set in this process
    A = model.coupling(N, mu, sigma, kind="powerlaw", seed=seed, mean_degree=N // 40).tocsr()
    Acoo = A.tocoo()
    rng = np.random.default_rng(seed + 1000)
    x = rng.uniform(0.5, 1.5, N); x /= x.mean()
    K, g = x.copy(), np.zeros(N)
    fid, nxt = np.arange(N), N
    demog = variant in ("demog", "base_demog")
    every, nstep, nburn = int(round(rec / dt)), int(round(T / dt)), int(round(burn / dt))
    X, Fid, tt = [], [], []
    for s in range(nstep):
        if variant in ("minflow", "minflow_nb"):
            fl = np.bincount(Acoo.row, Acoo.data * np.minimum(x[Acoo.row], x[Acoo.col]), N) / x
            r = 1.0 - fl - (x if variant == "minflow" else 0.0)
        else:
            phi = A @ x
            if variant in ("base", "base_demog"):
                r = 1.0 - x - phi
            elif variant in ("nobrake", "demog"):
                r = 1.0 - phi
            elif variant == "capacity":
                r = 1.0 - x / np.maximum(K, 1e-300) - phi
            elif variant == "sublin":
                pm = phi.mean()
                r = 1.0 - pm - x ** (a - 1.0) * (phi - pm)
            elif variant == "inertia":
                F = 1.0 - phi
                g = F + (g - F) * np.exp(-dt / (tau0 * x))
                r = g
        x = x * np.exp(r * dt)
        if variant == "capacity":
            K = x + (K - x) * np.exp(-eps * dt)
        if demog:
            x = x + np.sqrt(D * np.maximum(x, 0.0) * dt) * rng.standard_normal(N)
            dead = x <= 0
            if dead.any():
                x[dead] = D; fid[dead] = np.arange(nxt, nxt + dead.sum()); nxt += dead.sum()
                g[dead] = 0.0; K[dead] = D
        else:
            x = x + lam * (1.0 - x) * dt          # m = 1 here
        if delta > 0:
            dead = rng.random(N) < delta * dt
            if dead.any():
                x[dead] = s0; fid[dead] = np.arange(nxt, nxt + dead.sum()); nxt += dead.sum()
                g[dead] = 0.0; K[dead] = s0
        mx = x.mean(); x /= mx; K /= mx
        if not np.isfinite(mx):
            return None
        if s >= nburn and (s - nburn) % every == 0:
            X.append(x.astype(np.float32)); Fid.append(fid.copy()); tt.append(s * dt)
    return dict(X=np.array(X), fid=np.array(Fid, dtype=np.int32), deg=np.diff(A.indptr), t=np.array(tt),
                births=(nxt - N) / N / T)

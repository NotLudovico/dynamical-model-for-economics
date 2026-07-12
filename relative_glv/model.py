"""The relative (scale-invariant) generalised Lotka-Volterra model.

    x_i' = x_i [ 1 - x_i/m - (alpha x)_i / m ],   m = <x> = M/N.

Interactions act on the MEAN firm m, so the competition term is O(M) not O(M^2):
the aggregate M grows exponentially in physical time with no finite-time blow-up.
We integrate in a share + log-scale split (shares w_i = x_i/M on the simplex,
ln M separate), which is numerically unconditionally stable -- see `integrate`.

Sign convention: mu > 0 is mean COMPETITION (it lowers the growth rate).
"""
import os
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.integrate import solve_ivp

# power-law degree exponent (env-overridable so a whole figure run can be pinned to a
# different tail without editing callers), target mean degree
_ALPHA_PL, _MEAN_DEGREE = float(os.environ.get("RGLV_ALPHA_PL", 2.5)), 100


def coupling(N, mu, sigma, *, kind="fc", gamma=0.0, seed=0, mean_degree=None):
    """Interaction matrix alpha for the relative GLV.

    Couplings are scaled so the mean field stays finite as connectivity grows:
    per-interaction mean mu/C and std sigma/sqrt(C), with C the connectivity
    (C = N fully connected, C = mean degree on the graph). gamma sets the
    symmetric/antisymmetric correlation of the reciprocal pair via
    alpha ~ sqrt(1+gamma) S +/- sqrt(1-gamma) V, S=(M+M^T)/sqrt2, V=(M-M^T)/sqrt2.

    kind="fc":       fully-connected Gaussian, alpha_ij = mu/N + (sigma/sqrt N) z_ij,
                     zero diagonal. The clean ensemble the DMFT is derived for.
    kind="powerlaw": power-law configuration-model graph (exponent 2.5, mean
                     degree ~100), Roy disorder per edge, couplings normalised by
                     the GLOBAL mean degree C_eff. The realistic ensemble.
    kind="powerlaw_owndeg": same graph, but each row normalised by its OWN degree
                     k_i (a_ij = mu/k_i + (sigma/sqrt(k_i)) z_ij), so every firm's
                     disorder field has the same variance regardless of degree --
                     the principled per-fan-in (mean-field 1/sqrt(connectivity))
                     scaling on a heterogeneous graph. Unlike mean-degree, this
                     keeps the size-volatility exponent beta sustained as N->inf
                     (mean-degree over-couples the hubs and washes beta to 0).
                     gamma must be 0 (couplings are independent/asymmetric).
    kind="powerlaw_rowavg": EXPLORATORY variant. Same graph, but each interaction is the
                     AVERAGE of O(1) couplings over the k_i neighbours,
                     a_ij = (mu + sigma z_ij)/k_i. Mean field mu (as owndeg) but the noise
                     self-averages as sigma/sqrt(k_i): the field variance falls as 1/k_i,
                     so hubs are QUIETER and degree DRIVES the size-volatility law
                     directly. gamma must be 0.

    mean_degree overrides the target mean degree (default ~100) for the powerlaw* kinds:
    lower C => sparser graph => each firm's field is a sum of fewer neighbours => fatter,
    less-Gaussian fluctuations (own-degree departs from the fully-connected limit as C falls).

    Returns a dense ndarray (fc) or a scipy.sparse csr_array (powerlaw*); both
    support `alpha @ w`.
    """
    rng = np.random.default_rng(seed)
    if kind == "fc":
        z = rng.standard_normal((N, N))
        a = mu / N + (sigma / np.sqrt(N)) * z
        np.fill_diagonal(a, 0.0)
        return a
    if kind == "powerlaw":
        C = min(mean_degree or _MEAN_DEGREE, N - 1)
        kmin = C * (_ALPHA_PL - 2) / (_ALPHA_PL - 1)
        deg = np.maximum(
            (kmin * (1 - rng.uniform(size=N)) ** (-1 / (_ALPHA_PL - 1))).round().astype(int), 1)
        if deg.sum() % 2:
            deg[deg.argmin()] += 1
        G = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(seed)))
        G.remove_edges_from(nx.selfloop_edges(G))
        A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
        A.data[:] = 1.0
        C_eff = float(np.asarray(A.sum(axis=1)).mean())
        Au = sparse.triu(A, k=1).tocoo()
        ei, ej, nE = Au.row, Au.col, Au.row.size
        a, b = rng.normal(0, 1, nE), rng.normal(0, 1, nE)
        sym, anti = (a + b) / np.sqrt(2), (a - b) / np.sqrt(2)
        scale = sigma / np.sqrt(2 * C_eff)
        w_ij = mu / C_eff + scale * (np.sqrt(1 + gamma) * sym + np.sqrt(1 - gamma) * anti)
        w_ji = mu / C_eff + scale * (np.sqrt(1 + gamma) * sym - np.sqrt(1 - gamma) * anti)
        rows = np.concatenate([ei, ej])
        cols = np.concatenate([ej, ei])
        return sparse.csr_array((np.concatenate([w_ij, w_ji]), (rows, cols)), shape=(N, N))
    if kind == "powerlaw_owndeg":
        if gamma != 0.0:
            raise ValueError("powerlaw_owndeg supports only gamma=0 (asymmetric couplings)")
        C = min(mean_degree or _MEAN_DEGREE, N - 1)
        kmin = C * (_ALPHA_PL - 2) / (_ALPHA_PL - 1)
        deg = np.maximum(
            (kmin * (1 - rng.uniform(size=N)) ** (-1 / (_ALPHA_PL - 1))).round().astype(int), 1)
        if deg.sum() % 2:
            deg[deg.argmin()] += 1
        G = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(seed)))
        G.remove_edges_from(nx.selfloop_edges(G))
        A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
        ki = np.diff(A.indptr).astype(float)          # each node's own degree (fan-in)
        ki[ki == 0] = 1.0
        coo = A.tocoo()
        z = rng.normal(size=coo.row.size)
        vals = mu / ki[coo.row] + (sigma / np.sqrt(ki[coo.row])) * z
        return sparse.csr_array((vals, (coo.row, coo.col)), shape=(N, N))
    if kind == "powerlaw_rowavg":
        # Exploratory variant: interaction = AVERAGE of O(1) couplings over the k_i
        # neighbours, a_ij = (mu + sigma z)/k_i. Same graph build as powerlaw_owndeg;
        # differs ONLY in the noise prefactor (sigma/k_i, not sigma/sqrt(k_i)), so the
        # field noise self-averages as ~k^-1/2 and hubs are quieter (degree-driven beta).
        if gamma != 0.0:
            raise ValueError("powerlaw_rowavg supports only gamma=0 (asymmetric couplings)")
        C = min(mean_degree or _MEAN_DEGREE, N - 1)
        kmin = C * (_ALPHA_PL - 2) / (_ALPHA_PL - 1)
        deg = np.maximum(
            (kmin * (1 - rng.uniform(size=N)) ** (-1 / (_ALPHA_PL - 1))).round().astype(int), 1)
        if deg.sum() % 2:
            deg[deg.argmin()] += 1
        G = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(seed)))
        G.remove_edges_from(nx.selfloop_edges(G))
        A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
        ki = np.diff(A.indptr).astype(float)          # each node's own degree (fan-in)
        ki[ki == 0] = 1.0
        coo = A.tocoo()
        z = rng.normal(size=coo.row.size)
        vals = (mu + sigma * z) / ki[coo.row]         # per-edge std sigma/k_i -> field noise ~ 1/sqrt(k)
        return sparse.csr_array((vals, (coo.row, coo.col)), shape=(N, N))
    raise ValueError(
        f"unknown kind {kind!r} "
        "(expected 'fc', 'powerlaw', 'powerlaw_owndeg' or 'powerlaw_rowavg')")


def integrate(alpha, *, tmax, n_eval=1500, lam=0.0, seed=0,
              method="LSODA", rtol=1e-7, atol=1e-9):
    """Integrate the relative GLV in the share + log-scale split.

    Writing x_i = M w_i with w on the simplex (sum_i w_i = 1) and splitting off
    the scale gives a replicator for the shares plus a linear equation for ln M:

        f_i      = 1 - N w_i - N (alpha w)_i           (per-firm fitness, N = number of firms)
        <f>      = sum_i w_i f_i = g_eff               (aggregate growth rate)
        dw_i/dt  = w_i (f_i - <f>) + lam (1/N - w_i)   (replicator + optional share floor)
        d lnM/dt = g_eff

    The absolute abundance x_i = M w_i is NEVER formed, so nothing overflows even
    though M grows exponentially: w stays on the simplex and ln M grows linearly.
    The optional lam term is a persistent, mass-conserving immigration floor that
    fights condensation onto a single firm.

    Returns dict(t, W, lnM, success): W shape (N, n_eval), columns renormalised
    to the simplex; lnM the log aggregate (lnM(0)=0).
    """
    N = alpha.shape[0]
    rng = np.random.default_rng(seed)
    w0 = rng.uniform(0.5, 1.5, N)
    w0 /= w0.sum()

    def rhs(t, state):
        w = np.clip(state[:N], 0.0, None)
        s = w.sum()
        if s > 0:
            w = w / s
        f = 1.0 - N * w - N * (alpha @ w)
        fbar = float(w @ f)
        dw = w * (f - fbar) + lam * (1.0 / N - w)
        return np.concatenate([dw, [fbar]])

    r = solve_ivp(rhs, (0.0, tmax), np.concatenate([w0, [0.0]]), method=method,
                  t_eval=np.linspace(0.0, tmax, n_eval), rtol=rtol, atol=atol)
    W = np.clip(r.y[:N], 0.0, None)
    W = W / W.sum(0, keepdims=True)
    return dict(t=r.t, W=W, lnM=r.y[N], success=bool(r.success))


def growth_rate(t, lnM, frac=0.5):
    """Aggregate growth rate g_eff = d lnM/dt, as the slope of lnM over the late
    fraction `frac` of the trajectory (after the initial-condition transient)."""
    late = t > (1.0 - frac) * t[-1]
    return float(np.polyfit(t[late], lnM[late], 1)[0])


def survivors(W, floor=1e-6):
    """Boolean mask of firms whose share stays above `floor` across all of W
    (relative size S_i = N w_i, so floor is a fraction of the average firm)."""
    return W.min(axis=1) > floor

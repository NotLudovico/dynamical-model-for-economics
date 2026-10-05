"""Static heterogeneous cavity theory of the relative GLV with demographic noise and turnover.

One node at constant field h:  dS = S (h - S) dt + sqrt(D S) dW, exit at S = 0, rebirth at S = D.
The successive lives are i.i.d., so the node is a renewal process with stationary law

    p(x | h) = s(min(x, D)) m(x) / T(D),
    s'(y) = exp[-(2/D)(h y - y^2/2)],   m(x) = 2 / (D x s'(x)),   T(D) = int_0^inf s(min(x, D)) m(x) dx,

(occupation density of a 1D diffusion started at D, divided by the mean life T); the turnover rate is 1/T.

Cavity closure at zero reciprocity, on the degree classes of :mod:`relative_glv.hdmft`:
class-p field h = a - mu B_p + sigma sqrt(q_p) z,  B = K mean,  q = K Q,  sum_p w_p mean_p = 1,
with Q = E[<S>_h^2] (time-mean sizes quenched).  The static state is linearly stable while the Perron root
of sigma^2 diag(E[chi^2]) K is below one, chi = d<S>_h/dh; as D -> 0 chi tends to the survival step and the
line tends to the deterministic sigma_c of :func:`relative_glv.hdmft.heterogeneous_sigma_c`.
``K`` here is the class coupling ``kernel * weights[None, :] / connectance``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

_Z = np.linspace(-8.0, 8.0, 801)
_WZ = np.exp(-_Z**2 / 2) * (_Z[1] - _Z[0]) / np.sqrt(2 * np.pi)


def renewal_moments(h, D: float, n_lo: int = 1500, n_hi: int = 6000):
    """<S>, <S^2> and log T of the renewal law at fields ``h`` (integrals in logs: no overflow)."""
    h = np.atleast_1d(np.asarray(h, float))
    xmax = max(h.max(), 0.0) + 12 * np.sqrt(D) + 2.0
    x_lo = np.geomspace(1e-10 * D, D, n_lo)                        # flicker scale D/|h| needs a log grid
    x = np.concatenate([x_lo, np.linspace(D, xmax, n_hi)[1:]])
    f = np.exp(-(2 / D) * (h[:, None] * x_lo[None, :] - x_lo**2 / 2))   # s'(y) on [0, D]
    s_lo = x_lo[0] * f[:, :1] + np.concatenate(
        [np.zeros((h.size, 1)), np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * np.diff(x_lo), axis=1)], axis=1)
    s_min = np.concatenate([s_lo, np.repeat(s_lo[:, -1:], x.size - n_lo, axis=1)], axis=1)
    logG = np.log(s_min) + np.log(2 / (D * x))[None, :] + (2 / D) * (h[:, None] * x[None, :] - x**2 / 2)
    top = logG.max(1, keepdims=True)
    G = np.exp(logG - top)
    T = np.trapezoid(G, x, axis=1)
    return (np.trapezoid(G * x, x, axis=1) / T, np.trapezoid(G * x**2, x, axis=1) / T,
            np.log(T) + top[:, 0])


_tables: dict = {}


def renewal_table(D: float):
    """Fields, <S>, <S^2>, chi = d<S>/dh and log T on a fixed grid h in [-300, 25] (cached per D)."""
    if D not in _tables:
        hs = np.concatenate([np.linspace(-300, -20, 400)[:-1], np.linspace(-20, -5, 300)[:-1],
                             np.linspace(-5, 25, 3000)])
        n_hi = 6000 if D >= 1e-2 else 20000                       # plateau width sqrt(D/2) must be resolved
        parts = [renewal_moments(hs[i:i + 100], D, n_hi=n_hi) for i in range(0, hs.size, 100)]
        mean, second, logT = (np.concatenate([p[k] for p in parts]) for k in range(3))
        _tables[D] = (hs, mean, second, np.gradient(mean, hs), logT)
    return _tables[D]


_warm: dict = {}


def solve_renewal_fixed_point(mu, sigma, weights, class_coupling, D, iters=3000, tol=1e-10) -> dict:
    """Damped fixed-point iteration on (B, q); ``a = 1 - growth`` from the normalisation at each step."""
    hs, tm, ts, tc, tl = renewal_table(D)
    w, Kc = np.asarray(weights, float), np.asarray(class_coupling, float)
    B, q = _warm.get((D, w.size), (Kc.sum(1), Kc.sum(1)))

    def stats(B, q, a):
        h = a - mu * B[:, None] + sigma * np.sqrt(q)[:, None] * _Z[None, :]
        m = np.interp(h, hs, tm)
        return m @ _WZ, (m**2) @ _WZ, h, m

    def a_of(B, q):
        return brentq(lambda a: w @ stats(B, q, a)[0] - 1.0, -50.0, 50.0 + mu * B.max(), xtol=1e-12)

    for it in range(iters):
        mean, Q, _, _ = stats(B, q, a_of(B, q))
        B_new, q_new = Kc @ mean, Kc @ Q
        err = max(np.max(np.abs(B_new - B)), np.max(np.abs(np.log(q_new / q))))
        B, q = 0.5 * (B + B_new), np.sqrt(q * q_new)
        if err < tol:
            break
    _warm[(D, w.size)] = (B.copy(), q.copy())
    a = a_of(B, q)
    mean, Q, h, m = stats(B, q, a)
    chi2 = (np.interp(h, hs, tc) ** 2) @ _WZ
    radius = float(np.max(np.abs(np.linalg.eigvals(sigma**2 * chi2[:, None] * Kc))))
    temporal = sigma**2 * (Kc @ (np.interp(h, hs, ts) @ _WZ - Q))      # field variance from size fluctuations
    centre = a - mu * B
    total = w @ (sigma**2 * (Kc @ (np.interp(h, hs, ts) @ _WZ)) + (centre - w @ centre) ** 2)
    return dict(growth_rate=float(1.0 - a), stability_radius=radius, stable=radius < 1.0,
                turnover=float(w @ (np.exp(-np.interp(h, hs, tl)) @ _WZ)), class_mean=mean, B=B, q=q,
                temporal_share=float(w @ temporal / total), converged=bool(err < tol), iterations=it)


def renewal_sigma_c(mu, weights, class_coupling, D, bracket=(0.5, 2.5)) -> float:
    """Disorder at which the static state with demographic noise loses linear stability."""
    return brentq(lambda s: solve_renewal_fixed_point(mu, s, weights, class_coupling, D)["stability_radius"] - 1.0,
                  *bracket, xtol=1e-6)


def renewal_zero_growth(mu, weights, class_coupling, D, bracket=(0.5, 3.0)) -> float:
    """Disorder at which the static state's growth rate vanishes (the economy grows above it)."""
    return brentq(lambda s: solve_renewal_fixed_point(mu, s, weights, class_coupling, D)["growth_rate"],
                  *bracket, xtol=1e-6)

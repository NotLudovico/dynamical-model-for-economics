"""Relative-GLV DMFT fixed-point solver (relaxed / single-fixed-point phase).

The relative GLV is a disordered replicator. In the high-connectivity limit its
dynamics reduce to a single-site stochastic process whose stationary statistics
obey self-consistency equations for the survival fraction phi, the overlap q,
the response chi, and the growth rate g*. With Gaussian couplings of mean mu and
variance sigma^2 (gamma the symmetric/antisymmetric correlation), the survivors'
rescaled abundances are a clipped Gaussian, x* = (sqrt(q) sigma / v) max(Delta+z, 0),
with Delta the clipping threshold and v = 1 - gamma sigma^2 chi the renormalised
self-interaction. The Gaussian moments

    w_n(Delta) = int_{-Delta}^{inf} Dz (Delta+z)^n,    Dz = e^{-z^2/2} dz / sqrt(2 pi)

are known in closed form (w0=Phi, w1=Delta Phi + phi_pdf, w2=(1+Delta^2)Phi + Delta phi_pdf),
which collapses the self-consistency to a single scalar root for Delta:

    v^2 = sigma^2 w2(Delta),   chi = w0(Delta)/v,   normalisation M_1 = (sigma sqrt(q)/v) w1 = 1.

The mean competition mu is a *uniform* shift of every fitness, so it cancels from
the replicator's relative dynamics: the shape observables (Delta, q, chi, phi) are
mu-independent and mu only lowers the growth rate, g* = g0 - mu. At gamma=0 the
relaxed<->fluctuating boundary is mu-independent at sigma_c = sqrt(2) (set by
w0(Delta)=w2(Delta) -> Delta=0, v(0)^2 = sigma_c^2 / 2).

Full derivation: glv/docs/superpowers/specs/2026-06-21-relative-glv-dmft-derivation.md
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# Gaussian moments  w_n(Delta) = int_{-Delta}^{inf} Dz (Delta+z)^n,  Dz = e^{-z^2/2} dz / sqrt(2 pi)
def w0(d):
    return norm.cdf(d)


def w1(d):
    return d * norm.cdf(d) + norm.pdf(d)


def w2(d):
    return (1.0 + d * d) * norm.cdf(d) + d * norm.pdf(d)


def _v_of_delta(d, sigma, gamma):
    """v = 1 - gamma sigma^2 chi, eliminated via chi = w0/v -> v^2 - v + gamma sigma^2 w0 = 0."""
    if gamma == 0.0:
        return 1.0
    disc = 1.0 - 4.0 * gamma * sigma ** 2 * w0(d)
    if disc < 0.0:
        return np.nan          # symmetric-coupling branch breaks down (different physics)
    return 0.5 * (1.0 + np.sqrt(disc))


def sigma_c(gamma=0.0):
    """Disorder at the relaxed<->fluctuating boundary, eq (6).

    At threshold w0(Delta)=w2(Delta) -> Delta=0 (phi=1/2) universally, and the mean-square
    eq gives v(0)^2 = sigma_c^2 w2(0) = sigma_c^2/2.
    """
    if gamma == 0.0:
        return np.sqrt(2.0)

    def h(s):
        disc = 1.0 - 2.0 * gamma * s * s
        if disc < 0.0:
            return np.nan
        v0 = 0.5 * (1.0 + np.sqrt(disc))
        return v0 * v0 - 0.5 * s * s

    hi = 1.0 / np.sqrt(2.0 * gamma)          # sqrt-domain ceiling for gamma>0
    return brentq(h, 1e-6, hi * (1 - 1e-9), xtol=1e-12)


def solve_fixed_point(mu, sigma, gamma=0.0):
    """Solve the relaxed-phase self-consistency at one (mu, sigma, gamma)."""
    def resid(d):
        v = _v_of_delta(d, sigma, gamma)
        return v * v - sigma ** 2 * w2(d)

    bound = max(40.0, 5.0 / sigma)            # root Delta ~ v/sigma, so widen as sigma shrinks
    delta = brentq(resid, -bound, bound, xtol=1e-12, rtol=8.9e-16)
    v = _v_of_delta(delta, sigma, gamma)
    sqrtq = v / (sigma * w1(delta))          # from normalisation M_1 = (sigma sqrtq / v) w1 = 1
    g0 = 1.0 - sigma * sqrtq * delta          # mu-independent part of the growth rate
    return dict(
        mu=mu, sigma=sigma, gamma=gamma,
        delta=delta, q=sqrtq ** 2, sqrtq=sqrtq, v=v,
        phi=w0(delta), chi=w0(delta) / v,
        g0=g0, gstar=g0 - mu,
        stable=bool(sigma < sigma_c(gamma)),
    )


if __name__ == "__main__":
    print(f"sigma_c(gamma=0) = {sigma_c(0.0):.6f}  (= sqrt(2) = {np.sqrt(2):.6f})")
    for sig in (0.05, 0.5, 1.0, np.sqrt(2), 2.0):
        s = solve_fixed_point(mu=0.5, sigma=sig)
        print(f"  sigma={sig:5.3f}  Delta={s['delta']:+.3f}  phi={s['phi']:.3f}  "
              f"q={s['q']:.3f}  g*={s['gstar']:+.3f}  g0={s['g0']:+.3f}  stable={s['stable']}")

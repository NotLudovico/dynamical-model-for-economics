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


# ===================================================================== two-time DMFT
# Above sigma_c the relaxed fixed point above is unstable: shares fluctuate forever,
# C(t,s) does NOT freeze, and the single-site process must be solved self-consistently
# for the whole *function* C(t,s). At gamma=0 the Onsager memory term gamma sigma^2 int G y
# drops out, so the response G(t,s) is not needed for the dynamics. The ONLY self-consistent
# object is the noise covariance. The single-site law (appendix eq A.9, gamma=0, M_1=1):
#
#     d y / dt = y [ 1 - y + mu - g(t) + sigma eta(t) ],   <eta(t)eta(s)> = C(t,s),
#     g(t) = E[y F],  F = 1 - y + mu + sigma eta,   E[y(t)] = 1.
#
# Self-consistent scheme (an ensemble of n single-site firms IS the disorder average):
#   1. given C(t,s) on a time grid, Cholesky L (C = L L^T)
#   2. draw n noise paths eta_k = L @ randn(Nt)             (each ~ N(0, C))
#   3. integrate the ensemble jointly in share form (renormalise <y>=1 each step,
#      which is the -g(t) drift), accumulating ln M
#   4. remeasure C_new(t,s) = mean_k y_k(t) y_k(s)  (a Gram matrix, automatically PSD)
#   5. damp C <- (1-a) C + a C_new, iterate to convergence


def cholesky_psd(C, jitter=1e-10):
    """Cholesky of a (numerically) PSD covariance, with escalating jitter."""
    d = np.diag(C).mean()
    for k in range(8):
        try:
            return np.linalg.cholesky(C + (jitter * d * 10.0 ** k) * np.eye(len(C)))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("covariance not PSD even with jitter")


def draw_noise(L, n, rng):
    """n stationary-or-not Gaussian paths with covariance C = L L^T."""
    return (L @ rng.standard_normal((len(L), n))).T          # (n, Nt)


def _twotime_integrate(eta, mu, sigma, dt, u0, nsub=4, lam=0.0):
    """Integrate the share ensemble for fixed noise paths eta (n, Nt).

    Returns U (n, Nt) shares (<U[:,t]>=1 each t), g_t (Nt,) common growth rate.
    Single-site law: du/dt = u(F - g) + lam(1 - u). Operator-split per substep: exponential
    growth u *= exp(h(F-g)), then the firm-entry floor u += h lam (1-u), then renormalise
    <u>=1. lam>0 (entry) keeps abundances off the extinction boundary -> stationary phase;
    lam=0 -> the closed-economy aging idealisation. nsub substeps (eta interpolated) keep the
    explicit step stable at large sigma; the exp is clipped as a float-overflow guard.
    """
    n, Nt = eta.shape
    u = u0.copy()
    U = np.empty((n, Nt)); g_t = np.empty(Nt)
    h = dt / nsub
    for t in range(Nt):
        e0 = eta[:, t]; e1 = eta[:, t + 1] if t + 1 < Nt else e0
        F = 1.0 - u + mu + sigma * e0
        g_t[t] = float(u @ F) / n
        U[:, t] = u
        for j in range(nsub):                               # substep within [t, t+1]
            e = e0 + (e1 - e0) * (j + 0.5) / nsub
            F = 1.0 - u + mu + sigma * e
            g = float(u @ F) / n
            u = u * np.exp(np.clip(h * (F - g), -30.0, 30.0))
            if lam:
                u = u + h * lam * (1.0 - u)                 # firm entry (anti-extinction floor)
            u = np.maximum(u, 0.0); u *= n / u.sum()        # pin <u> = 1 each substep
    return U, g_t


def solve_twotime(mu, sigma, Nt=320, dt=0.2, n=4000, iters=34, burn=18, damp=0.4,
                  late=0.6, seed=0, lam=0.0, verbose=False):
    """Self-consistent two-time DMFT at gamma=0. Returns dict of observables.

    The self-consistency is a Monte-Carlo map (finite ensemble), so it does not converge
    to a point but fluctuates about the attractor. We iterate `iters` times and POOL the
    late-window samples of the post-`burn` iterations to average out the MC noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(Nt) * dt
    lo = int(late * Nt)                                     # late (stationary) window
    # warm start: a DECAYING C(t,s) = 1 + (q-1) exp(-|t-s|/tau0). A constant (frozen) C
    # is the unstable relaxed FP and the iteration sticks there; seeding decorrelation
    # lets the self-consistency find the true fluctuating attractor above sigma_c.
    rel = solve_fixed_point(-mu, sigma)                    # solver uses opposite mu sign
    q0 = max(rel["q"], 1.0)
    lag = np.abs(t[:, None] - t[None, :])
    C = 1.0 + (q0 - 1.0) * np.exp(-lag / 3.0)
    u0 = rng.uniform(0.5, 1.5, n); u0 *= n / u0.sum()

    hist = []; Cbar = np.zeros_like(C); nbar = 0
    U_pool, g_pool = [], []
    for it in range(iters):
        L = cholesky_psd(C)
        eta = draw_noise(L, n, rng)
        U, g_t = _twotime_integrate(eta, mu, sigma, dt, u0, lam=lam)
        Cn = (U.T @ U) / n                                  # C_new(t,s) = <y(t)y(s)>, PSD
        err = np.abs(Cn[lo:, lo:] - C[lo:, lo:]).mean()
        hist.append(err)
        C = (1 - damp) * C + damp * Cn
        if it >= burn:                                      # pool the tail to beat MC noise
            Cbar += Cn; nbar += 1
            U_pool.append(U[:, lo:]); g_pool.append(g_t[lo:])
        if verbose:
            print(f"  it{it:02d} err={err:.4f} q={Cn[lo:,lo:].diagonal().mean():.3f} "
                  f"g={g_t[lo:].mean():+.3f}", flush=True)

    Cbar /= nbar
    Up = np.concatenate(U_pool, axis=1)                    # (n, n_pooled_times)
    gp = np.concatenate(g_pool)
    q = float(Cbar[lo:, lo:].diagonal().mean())           # E[y^2]
    g_eff = float(gp.mean())                               # aggregate growth rate
    # survival is PER-RUN (each tail iteration is an independent noise realisation), then
    # averaged; threshold matches the sim's share>1e-6 (share = u/n, so u > 1e-6 n).
    thr = 1e-6 * n
    surv = float(np.mean([(W.min(1) > thr).mean() for W in U_pool]))
    taus = np.arange(0, Nt - lo)
    Ctau = np.array([np.mean([Cbar[a, a + d] for a in range(lo, Nt - d)]) for d in taus])
    decorr = float((Ctau[-1] - 1.0) / (Ctau[0] - 1.0))    # connected: ->0 decorrelated, 1 frozen
    return dict(mu=mu, sigma=sigma, t=t, C=Cbar, Ctau=Ctau, dt=dt, lo=lo,
                q=q, g_eff=g_eff, surv=surv, decorr=decorr, U=U, g_t=g_t,
                Up=Up, U_list=U_pool, err=float(np.mean(hist[burn:])), iters=it, rel=rel)

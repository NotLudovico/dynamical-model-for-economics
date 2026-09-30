"""Heterogeneous DMFT for the relative GLV on correlated degree classes.

The connection kernel ``P[p, r]`` is the probability that a node in class
``p`` is connected to a node in class ``r``.  ``weights[r]`` is the node
fraction in class ``r`` and ``connectance=C/N``.  At zero reciprocity and zero
immigration, the fixed-point law in class ``p`` is

    S_p = max(0, a - mu B_p + sigma sqrt(q_p) z),

where ``a = 1 - growth_rate``.  The class fields are coupled by the full
connection kernel, following Patil, Altieri and Aguirre-Lopez (2026), while
``a`` is fixed by the relative-model constraint E[S] = 1.

The sign convention matches :mod:`relative_glv.model`: ``mu > 0`` is mean
competition.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, least_squares

from relative_glv.dmft import solve_fixed_point, w0, w1, w2


def fixed_point_abundance(field: np.ndarray, lam: float) -> np.ndarray:
    """Positive stationary root for ``S(field-S)+lam(1-S)=0``."""
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    field = np.asarray(field, dtype=float)
    if lam == 0.0:
        return np.maximum(field, 0.0)
    linear = field - lam
    discriminant = np.sqrt(linear**2 + 4.0 * lam)
    return np.where(
        linear >= 0.0,
        0.5 * (linear + discriminant),
        2.0 * lam / (discriminant - linear),
    )


def bin_degree_distribution(degrees: np.ndarray, n_bins: int = 30) -> dict:
    """Compress a realized degree sequence into equal-population classes.

    Each representative is the mean degree inside its class, so the realized
    mean degree (and hence ``C/N``) is preserved exactly by the binned law.
    """
    degrees = np.asarray(degrees, dtype=float)
    if degrees.ndim != 1 or degrees.size == 0:
        raise ValueError("degrees must be a non-empty one-dimensional array")
    if np.any(degrees <= 0.0):
        raise ValueError("all nodes must have positive degree")
    if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")

    chunks = np.array_split(np.sort(degrees), min(int(n_bins), degrees.size))
    counts = np.array([chunk.size for chunk in chunks], dtype=float)
    representatives = np.array([chunk.mean() for chunk in chunks], dtype=float)
    weights = counts / degrees.size
    return {
        "degree_fractions": representatives / degrees.size,
        "representative_degrees": representatives,
        "weights": weights,
        "connectance": float(degrees.mean() / degrees.size),
    }


def soft_configuration_kernel(
    degree_fractions: np.ndarray,
    weights: np.ndarray,
    connectance: float,
) -> dict:
    """Infer the maximum-entropy connection kernel from observed degrees.

    Hidden degree fractions ``h_p`` define

    ``P[p, r] = h_p h_r / (h_p h_r + connectance)``.

    They are chosen so that ``P @ weights`` reproduces each observed degree
    fraction ``k_p/N``.  This is the soft-configuration mapping used to retain
    structural-cutoff correlations in ultra-small-world HDMFT.
    """
    degree_fractions = np.asarray(degree_fractions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if degree_fractions.ndim != 1 or weights.shape != degree_fractions.shape:
        raise ValueError("degree_fractions and weights must be one-dimensional arrays of equal length")
    if np.any(degree_fractions <= 0.0) or np.any(degree_fractions >= 1.0):
        raise ValueError("degree fractions must lie strictly between zero and one")
    if np.any(weights <= 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be positive and sum to one")
    if connectance <= 0.0 or connectance >= 1.0:
        raise ValueError("connectance must lie strictly between zero and one")
    if not np.isclose(weights @ degree_fractions, connectance, rtol=1e-8, atol=1e-12):
        raise ValueError("connectance must equal the weighted mean degree fraction")

    def kernel_from_log_hidden(log_hidden):
        hidden = np.exp(log_hidden)
        product = hidden[:, None] * hidden[None, :]
        return hidden, product / (product + connectance)

    def residual(log_hidden):
        _hidden, kernel = kernel_from_log_hidden(log_hidden)
        return (kernel @ weights - degree_fractions) / degree_fractions

    solution = least_squares(
        residual,
        np.log(degree_fractions),
        bounds=(-50.0, 20.0),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=20_000,
    )
    hidden, kernel = kernel_from_log_hidden(solution.x)
    residual_norm = float(np.max(np.abs(kernel @ weights - degree_fractions)))
    if not solution.success or residual_norm > 1e-10:
        raise RuntimeError(
            f"hidden-degree inversion did not converge (max residual {residual_norm:.3g})"
        )
    return {"hidden_degree_fractions": hidden, "kernel": kernel, "residual": residual_norm}


def solve_heterogeneous_fixed_point(
    mu: float,
    sigma: float,
    weights: np.ndarray,
    connection_kernel: np.ndarray,
    connectance: float,
    lam: float = 0.0,
) -> dict:
    """Solve the binned heterogeneous fixed point at zero reciprocity.

    The returned stability radius is the Perron root of
    ``sigma**2 * diag(response_weight) * P * diag(weights) / connectance``.
    At zero immigration the response weight is the survival probability.  At
    positive immigration it is ``E[(S**2 / (S**2 + lam))**2]``.  The unique
    fixed point is linearly stable when the radius is below one.
    """
    weights = np.asarray(weights, dtype=float)
    kernel = np.asarray(connection_kernel, dtype=float)
    n_classes = weights.size
    if weights.ndim != 1 or n_classes == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if kernel.shape != (n_classes, n_classes):
        raise ValueError("connection_kernel must be square with one row per class")
    if np.any(weights <= 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be positive and sum to one")
    if connectance <= 0.0 or connectance > 1.0:
        raise ValueError("connectance must lie in (0, 1]")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    if np.any(kernel < 0.0) or np.any(kernel > 1.0):
        raise ValueError("connection probabilities must lie in [0, 1]")

    class_coupling = kernel * weights[None, :] / connectance
    relative_degree = class_coupling.sum(axis=1)
    if np.any(relative_degree <= 0.0):
        raise ValueError("every class must have positive expected degree")

    homogeneous = solve_fixed_point(mu=mu, sigma=sigma)
    B0 = relative_degree.copy()
    q0 = homogeneous["q"] * relative_degree
    a0 = 1.0 - homogeneous["gstar"]
    x0 = np.concatenate([B0, np.log(q0), [a0]])
    if lam:
        quadrature_x, quadrature_w = np.polynomial.hermite.hermgauss(80)
        gaussian_z = np.sqrt(2.0) * quadrature_x
        gaussian_w = quadrature_w / np.sqrt(np.pi)

    def moments(x):
        B = x[:n_classes]
        q = np.exp(x[n_classes : 2 * n_classes])
        a = x[-1]
        delta = (a - mu * B) / (sigma * np.sqrt(q))
        if lam == 0.0:
            mean = sigma * np.sqrt(q) * w1(delta)
            second = sigma**2 * q * w2(delta)
            phi = w0(delta)
            response_weight = phi
        else:
            field = a - mu * B[:, None] + sigma * np.sqrt(q)[:, None] * gaussian_z[None, :]
            abundance = fixed_point_abundance(field, lam)
            mean = abundance @ gaussian_w
            second = abundance**2 @ gaussian_w
            susceptibility = abundance**2 / (abundance**2 + lam)
            response_weight = susceptibility**2 @ gaussian_w
            phi = np.ones(n_classes)
        return B, q, a, delta, mean, second, phi, response_weight

    def residual(x):
        B, q, _a, _delta, mean, second, _phi, _response_weight = moments(x)
        target_B = class_coupling @ mean
        target_q = class_coupling @ second
        return np.concatenate(
            [
                B - target_B,
                np.log(q) - np.log(np.maximum(target_q, 1e-300)),
                [weights @ mean - 1.0],
            ]
        )

    lower = np.concatenate([np.zeros(n_classes), np.full(n_classes, -700.0), [-np.inf]])
    upper = np.full(2 * n_classes + 1, np.inf)
    solution = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=20_000,
    )
    B, q, a, delta, mean, second, phi, response_weight = moments(solution.x)
    stability_matrix = sigma**2 * response_weight[:, None] * class_coupling
    stability_radius = float(np.max(np.abs(np.linalg.eigvals(stability_matrix))))
    residual_norm = float(np.max(np.abs(residual(solution.x))))
    if not solution.success or residual_norm > 1e-8:
        raise RuntimeError(
            f"heterogeneous fixed point did not converge (max residual {residual_norm:.3g})"
        )

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "lam": float(lam),
        "growth_rate": float(1.0 - a),
        "survival": float(weights @ phi),
        "second_moment": float(weights @ second),
        "class_mean": mean,
        "class_second_moment": second,
        "class_survival": phi,
        "class_response_weight": response_weight,
        "B": B,
        "q": q,
        "delta": delta,
        "relative_degree": relative_degree,
        "stability_radius": stability_radius,
        "stable": bool(stability_radius < 1.0),
        "residual": residual_norm,
    }


def heterogeneous_sigma_c(
    mu: float,
    weights: np.ndarray,
    connection_kernel: np.ndarray,
    connectance: float,
    lam: float = 0.0,
    bracket: tuple[float, float] = (0.2, 4.0),
) -> dict:
    """Locate the loss of fixed-point stability for a given graph kernel."""

    def stability_residual(sigma):
        fixed_point = solve_heterogeneous_fixed_point(
            mu=mu,
            sigma=sigma,
            weights=weights,
            connection_kernel=connection_kernel,
            connectance=connectance,
            lam=lam,
        )
        return fixed_point["stability_radius"] - 1.0

    sigma_c = float(brentq(stability_residual, *bracket, xtol=1e-10, rtol=1e-10))
    fixed_point = solve_heterogeneous_fixed_point(
        mu=mu,
        sigma=sigma_c,
        weights=weights,
        connection_kernel=connection_kernel,
        connectance=connectance,
        lam=lam,
    )
    return {"sigma_c": sigma_c, "fixed_point": fixed_point}


def heterogeneous_zero_growth(
    sigma: float,
    weights: np.ndarray,
    connection_kernel: np.ndarray,
    connectance: float,
    lam: float = 0.0,
    bracket: tuple[float, float] = (0.0, 5.0),
) -> dict:
    """Locate ``g*=0`` at fixed disorder on the heterogeneous fixed-point branch.

    ``mu_c`` uses the code convention in which positive mean interaction is
    competition.  ``mu_thesis=-mu_c`` is the coordinate used in the thesis.
    The root remains mathematically defined when the fixed point is unstable;
    ``controlled`` is true only where the stationary fixed-point theory is
    linearly stable.
    """

    def growth_residual(mu_c):
        fixed_point = solve_heterogeneous_fixed_point(
            mu=mu_c,
            sigma=sigma,
            weights=weights,
            connection_kernel=connection_kernel,
            connectance=connectance,
            lam=lam,
        )
        return fixed_point["growth_rate"]

    mu_c = float(brentq(growth_residual, *bracket, xtol=1e-10, rtol=1e-10))
    fixed_point = solve_heterogeneous_fixed_point(
        mu=mu_c,
        sigma=sigma,
        weights=weights,
        connection_kernel=connection_kernel,
        connectance=connectance,
        lam=lam,
    )
    return {
        "sigma": float(sigma),
        "mu_c": mu_c,
        "mu_thesis": -mu_c,
        "controlled": fixed_point["stable"],
        "fixed_point": fixed_point,
    }


def heterogeneous_zero_growth_line(
    sigmas: np.ndarray,
    weights: np.ndarray,
    connection_kernel: np.ndarray,
    connectance: float,
    lam: float = 0.0,
    bracket: tuple[float, float] = (0.0, 5.0),
) -> dict:
    """Trace the fixed-point ``g*=0`` contour over an array of disorder values."""
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.ndim != 1 or sigmas.size == 0:
        raise ValueError("sigmas must be a non-empty one-dimensional array")

    points = [
        heterogeneous_zero_growth(
            sigma=float(sigma),
            weights=weights,
            connection_kernel=connection_kernel,
            connectance=connectance,
            lam=lam,
            bracket=bracket,
        )
        for sigma in sigmas
    ]
    fixed_points = [point["fixed_point"] for point in points]
    return {
        "sigma": sigmas.copy(),
        "mu_c": np.array([point["mu_c"] for point in points]),
        "mu_thesis": np.array([point["mu_thesis"] for point in points]),
        "growth_rate": np.array([point["growth_rate"] for point in fixed_points]),
        "stability_radius": np.array(
            [point["stability_radius"] for point in fixed_points]
        ),
        "controlled": np.array([point["controlled"] for point in points], dtype=bool),
        "points": points,
    }


def solve_twotime_heterogeneous(
    mu: float,
    sigma: float,
    connection_kernel: np.ndarray,
    connectance: float,
    n_per_class: int = 400,
    Nt: int = 960,
    dt: float = 0.25,
    iters: int = 24,
    burn: int = 12,
    damp: float = 0.5,
    lam: float = 0.0,
    nsub: int = 4,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Two-time HDMFT on equal-population degree classes (fluctuating phase, gamma=0).

    Class ``p`` runs the effective process of eq. (dmft-heterogeneous-process),

        dS/dt = S [1 - S - mu B_p(t) + sigma eta_p(t) - g(t)] + lam (1 - S),
        <eta_p(t) eta_p(s)> = q_p(t,s),

    closed by  B_p = sum_r K_pr E[S_r(t)],  q_p = sum_r K_pr E[S_r(t) S_r(s)],
    K = P diag(rho) / c,  and E[S] = 1 across all classes (the -g(t) drift).
    Classes must have equal weight (``bin_degree_distribution`` output), so the
    ensemble of ``n_per_class`` paths per class IS the degree distribution.
    Like ``dmft.solve_twotime`` this is a Monte-Carlo fixed-point map: the
    post-``burn`` iterations are pooled. Returns the final ensemble paths ``U``
    (n_classes, n_per_class, Nt) plus the pooled class fields.
    """
    from relative_glv.dmft import cholesky_psd

    kernel = np.asarray(connection_kernel, dtype=float)
    n_cls = kernel.shape[0]
    K = kernel / (n_cls * connectance)               # rho_r = 1/n_cls
    rel_k = K.sum(axis=1)
    rng = np.random.default_rng(seed)
    t = np.arange(Nt) * dt
    lag = np.abs(t[:, None] - t[None, :])
    q = rel_k[:, None, None] * (1.0 + 2.0 * np.exp(-lag / 3.0))[None]   # decaying warm start
    B = np.repeat(rel_k[:, None], Nt, axis=1)
    u0 = rng.uniform(0.5, 1.5, (n_cls, n_per_class))
    u0 /= u0.mean()
    h = dt / nsub
    q_bar, B_bar, n_bar, hist = np.zeros_like(q), np.zeros_like(B), 0, []

    for it in range(iters):
        eta = np.stack([cholesky_psd(q[p]) @ rng.standard_normal((Nt, n_per_class))
                        for p in range(n_cls)], axis=1)          # (Nt, n_cls, n_per)
        u, U, g_t = u0.copy(), np.empty((Nt, n_cls, n_per_class)), np.empty(Nt)
        for s in range(Nt):
            U[s] = u
            e0, e1 = eta[s], eta[min(s + 1, Nt - 1)]
            b0, b1 = B[:, s, None], B[:, min(s + 1, Nt - 1), None]
            for j in range(nsub):
                x = (j + 0.5) / nsub
                F = 1.0 - u - mu * (b0 + (b1 - b0) * x) + sigma * (e0 + (e1 - e0) * x)
                g = float((u * F).mean())
                if j == 0:
                    g_t[s] = g
                u = u * np.exp(np.clip(h * (F - g), -30.0, 30.0))
                if lam:
                    u = u + h * lam * (1.0 - u)
                u = np.maximum(u, 0.0)
                u /= u.mean()                                    # E[S] = 1
        Up = U.transpose(1, 0, 2)                                # (n_cls, Nt, n_per)
        C_r = Up @ Up.transpose(0, 2, 1) / n_per_class           # class autocorrelation
        q_new = np.tensordot(K, C_r, axes=1)
        B_new = K @ U.mean(axis=2).T
        err = float(np.abs(q_new - q).mean() / np.abs(q).mean())
        hist.append(err)
        q = (1 - damp) * q + damp * q_new
        B = (1 - damp) * B + damp * B_new
        if it >= burn:
            q_bar += q_new; B_bar += B_new; n_bar += 1
        if verbose:
            print(f"  it{it:02d} rel.err={err:.4f} g={g_t[Nt // 2:].mean():+.3f}", flush=True)

    return {
        "t": t, "U": np.moveaxis(U, 0, -1), "g_t": g_t,
        "q": q_bar / n_bar, "B": B_bar / n_bar, "rel_k": rel_k,
        "err": float(np.mean(hist[burn:])), "hist": np.array(hist),
    }

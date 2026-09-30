import numpy as np

import relative_glv
from relative_glv.dmft import solve_fixed_point
from relative_glv.hdmft import (
    bin_degree_distribution,
    fixed_point_abundance,
    heterogeneous_sigma_c,
    heterogeneous_zero_growth,
    heterogeneous_zero_growth_line,
    soft_configuration_kernel,
    solve_heterogeneous_fixed_point,
)


def test_single_degree_class_recovers_homogeneous_dmft():
    """A one-class connection kernel must reduce exactly to the FC theory."""
    mu, sigma, connectance = 0.5, 1.0, 0.2
    homogeneous = solve_fixed_point(mu=mu, sigma=sigma)
    heterogeneous = solve_heterogeneous_fixed_point(
        mu=mu,
        sigma=sigma,
        weights=np.array([1.0]),
        connection_kernel=np.array([[connectance]]),
        connectance=connectance,
    )

    assert np.isclose(heterogeneous["growth_rate"], homogeneous["gstar"], atol=1e-8)
    assert np.isclose(heterogeneous["survival"], homogeneous["phi"], atol=1e-8)
    assert np.isclose(heterogeneous["second_moment"], homogeneous["q"], atol=1e-8)
    assert np.isclose(
        heterogeneous["stability_radius"], sigma**2 * homogeneous["phi"], atol=1e-8
    )


def test_soft_configuration_kernel_reproduces_observed_class_degrees():
    weights = np.array([0.7, 0.3])
    degree_fractions = np.array([0.08, 0.22])
    connectance = float(weights @ degree_fractions)

    result = soft_configuration_kernel(
        degree_fractions=degree_fractions,
        weights=weights,
        connectance=connectance,
    )

    assert np.allclose(result["kernel"] @ weights, degree_fractions, atol=1e-10)
    assert np.allclose(result["kernel"], result["kernel"].T, atol=1e-12)
    assert np.all((result["kernel"] >= 0.0) & (result["kernel"] <= 1.0))


def test_degree_binning_preserves_the_realized_mean_degree():
    degrees = np.array([1, 1, 2, 2, 4, 8, 16, 32])
    result = bin_degree_distribution(degrees, n_bins=3)

    assert result["weights"].size <= 3
    assert np.isclose(result["weights"].sum(), 1.0)
    assert np.isclose(
        result["weights"] @ result["degree_fractions"], degrees.mean() / degrees.size
    )
    assert np.all(np.diff(result["degree_fractions"]) >= 0.0)


def test_heterogeneous_sigma_c_recovers_sqrt_two_for_one_class():
    critical = heterogeneous_sigma_c(
        mu=1.7,
        weights=np.array([1.0]),
        connection_kernel=np.array([[0.1]]),
        connectance=0.1,
    )

    assert np.isclose(critical["sigma_c"], np.sqrt(2.0), atol=1e-7)
    assert np.isclose(critical["fixed_point"]["stability_radius"], 1.0, atol=1e-7)


def test_hdmft_solver_is_available_from_the_public_package():
    assert relative_glv.solve_heterogeneous_fixed_point is solve_heterogeneous_fixed_point
    assert relative_glv.heterogeneous_sigma_c is heterogeneous_sigma_c
    assert relative_glv.heterogeneous_zero_growth is heterogeneous_zero_growth
    assert relative_glv.heterogeneous_zero_growth_line is heterogeneous_zero_growth_line


def test_immigration_fixed_point_is_positive_and_solves_the_quadratic():
    field = np.array([-2.0, 0.0, 3.0])
    lam = 1e-3
    abundance = fixed_point_abundance(field, lam)

    assert np.all(abundance > 0.0)
    assert np.allclose(abundance**2 - (field - lam) * abundance - lam, 0.0, atol=1e-12)


def test_fixed_point_with_immigration_preserves_relative_size_normalisation():
    result = solve_heterogeneous_fixed_point(
        mu=1.7,
        sigma=1.4,
        weights=np.array([1.0]),
        connection_kernel=np.array([[0.1]]),
        connectance=0.1,
        lam=1e-3,
    )

    assert np.isclose(result["class_mean"][0], 1.0, atol=1e-8)
    assert result["survival"] == 1.0
    assert 0.0 < result["class_response_weight"][0] < 1.0


def test_zero_growth_point_recovers_homogeneous_dmft_growth_intercept():
    sigma, connectance = 1.0, 0.1
    expected_mu_c = solve_fixed_point(mu=0.0, sigma=sigma)["g0"]

    result = heterogeneous_zero_growth(
        sigma=sigma,
        weights=np.array([1.0]),
        connection_kernel=np.array([[connectance]]),
        connectance=connectance,
    )

    assert np.isclose(result["mu_c"], expected_mu_c, atol=1e-8)
    assert np.isclose(result["mu_thesis"], -expected_mu_c, atol=1e-8)
    assert np.isclose(result["fixed_point"]["growth_rate"], 0.0, atol=1e-9)
    assert result["controlled"]


def test_zero_growth_line_marks_unstable_fixed_point_continuation():
    connectance = 0.1
    result = heterogeneous_zero_growth_line(
        sigmas=np.array([1.0, 1.5]),
        weights=np.array([1.0]),
        connection_kernel=np.array([[connectance]]),
        connectance=connectance,
    )

    assert np.allclose(result["sigma"], [1.0, 1.5])
    assert np.allclose(result["mu_thesis"], -result["mu_c"])
    assert np.array_equal(result["controlled"], [True, False])
    assert np.allclose(result["growth_rate"], 0.0, atol=1e-9)


def test_twotime_one_class_mean_competition_cancels():
    """Regular graph: mu B = mu is a uniform shift absorbed by g(t); shapes unchanged."""
    from relative_glv.hdmft import solve_twotime_heterogeneous

    kw = dict(sigma=1.8, connection_kernel=np.array([[0.1]]), connectance=0.1,
              n_per_class=200, Nt=60, dt=0.25, iters=3, burn=1, seed=3)
    a = solve_twotime_heterogeneous(mu=0.0, **kw)
    b = solve_twotime_heterogeneous(mu=1.5, **kw)
    np.testing.assert_allclose(a["U"], b["U"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(a["g_t"] - b["g_t"], 1.5, atol=1e-8)
    np.testing.assert_allclose(a["B"], 1.0, atol=1e-12)

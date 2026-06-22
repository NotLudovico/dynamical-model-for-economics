import numpy as np
from relative_glv.dmft import solve_fixed_point, sigma_c


def test_sigma_c_is_sqrt2_at_gamma0():
    assert abs(sigma_c(0.0) - np.sqrt(2.0)) < 1e-9


def test_mu_is_a_uniform_growth_shift():
    # mu cancels in the replicator's relative dynamics: shape (phi, q, delta) is
    # mu-independent and g* = g0 - mu.
    a = solve_fixed_point(mu=0.0, sigma=1.0)
    b = solve_fixed_point(mu=1.0, sigma=1.0)
    assert abs(a["phi"] - b["phi"]) < 1e-9
    assert abs(a["q"] - b["q"]) < 1e-9
    assert abs((b["gstar"] - a["gstar"]) - (-1.0)) < 1e-9


def test_survival_decreases_with_disorder():
    # more disorder -> fewer survivors (phi falls), in the relaxed phase
    phis = [solve_fixed_point(0.5, s)["phi"] for s in (0.2, 0.6, 1.0, 1.3)]
    assert all(x > y for x, y in zip(phis, phis[1:]))
    assert solve_fixed_point(0.5, 1.0)["stable"] is True
    assert solve_fixed_point(0.5, 2.0)["stable"] is False

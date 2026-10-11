import numpy as np

from relative_glv.model import configuration_coupling
from relative_glv.msb import excess_kurtosis, hill, lives, tail_exponent, zeta_bins


def test_lives_splits_on_deaths_and_skips_short_runs():
    S = np.array([[1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 2.0, 4.0]])
    L = np.array([[0, 0, 0, 1, 1, 1, 2, 3, 3]])    # lives: samples 0-2, 3-5, 6, 7-8
    out = lives(S, L)
    assert [x[2] for x in out] == [2, 2]           # lives with fewer than 3 samples are skipped
    assert out[0][5] == 0 and out[1][6] == 3
    np.testing.assert_allclose(out[0][0], 2.0)
    np.testing.assert_allclose(out[0][4], np.diff(np.log([1.0, 2.0, 3.0])))
    S2, L2 = np.array([[1.0, 2.0, 4.0]]), np.array([[0, 0, 0]])
    assert lives(S2, L2) == []                     # constant growth: zero volatility, dropped


def test_hill_recovers_pareto_index():
    rng = np.random.default_rng(0)
    z = (1 - rng.random(400_000)) ** (-1 / 1.5)
    assert abs(hill(z, 0.01) - 1.5) < 0.1


def test_kurtosis_and_tail_exponent_of_gaussian_and_laplace():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1_000_000)
    assert abs(excess_kurtosis(x)) < 0.05
    assert abs(excess_kurtosis(rng.laplace(size=1_000_000)) - 3) < 0.2
    assert tail_exponent(x) > tail_exponent(rng.laplace(size=1_000_000))


def test_zeta_bins_of_an_exact_power_law():
    S = np.geomspace(1e-2, 1e1, 5000)
    z, res, v0, v1, dec, loc = zeta_bins(S, 0.5 * S ** -0.2)
    np.testing.assert_allclose(z, [0.2, 0.4, 0.6, 0.8], atol=1e-9)
    assert abs(res) < 1e-9 and v0 > v1 and abs(dec - 3) < 0.2
    np.testing.assert_allclose(loc, 0.2, atol=1e-9)


def test_configuration_coupling_degrees_and_scaling():
    A = configuration_coupling(4000, 1.76, 1.75, alpha=2.5, mean_degree=100, seed=0)
    deg = np.diff(A.indptr)
    assert 75 < deg.mean() < 105                   # merged multi-edges of the hubs lower it
    assert (A.diagonal() == 0).all()
    C_eff = deg.mean()
    np.testing.assert_allclose(A.data.mean() * C_eff, 1.76, rtol=0.05)
    np.testing.assert_allclose(A.data.std() * np.sqrt(C_eff), 1.75, rtol=0.05)
    R = configuration_coupling(2000, 0.0, 1.0, mean_degree=50, seed=1, regular=True)
    d = np.diff(R.indptr)
    assert d.max() <= 50 and d.mean() > 48

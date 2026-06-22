import numpy as np
from scipy import sparse
from relative_glv.model import coupling


def test_fc_shape_diag_and_scaling():
    N, mu, sigma = 400, 1.0, 0.8
    a = coupling(N, mu, sigma, kind="fc", seed=0)
    assert a.shape == (N, N)
    assert np.allclose(np.diag(a), 0.0)            # zero self-interaction
    off = a[~np.eye(N, dtype=bool)]
    assert abs(off.mean() - mu / N) < 5e-4          # mean mu/N
    assert abs(off.std() - sigma / np.sqrt(N)) < 5e-3   # std sigma/sqrt(N)


def test_fc_is_deterministic_in_seed():
    a = coupling(50, 0.5, 1.0, kind="fc", seed=7)
    b = coupling(50, 0.5, 1.0, kind="fc", seed=7)
    assert np.array_equal(a, b)


def test_powerlaw_is_sparse_square_zero_diagonal():
    N = 600
    a = coupling(N, 1.0, 1.5, kind="powerlaw", seed=1)
    assert sparse.issparse(a)
    assert a.shape == (N, N)
    assert np.allclose(a.diagonal(), 0.0)           # config model removes self-loops
    assert a.nnz > 0


def test_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        coupling(10, 0.0, 1.0, kind="banana")


from relative_glv.model import integrate, growth_rate, survivors


def test_integrate_keeps_shares_on_the_simplex():
    a = coupling(200, 1.0, 0.8, kind="fc", seed=0)
    r = integrate(a, tmax=40.0, n_eval=200, seed=0)
    assert r["success"]
    assert r["W"].shape == (200, 200)
    assert np.allclose(r["W"].sum(axis=0), 1.0, atol=1e-8)   # simplex preserved
    assert np.isfinite(r["lnM"]).all()                        # ln M never overflows


def test_relaxed_economy_grows_at_negative_mu():
    # weak competition (mu small) in the relaxed phase -> aggregate grows: g_eff > 0
    a = coupling(300, 0.0, 0.8, kind="fc", seed=1)
    r = integrate(a, tmax=60.0, n_eval=300, seed=1)
    assert growth_rate(r["t"], r["lnM"]) > 0.0


def test_survivors_mask_length_and_dtype():
    a = coupling(150, 1.0, 0.8, kind="fc", seed=2)
    r = integrate(a, tmax=30.0, n_eval=150, seed=2)
    m = survivors(r["W"])
    assert m.shape == (150,) and m.dtype == bool

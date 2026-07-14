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


def test_powerlaw_with_min_degree_one_includes_low_degree_nodes():
    a = coupling(2_000, 1.0, 1.5, kind="powerlaw", seed=4, mean_degree=100, min_degree=1)
    degree = np.diff(a.indptr)

    assert degree.min() <= 2
    assert (degree < 10).any()


def test_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        coupling(10, 0.0, 1.0, kind="banana")


def test_powerlaw_field_variance_grows_with_degree():
    # Mean-degree normalization: a firm with k_i neighbours sums k_i couplings of
    # variance sigma^2/C, so the disorder-field variance (row sum of squares) grows
    # with its degree -- the channel through which the graph reaches the firm-level
    # statistics.
    from scipy.stats import spearmanr
    N = 1500
    mean = coupling(N, 1.0, 1.5, kind="powerlaw", seed=3)
    assert sparse.issparse(mean) and mean.shape == (N, N) and mean.nnz > 0
    mean_ss = np.asarray(mean.multiply(mean).sum(1)).ravel()
    rho_mean = spearmanr(np.diff(mean.indptr), mean_ss).correlation
    assert rho_mean > 0.7               # field variance grows with degree


def test_powerlaw_rowavg_field_variance_falls_with_degree():
    # Row-average variant a_ij=(mu+sigma z)/k_i: the interaction is the MEAN of O(1)
    # couplings over the k_i neighbours, so per-edge std ~ sigma/k_i and the field
    # variance (row sum of squares) FALLS as 1/k_i -- hubs self-average their noise away
    # (field noise ~ k^-1/2). Opposite extreme to mean-degree (grows with k).
    from scipy.stats import spearmanr
    N = 1500
    a = coupling(N, 1.0, 1.5, kind="powerlaw_rowavg", seed=3)
    assert sparse.issparse(a) and a.shape == (N, N) and a.nnz > 0
    deg = np.diff(a.indptr).astype(float)
    ss = np.asarray(a.multiply(a).sum(1)).ravel()
    rho = spearmanr(deg, ss).correlation
    assert rho < -0.7                    # field variance falls with degree (hubs quieter)


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

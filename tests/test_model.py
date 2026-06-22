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

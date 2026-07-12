import numpy as np


def test_economy_ccdf_gives_each_economy_equal_weight():
    from relative_glv.size_distribution import economy_ccdf

    result = economy_ccdf(
        [np.array([1.0, 100.0]), np.ones(4)],
        np.array([1.0, 10.0]),
    )

    np.testing.assert_allclose(result["by_economy"], [[1.0, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(result["mean"], [1.0, 0.25])


def test_hill_tail_index_recovers_known_log_spaced_tail():
    from relative_glv.size_distribution import hill_tail_index

    tail_index = hill_tail_index(np.array([1.0, np.e, np.e**2]))

    assert tail_index == 1.0

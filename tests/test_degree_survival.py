import numpy as np
import warnings


def test_survival_by_degree_returns_binwise_rates_and_counts():
    from relative_glv.degree_survival import survival_by_degree

    result = survival_by_degree(
        degree=np.array([1.0, 2.0, 3.0, 4.0, 8.0]),
        survived=np.array([True, False, True, False, False]),
        bin_edges=np.array([1.0, 3.0, 9.0]),
    )

    np.testing.assert_array_equal(result["count"], [2, 3])
    np.testing.assert_array_equal(result["survived"], [1, 1])
    np.testing.assert_allclose(result["probability"], [0.5, 1 / 3])
    assert np.all(result["ci_low"] < result["probability"])
    assert np.all(result["ci_high"] > result["probability"])


def test_survival_by_degree_by_economy_weights_each_economy_equally():
    from relative_glv.degree_survival import survival_by_degree_by_economy

    result = survival_by_degree_by_economy(
        degree=np.array([2.0, 2.0, 2.0, 2.0]),
        survived=np.array([True, False, False, False]),
        economy=np.array([0, 1, 1, 1]),
        bin_edges=np.array([1.0, 3.0]),
    )

    assert result["count"][0] == 4
    assert result["economies"][0] == 2
    assert result["probability"][0] == 0.5


def test_reentry_masks_distinguish_reentry_from_first_entry():
    from relative_glv.degree_survival import reentry_masks

    active = np.array([
        [True, True, False, True],
        [False, True, True, True],
        [True, True, True, True],
        [False, False, False, False],
    ])

    result = reentry_masks(active)

    np.testing.assert_array_equal(result["persistent"], [False, False, True, False])
    np.testing.assert_array_equal(result["exited"], [True, False, False, False])
    np.testing.assert_array_equal(result["reentered"], [True, False, False, False])


def test_survival_by_degree_handles_empty_bins_without_warnings():
    from relative_glv.degree_survival import survival_by_degree

    with warnings.catch_warnings(record=True) as caught:
        result = survival_by_degree(
            degree=np.array([1.0, 2.0]),
            survived=np.array([True, False]),
            bin_edges=np.array([1.0, 3.0, 9.0]),
        )

    assert not caught
    assert np.isnan(result["probability"][1])

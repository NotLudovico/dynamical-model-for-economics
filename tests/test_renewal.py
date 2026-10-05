import numpy as np


def test_renewal_law_matches_plateau_and_simulated_turnover():
    from relative_glv.renewal import renewal_moments
    mean, second, logT = renewal_moments([0.0, 3.0], D=0.1)
    # h = 0: one-node simulation (2000 time units, 400 nodes) gave turnover 0.2603, <S> 0.1238
    assert abs(np.exp(-logT[0]) - 0.2603) < 0.003
    assert abs(mean[0] - 0.1238) < 0.002
    # deep plateau: S ~ N(h, D/2) up to O(D/h) corrections
    assert abs(mean[1] - 3.0) < 0.05
    assert abs(second[1] - mean[1] ** 2 - 0.05) < 0.01


def test_renewal_fixed_point_reduces_to_full_survival_step_at_small_noise():
    from relative_glv.renewal import solve_renewal_fixed_point
    w = np.array([1.0])
    K = np.array([[1.0]])                                   # one class: fully connected limit
    r = solve_renewal_fixed_point(1.76, 1.0, w, K, D=0.01)
    assert r["converged"] and r["stable"]
    assert abs(r["class_mean"][0] - 1.0) < 1e-8             # normalisation E[S] = 1

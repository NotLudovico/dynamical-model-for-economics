import numpy as np
from relative_glv.msb import rescale, size_volatility, spell_volatility, tent_stats


def test_rescale_unit_scale_on_gaussian():
    # for a Gaussian, sqrt(pi/2)*MAD == std, so rescaled data has std ~ 1, mean ~ 0
    g = np.random.default_rng(0).standard_normal(200_000)
    z = rescale(g)
    assert abs(z.mean()) < 0.02
    assert abs(z.std() - 1.0) < 0.02


def test_rescale_drops_nonfinite():
    z = rescale(np.array([1.0, 2.0, np.nan, np.inf, -1.0]))
    assert np.isfinite(z).all()


def test_tent_stats_laplace_is_symmetric_and_fat():
    g = np.random.default_rng(1).laplace(size=200_000)
    s = tent_stats(g)
    assert abs(s["bowley"]) < 0.03        # symmetric
    assert s["exkurt"] > 2.0              # fat (Laplace excess kurtosis = 3)


def test_size_volatility_recovers_a_planted_exponent():
    # plant sigma(S) ~ S^-beta with beta=0.3 and check the estimator recovers it.
    # Each firm fluctuates (stationary) around a fixed base size with log-noise
    # amplitude base^-beta; converting to shares preserves relative sizes, so
    # S_i = N w_i restores the planted sizes and the decline slope is beta.
    rng = np.random.default_rng(2)
    N, T, beta_true = 4000, 80, 0.3
    base = np.logspace(0.5, 3, N)                         # firm sizes over ~2.5 decades
    amp = 0.5 * base ** (-beta_true)                      # planted per-firm log-volatility
    lnS = np.log(base)[:, None] + amp[:, None] * rng.standard_normal((N, T))
    W = np.exp(lnS); W /= W.sum(0, keepdims=True)         # -> shares; S_i = N w_i restores size
    t = np.linspace(0, T - 1, T)
    out = size_volatility(W, t, window=(0, T - 1), dt=1.0)
    assert abs(out["beta"] - beta_true) < 0.08
    assert out["r2"] > 0.9


def test_spell_volatility_splits_spells_and_matches_balanced_panel():
    # firm 0 dips below the floor mid-window -> two spells; the rest never dip, so their
    # spells are the whole window and must agree with size_volatility firm by firm
    rng = np.random.default_rng(3)
    N, T = 4000, 80
    base = np.logspace(0.5, 2, N)                         # smallest share ~3e-5, never below the floor
    lnS = np.log(base)[:, None] + 0.5 * base[:, None] ** -0.3 * rng.standard_normal((N, T))
    lnS[0, 30:40] = np.log(1e-9)                          # firm 0 delisted for 10 steps
    W = np.exp(lnS); W /= W.sum(0, keepdims=True)
    t = np.arange(T, dtype=float)
    sp = spell_volatility(W, t, window=(0, T - 1), dt=1.0)
    sv = size_volatility(W[1:], t, window=(0, T - 1), dt=1.0)
    assert sp["Sbar"].size == N + 1                       # firm 0 contributes two spells
    assert sorted(sp["length"][:2]) == [30, 40] and sp["censored"].all()
    assert (sp["firm"][:2] == 0).all() and (sp["firm"][2:] == np.arange(1, N)).all()
    assert np.allclose(sp["vol"][2:], sv["vol"])          # undipped firms: identical estimator
    assert abs(sp["beta"] - 0.3) < 0.08

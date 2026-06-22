import numpy as np
from relative_glv.model import coupling, integrate
from relative_glv.dmft import solve_fixed_point, sigma_c

N, TMAX, LATE = 800, 140.0, (90.0, 130.0)
SEEDS = (101, 202)


def _sim(mu, sigma):
    """Matched fully-connected sim: return (g_eff, survival, fluctuation), seed-averaged."""
    out = []
    for sd in SEEDS:
        a = coupling(N, mu, sigma, kind="fc", seed=sd)
        r = integrate(a, tmax=TMAX, n_eval=700, lam=0.0, seed=sd, method="LSODA")
        t, W, lnM = r["t"], r["W"], r["lnM"]
        late = t > LATE[0]
        g_eff = float(np.polyfit(t[late], lnM[late], 1)[0])
        win = (t >= LATE[0]) & (t <= LATE[1])
        surv_mask = W[:, win].min(1) > 1e-6
        surv = float(surv_mask.mean())
        if surv_mask.sum() > 20:
            tg = np.arange(LATE[0], LATE[1] + 1e-9, 0.5)
            lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12)))
                            for i in np.where(surv_mask)[0]])
            fluct = float(np.diff(lnS, axis=1).std())
        else:
            fluct = 0.0
        out.append((g_eff, surv, fluct))
    return np.mean(out, axis=0)


def test_relaxed_phase_matches_dmft():
    sc = sigma_c(0.0)
    errs_g, errs_s = [], []
    for sigma in (0.7, 1.0):
        ge, su, _ = _sim(0.5, sigma)
        d = solve_fixed_point(0.5, sigma)
        assert sigma < sc
        errs_g.append(abs(ge - d["gstar"]))
        errs_s.append(abs(su - d["phi"]))
    assert np.mean(errs_g) < 0.12, "relaxed-phase growth rate disagrees with DMFT"
    assert np.mean(errs_s) < 0.10, "relaxed-phase survival disagrees with DMFT"


def test_chaos_turns_on_near_sigma_c():
    _, _, f_lo = _sim(0.5, 0.7)     # below sqrt(2): relaxed
    _, _, f_hi = _sim(0.5, 2.0)     # above sqrt(2): fluctuating
    assert f_lo < 0.01 and f_hi > 0.02, "chaos onset not near sigma_c=sqrt(2)"


def test_mu_is_a_pure_growth_shift():
    g0, s0, _ = _sim(0.0, 1.0)
    g1, s1, _ = _sim(1.0, 1.0)
    assert abs(s1 - s0) < 0.05, "survival should be mu-independent"
    assert abs((g1 - g0) - (-1.0)) < 0.10, "g_eff should shift by -delta_mu"

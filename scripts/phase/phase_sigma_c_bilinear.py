"""Exact HDMFT threshold for a bilinear (uncorrelated) kernel, lam = 0, gamma = 0, any degree law:
    <k^2 D w1(D)> = 0,  M = mu <k^{3/2} w1(D)>,  D_k = A k^{-1/2} + M k^{1/2},  sigma_c^-2 = <k^2 Phi(D)>,
with k = degree / mean degree.  First order in Var k:  sigma_c = sqrt2 [1 - (Var k / 2)(1 - mu^2 / 2 pi)].
Compared with the full HDMFT (60 classes) on the bilinear kernel (where it is a valid probability) and
on the SCM kernel.  mu in the thesis convention (competition < 0).
Run: uv run python scripts/phase/phase_sigma_c_bilinear.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

w1 = lambda d: d * norm.cdf(d) + norm.pdf(d)


def sigma_c_bilinear(mu, kappa):
    k = kappa / kappa.mean()
    m0 = 1 / np.sqrt(2 * np.pi)
    def eqs(x):
        A, M = x; D = A / np.sqrt(k) + M * np.sqrt(k)
        return [np.mean(k ** 2 * D * w1(D)), M - mu * np.mean(k ** 1.5 * w1(D))]
    A, M = fsolve(eqs, [-mu * m0, mu * m0], xtol=1e-12)
    D = A / np.sqrt(k) + M * np.sqrt(k)
    assert np.max(np.abs(eqs([A, M]))) < 1e-9
    return np.mean(k ** 2 * norm.cdf(D)) ** -0.5


def sigma_c_first_order(mu, kappa):
    v = np.var(kappa / kappa.mean())
    return np.sqrt(2) * (1 - v / 2 * (1 - mu ** 2 / (2 * np.pi)))


def hdmft(mu, kappa_deg, kernel):
    from relative_glv.hdmft import bin_degree_distribution, soft_configuration_kernel, heterogeneous_sigma_c
    cl = bin_degree_distribution(kappa_deg, n_bins=60)
    if kernel == "scm":
        K = soft_configuration_kernel(cl["degree_fractions"], cl["weights"], cl["connectance"])["kernel"]
    else:
        v = cl["degree_fractions"]; K = np.outer(v, v) / (cl["weights"] @ v)
        if K.max() > 1: return np.nan
    return heterogeneous_sigma_c(-mu, cl["weights"], K, cl["connectance"], lam=0.0, bracket=(0.3, 3.0))["sigma_c"]


if __name__ == "__main__":
    import relative_glv.model as Mo
    MUS = (0.0, -1.0, -2.0, 1.0)
    for alpha in (6.0, 3.5, 2.5):
        Mo._ALPHA_PL = alpha
        k = np.diff(Mo.coupling(4000, mu=0.0, sigma=1.0, kind="powerlaw", seed=0, mean_degree=100).indptr).astype(float)
        kap = k / k.mean()
        print(f"alpha={alpha}: Var kappa {np.var(kap):.3f}, skew {((kap - 1) ** 3).mean() / np.var(kap) ** 1.5:.2f}, k_max/k_mean {kap.max():.1f}")
        for mu in MUS:
            print(f"   mu={mu:+.1f}  first order {sigma_c_first_order(mu, kap):.3f}  exact bilinear {sigma_c_bilinear(mu, kap):.3f}"
                  f"  HDMFT bilinear {hdmft(mu, k, 'bil'):.3f}  HDMFT SCM {hdmft(mu, k, 'scm'):.3f}")

"""Zero-growth line for a bilinear kernel, lam = 0: exact system
    sigma^2 <k^2 w2(D)> = 1,  M = mu <k^{3/2} w1(D)>,  A = <k^{1/2} w1(D)>,  D_k = A k^{-1/2} + M k^{1/2},
and its first-order shift mu0 = mu0_reg(sigma) + c(sigma) Var k, with c(sigma -> sqrt2) = 5/4 - pi/2.
Checked against the exact solve on the alpha = 6 and 3.5 phase-grid graphs.
Run: uv run python scripts/phase_zero_growth_bilinear.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, fsolve
w0 = norm.cdf; phi = norm.pdf
w1 = lambda d: d * norm.cdf(d) + norm.pdf(d)
w2 = lambda d: (1 + d * d) * norm.cdf(d) + d * norm.pdf(d)

def base(sig):
    D = brentq(lambda d: sig * sig * w2(d) - 1, -20, 60)
    return D, D / w1(D) - 1                        # Delta0, mu0 (regular zero-growth line)

def first_order(sig, v):
    D, mu = base(sig); W0, W1, W2, P = w0(D), w1(D), w2(D), phi(D)
    c1, c2 = (mu - 1) * W1 / 2, (3 - mu) * W1 / 8
    Y = 4 * c1 * W1 + 2 * c2 * W1 + c1 ** 2 * W0 + W2
    dD = -v * Y / (2 * W1)
    X = (1 + mu) * (c2 * W0 + c1 ** 2 * P / 2) + c1 * W0 * (1 + 3 * mu) / 2 + W1 * (3 * mu - 1) / 8
    dmu = (dD * (1 - (1 + mu) * W0) - v * X) / W1
    return mu, mu + dmu, dmu / v

def exact(sig, kap):
    k = kap / kap.mean(); D0, mu0 = base(sig)
    def eqs(x):
        A, M, mu = x; D = A / np.sqrt(k) + M * np.sqrt(k)
        return [sig ** 2 * np.mean(k ** 2 * w2(D)) - 1, M - mu * np.mean(k ** 1.5 * w1(D)), A - np.mean(np.sqrt(k) * w1(D))]
    W1 = w1(D0); x = fsolve(eqs, [W1, mu0 * W1, mu0], xtol=1e-12)
    assert np.max(np.abs(eqs(x))) < 1e-9
    return x[2]

if __name__ == "__main__":
    import relative_glv.model as Mo
    for alpha in (6.0, 3.5):
        Mo._ALPHA_PL = alpha
        kd = np.diff(Mo.coupling(4000, mu=0.0, sigma=1.0, kind="powerlaw", seed=0, mean_degree=100).indptr).astype(float)
        kap = kd / kd.mean(); v = kap.var()
        print(f"alpha={alpha}, Var kappa={v:.3f}")
        for s in (0.6, 0.8, 1.0, 1.2, 1.35):
            mu0, mu1, coef = first_order(s, v)
            print(f"  sigma={s:.2f}: regular {mu0:+.4f}  first order {mu1:+.4f} (dmu/dVar = {coef:+.4f})  exact bilinear {exact(s, kap):+.4f}")

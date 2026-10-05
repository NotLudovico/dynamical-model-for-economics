"""Exact 1D-diffusion results at constant Phi for dS = S(Phi-S)dt + sqrt(DS)dW (scale function s'(y) = exp(-(2/D)(Phi y - y^2/2))):
  P(reach h before 0 | x0) = int_0^x0 s' / int_0^h s'
  T_ext(x0) = 2 int_0^x0 s'(y) int_y^inf dz / (D z s'(z)) dy      (mean time to extinction)
Checked against direct simulation."""
import numpy as np
from scipy.integrate import quad
D, dt = 0.1, 0.002
rng = np.random.default_rng(1)
sp = lambda y, P: np.exp(-(2 / D) * (P * y - y * y / 2))
def p_hit(x0, h, P): return quad(sp, 0, x0, args=(P,))[0] / quad(sp, 0, h, args=(P,))[0]
def T_ext(x0, P):
    inner = lambda y: quad(lambda z: 1 / (D * z * sp(z, P)), y, P + 10 * np.sqrt(D) + 1, limit=200)[0]
    return 2 * quad(lambda y: sp(y, P) * inner(y), 0, x0, limit=200)[0]

print("establishment = P(reach Phi before 0 | S0 = D):")
for P in (0.2, 0.3, 0.6, 1.0):
    n = 20000; S = np.full(n, D); done = np.zeros(n, bool); hit = np.zeros(n, bool)
    while not done.all():
        S = S + S * (P - S) * dt + np.sqrt(D * np.maximum(S, 0) * dt) * rng.standard_normal(n)
        hit |= (~done) & (S >= P); done |= (S >= P) | (S <= 0)
    print(f"   Phi {P:.1f}: sim {hit.mean():.3f}   exact {p_hit(D, P, P):.3f}   (linear Feller 1-e^-2Phi {1-np.exp(-2*P):.3f})")

print("mean extinction time:")
for P, x0 in ((0.0, D), (-0.5, D), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)):
    n = 4000 if P > 0 else 20000; S = np.full(n, x0); t = np.zeros(n); alive = np.ones(n, bool); k = 0
    while alive.any() and k * dt < 2000:
        S[alive] = S[alive] + S[alive] * (P - S[alive]) * dt + np.sqrt(D * np.maximum(S[alive], 0) * dt) * rng.standard_normal(alive.sum())
        t[alive] += dt; alive &= S > 0; k += 1
    print(f"   Phi {P:+.1f} from S0 {x0:.2f}: sim {t.mean():7.2f} (censored {alive.mean():.3f})   exact {T_ext(x0, P):7.2f}")

"""Window convergence of the spell estimator in the single-firm surrogate (alpha=6 fitness), against the
full model's long runs (0.461, 0.422, 0.391, 0.374 at T_obs = 240, 480, 960, 1920).
Run: uv run python scripts/lifecycle/window_surrogate.py"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from joblib import Parallel, delayed
from surrogate import run, fit_tau, NE, DT, START
from beta_spells import cq_one
from relative_glv.msb import spell_volatility

TOBS, FULL = (240.0, 480.0, 960.0, 1920.0), (0.461, 0.422, 0.391, 0.374)
z = np.load(f"{DATA}/fit_a6.0_s1.75_l1e-03.npz")
m_all, f_all, tau = z["m"], z["f"], fit_tau(z["acf"])

def one(e):
    S = run(m_all[e * NE:(e + 1) * NE], f_all[e * NE:(e + 1) * NE], tau, 1e-3, 100 + e, t_end=START + max(TOBS))
    t = START + DT * np.arange(S.shape[1])
    out = []
    for T in TOBS:
        sp = spell_volatility(S / NE, t, window=(START, START + T), dt=DT, floor=8e-3 / NE)
        out.append((sp["beta"], sp["censored"].mean()))
    return out

R = np.array(Parallel(n_jobs=8)(delayed(one)(e) for e in range(8)))      # (econ, T, 2)
for i, T in enumerate(TOBS):
    print(f"T_obs={T:5.0f}: surrogate beta {np.median(R[:, i, 0]):.3f} (cut-off spells {np.median(R[:, i, 1]):.2f}) | "
          f"full model {FULL[i]:.3f}")

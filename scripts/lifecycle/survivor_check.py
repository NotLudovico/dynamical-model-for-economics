"""Thesis (survivor) estimator in the single-firm surrogate vs full model (window [180, 240])."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from joblib import Parallel, delayed
from surrogate import run, fit_tau, NE, DT, START
from beta_spells import cq_one
from relative_glv.msb import size_volatility

full = {f"{DATA}/fit_a6.0_s1.75_l1e-03.npz": (0.894, (1.96, 2.87, 3.70)), f"{DATA}/fit_a2.5_s1.75_l1e-03.npz": (0.414, (1.87, 2.58, 3.15))}
for f in full:
    z = np.load(f); m_all, f_all, tau = z["m"], z["f"], fit_tau(z["acf"])
    def one(e):
        S = run(m_all[e * NE:(e + 1) * NE], f_all[e * NE:(e + 1) * NE], tau, 1e-3, 100 + e, t_end=START + 60.0)
        t = START + DT * np.arange(S.shape[1])
        sv = size_volatility(S / NE, t, window=(START, START + 60.0), dt=DT)
        return sv["beta"], cq_one(sv["Sbar"], sv["vol"]), sv["Sbar"].size / NE
    R = Parallel(n_jobs=8)(delayed(one)(e) for e in range(8))
    Z = np.array([z for _, z, _ in R if z is not None]); zr = Z.mean(0)[1:] / Z.mean(0)[0]
    print(f"{f}: survivor estimator  full model beta {full[f][0]:.3f} ratios {full[f][1]} | surrogate beta "
          f"{np.median([b for b, _, _ in R]):.3f} ratios " + " ".join(f"{x:.2f}" for x in zr)
          + f" | survivors {np.median([s for _, _, s in R]):.2f}")

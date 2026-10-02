"""Fraction of time listed vs degree, against the (m, f) law: P(listed | kappa) ~ Phi((1 - gbar - mu kappa Bbar) /
(sigma sqrt(kappa Q_0))), order parameters from the simulation (mf_law.py cavity values). No fit.
Run: uv run python scripts/lifecycle/listed_by_degree.py"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from scipy.stats import norm

MU, N = 1.76, 8000
SIM = {(6.0, 1.75): (0.176, 1.000, 4.513), (2.5, 1.75): (0.332, 0.927, 5.474), (3.5, 1.9): (0.936, 1.032, 6.859)}
for (alpha, sigma), (g, B, Q0) in SIM.items():
    d = np.load(f"{DATA}/fit_a{alpha}_s{sigma}_l1e-03.npz")
    k, frac = d["k"].astype(float), d["frac_listed"]
    kap = k / np.repeat([k[e * N:(e + 1) * N].mean() for e in range(8)], N)
    edges = np.quantile(kap, np.linspace(0, 1, 9))
    edges[-1] = np.quantile(kap, 0.999)
    print(f"alpha={alpha} sigma={sigma}:  kappa  listed(sim)  law")
    for lo, hi in zip(edges[:-1], np.append(edges[1:-1], kap.max() + 1)):
        s = (kap >= lo) & (kap < hi)
        kc = np.median(kap[s])
        law = norm.cdf((1 - g - MU * kc * B) / (sigma * np.sqrt(kc * Q0)))
        print(f"    {kc:6.2f}   {frac[s].mean():.3f}   {law:.3f}")

"""Spell-panel table rows of the theory note with the frozen estimator (msb.upper_zeta).
Full model from data/headline_spells (scripts/headline_spells.py); single-firm surrogate on the saved
fitness files: measured (m_i, f_i), the fitness law with simulated and with HDMFT order parameters,
and the alpha = 2.5 amplitude ablations.  Run: uv run python scripts/lifecycle/recompute_tables.py"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.dirname(os.path.abspath(__file__))]
import numpy as np
from relative_glv.msb import upper_zeta
from surrogate import surrogate

L = os.path.join(ROOT, "data", "lifecycle"); H = os.path.join(ROOT, "data", "headline_spells")
fmt = lambda b, z: f"{b:.3f}; " + ", ".join(f"{x:.2f}" for x in z)


def full(tag):
    d = np.load(os.path.join(H, tag + ".npz")); e = d["economy"]
    Z = np.array([upper_zeta(d["Sbar"][e == i], d["vol"][e == i]) for i in np.unique(e)])
    return np.median(Z[:, 0]), Z.mean(0)[1:] / Z.mean(0)[0]


for a, s in (("6.0", "1.75"), ("3.5", "1.9"), ("2.5", "1.75")):
    row = [fmt(*full(f"a{a}_s{s}_l1e-03"))]
    for name in (f"fit_a{a}_s{s}", f"law_measured_a{a}_s{s}", f"law_cavity_a{a}_s{s}", f"law_hdmft_a{a}_s{s}"):
        b, zr, _ = surrogate(os.path.join(L, f"{name}_l1e-03.npz"))
        row.append(fmt(b, zr))
    print(f"alpha={a} sigma={s} | full {row[0]} | measured {row[1]} (law file {row[2]}) | law sim {row[3]} | law HDMFT {row[4]}", flush=True)
for name in ("measured", "shuffled", "homogeneous"):
    b, zr, _ = surrogate(os.path.join(L, f"ablation_{name}_l1e-03.npz"))
    print(f"ablation alpha=2.5 {name:12s} {fmt(b, zr)}", flush=True)

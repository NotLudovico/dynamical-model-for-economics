"""Amplitude ablations at alpha=2.5: shuffle f_i across firms (breaks the m-f / degree link) and make f
homogeneous. Run: uv run python scripts/lifecycle/ablation.py"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from surrogate import surrogate

src = f"{DATA}/fit_a2.5_s1.75_l1e-03.npz"
d = dict(np.load(src))
variants = {"measured": d["f"], "shuffled": np.random.default_rng(0).permutation(d["f"]),
            "homogeneous": np.full_like(d["f"], d["f"].mean())}
for name, f in variants.items():
    path = f"{DATA}/ablation_{name}_l1e-03.npz"
    np.savez(path, **{**d, "f": f})
    b, zr, _ = surrogate(path)
    print(f"{name:12s} beta {b:.3f} ratios " + " ".join(f"{x:.2f}" for x in zr))

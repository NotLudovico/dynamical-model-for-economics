"""Null test for the tail index b of sigma/sigmabar (MSB: b = 4.6).

The demographic model gives b = 4.40-4.44 at every parameter point for lives with >= 2 growth rates.
Here every life keeps its size Sbar and its number of growth rates n from the model (retuned point,
sigma 1.70, D 0.1, N 16000, 4 economies), but its growth rates are iid draws with a true volatility
that depends on size only (the model's binned mean): no firm-to-firm heterogeneity at all.  The
volatility is then estimated exactly as in the model and in MSB (sqrt(pi/2) MAD around the own mean)
and b is measured the same way (Hill index of the top 5% of v / E[v | bin], 25 bins).  If the null
gives b ~ 4.4, the model's value is an estimator effect of short lives, not a property of firms.
Run: uv run python scripts/demographic/b_null.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
from robustness import path
from relative_glv.msb import hill

LABEL, SEEDS = "rt_s1.7_D0.1", range(4)


def ratio_b(Sb, v, p=0.05, nb=25):
    P = np.array_split(np.argsort(Sb), nb)
    r = np.empty(v.size)
    for q in P:
        r[q] = v[q] / v[q].mean()
    return hill(r, p)


def synth(n, scale, draw, rng):
    out = np.empty(n.size)
    for m in np.unique(n):
        idx = np.flatnonzero(n == m)
        g = draw(rng, (idx.size, m)) * scale[idx, None]
        out[idx] = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    return out


def main():
    ds = [np.load(path(LABEL, s)) for s in SEEDS]
    Sb = np.concatenate([d["Sbar"] for d in ds]).astype(float)
    v = np.concatenate([d["v"] for d in ds]).astype(float)
    n = np.concatenate([d["n"] for d in ds])
    ok = v > 0
    Sb, v, n = Sb[ok], v[ok], n[ok]
    P = np.array_split(np.argsort(Sb), 25)
    scale = np.empty(v.size)
    for q in P:
        scale[q] = v[q].mean()
    rng = np.random.default_rng(0)
    draws = {
        "Gaussian": lambda r, s: r.standard_normal(s),
        "Laplace": lambda r, s: r.laplace(size=s),
        "Student t, 3 d.o.f.": lambda r, s: r.standard_t(3, size=s),
    }
    print(f"{LABEL}: {v.size} lives;  share with n = 2, 3, 4 growth rates: "
          f"{np.mean(n == 2):.2f} {np.mean(n == 3):.2f} {np.mean(n == 4):.2f}")
    print(f"{'':28} {'b (n>=2)':>9} {'b (n>=5)':>9} {'b (n>=10)':>10} {'b (n>=20)':>10}")
    row = lambda name, vv: print(f"{name:28} " + " ".join(f"{ratio_b(Sb[n >= m], vv[n >= m]):9.2f}" for m in (2, 5, 10, 20)))
    row("model", v)
    for name, dr in draws.items():
        row(f"null: iid {name}", synth(n, scale, dr, rng))
    print("Compustat: b = 4.6 (robust to >= 20 growth rates)")


if __name__ == "__main__":
    main()

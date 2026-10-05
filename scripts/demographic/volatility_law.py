"""Step 3b: filtered prediction of per-life growth variance in the full model (facts.npz, seeds 0, 1).
    var g  ~  V_det + B_filt
V_det : variance of the growth of the NOISE-FREE one-firm ODE dS = S(Phi - S) dt driven by the life's measured
        Phi(t) from its actual starting size (fitness forcing, filtered and nonlinear, transients included)
B_filt: demographic part of an OU in ln S with local relaxation rate S:  mean_k (D/S_k^2)(1 - exp(-S_k dt_g))
Compared with the measured var g by life-mean size."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "demographic")
import numpy as np
from relative_glv import model
D, DTg, N, dt = 0.1, 0.5, 8000, 0.01
per = int(round(DTg / dt))
model._ALPHA_PL = 2.5
d = np.load(os.path.join(DATA, "facts.npz"))
rows = []
for seed in (0, 1):
    S = d["S"][seed].astype(float); L = d["L"][seed]
    a = model.coupling(N, 1.76, 1.75, kind="powerlaw", seed=seed, mean_degree=200)
    f = 1.0 - S - (a @ S); Phi = 1.0 - (a @ S) - (S * f).sum(0) / N
    T = S.shape[1]
    lives = [(i, seg) for i in range(N) for seg in np.split(np.arange(T), np.flatnonzero(np.diff(L[i])) + 1) if seg.size >= 21]
    # vectorised noise-free integration: pad lives to a common length
    n = max(seg.size for _, seg in lives)
    ph = np.full((len(lives), n), np.nan); xs = np.full((len(lives), n), np.nan)
    for k, (i, seg) in enumerate(lives):
        ph[k, :seg.size] = Phi[i, seg]; xs[k, :seg.size] = S[i, seg]
    lnS = np.log(xs[:, 0]); det = [lnS.copy()]
    for k in range(n - 1):
        p0, p1 = ph[:, k], ph[:, k + 1]
        for j in range(per):
            pj = p0 + (p1 - p0) * j / per
            lnS = lnS + (pj - np.exp(lnS)) * dt
        det.append(lnS.copy())
    det = np.array(det).T
    for k, (i, seg) in enumerate(lives):
        m = seg.size
        x = xs[k, :m]; g = np.diff(np.log(x)); gd = np.diff(det[k, :m])
        Bf = np.mean(D / x[:-1] ** 2 * (1 - np.exp(-x[:-1] * DTg)))
        rows.append((x.mean(), g.var(), gd.var(), Bf))
R = np.array(rows)
P = np.array_split(np.argsort(R[:, 0]), 10)
print(f"lives >= 20: {len(R)}")
print(f"{'Sbar':>6} | {'sd g':>6} | {'sqrt(Vdet+Bf)':>13} | ratio | {'sqrt Vdet':>9} {'sqrt Bf':>7}")
for p in P:
    m = R[p].mean(0)
    pr = np.sqrt(m[2] + m[3])
    print(f"{m[0]:6.3f} | {np.sqrt(m[1]):6.3f} | {pr:13.3f} | {pr / np.sqrt(m[1]):5.2f} | {np.sqrt(m[2]):9.3f} {np.sqrt(m[3]):7.3f}")
x = np.log([R[p, 0].mean() for p in P])
for name, col in (("measured", 1), ("predicted", None)):
    y = 0.5 * np.log([R[p, 1].mean() if col else R[p, 2].mean() + R[p, 3].mean() for p in P])
    print(f"slope of sd vs Sbar, {name}: {-np.polyfit(x, y, 1)[0]:.3f}")

"""The 'divergent' cells of the phase grid are a numerical failure, not a phase.  Integrate a failed
seed of the weak-competition, strong-disorder corner in chunks of 20 time units, clipping the state to
w >= 0 between chunks.  It integrates in ~1 min and stays bounded: g ~ 6-60, largest firm S ~ 100-200,
~150-250 firms above the mean.  The grid runs failed because the immigration floor w ~ lam/(N|f|) ~ 5e-9
lies far below atol = 1e-7.  Run: uv run python scripts/phase_divergent_check.py"""
import time, numpy as np
from scipy.integrate import solve_ivp
from relative_glv.model import coupling
d = np.load("data/phase_meandeg_fine_24x20_10s.npz"); sig, mus, raw = d["sigmas"], d["mus"], d["raw"]
i = len(sig) - 1; j = int(np.argmin(np.abs(mus - 0.13))); seed = int(np.where(~np.isfinite(raw[i, j, :, 0]))[0][0])
N, lam = 4000, 1e-3
a = coupling(N, mus[j], sig[i], kind="powerlaw", seed=seed, mean_degree=100)
k = np.diff(a.indptr)
rng = np.random.default_rng(seed); w = rng.uniform(0.5, 1.5, N); w /= w.sum()
def rhs(t, s):
    x = np.clip(s, 0, None); x = x / x.sum(); f = 1 - N * x - N * (a @ x)
    return x * (f - x @ f) + lam * (1 / N - x)
print(f"cell sigma={sig[i]:.3f} mu={-mus[j]:+.2f} seed={seed}")
t, T0 = 0.0, time.time()
while t < 400 and time.time() - T0 < 240:
    c = time.time()
    r = solve_ivp(rhs, (t, t + 20), w, method="RK45", rtol=1e-4, atol=1e-7)
    w = np.clip(r.y[:, -1], 0, None); w /= w.sum(); t += 20
    f = 1 - N * w - N * (a @ w); g = w @ f
    top = np.argsort(w)[-3:][::-1]
    print(f" t={t:5.0f}  {time.time()-c:6.1f}s  steps {r.t.size:6d}  g={g:7.2f}  max S={N*w.max():8.1f} (deg {k[top[0]]})  top3 S {np.round(N*w[top],1)}  firms S>1: {(N*w>1).sum()}", flush=True)

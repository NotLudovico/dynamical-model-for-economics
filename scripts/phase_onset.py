"""Onset of fluctuations across the HDMFT stability line: amplitude vs sigma at fixed mu.
Power law 2.5, N=4000, C=100, lam=1e-3 (the phase-grid ensemble), long runs, late window.
Amplitude A = median over firms listed throughout (S > 0.1) of the time std of ln S;
churn = std of dt=0.5 growth rates of firms above S = 1e-2 throughout (the phase-grid statistic).
Run: uv run python scripts/phase_onset.py   -> data/phase_onset.npz
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
from relative_glv.model import coupling

N, C, LAM, TMAX, WIN = 4000, 100, 1e-3, 800.0, (600.0, 800.0)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "phase_onset.npz")
SC = {1.97: 1.515, 1.05: 1.251, 0.66: 1.108}          # HDMFT sigma_c, lam=1e-3, 60 classes (code-convention mu)


def one(mu, sigma, seed):
    a = coupling(N, mu, sigma, kind="powerlaw", seed=seed, mean_degree=C)
    rng = np.random.default_rng(seed); w0 = rng.uniform(0.5, 1.5, N); w0 /= w0.sum()
    def rhs(t, s):
        w = np.clip(s, 0.0, None); w = w / w.sum()
        f = 1.0 - N * w - N * (a @ w)
        return w * (f - w @ f) + LAM * (1.0 / N - w)
    r = solve_ivp(rhs, (0.0, TMAX), w0, method="RK45", t_eval=np.arange(WIN[0], WIN[1] + 1e-9, 0.5),
                  rtol=1e-4, atol=1e-7)
    if not r.success:
        return mu, sigma, seed, np.nan, np.nan
    lnS = np.log(np.clip(r.y * N, 1e-2, None))
    big = (lnS > np.log(0.1)).all(1); live = (lnS > np.log(1e-2)).all(1)
    amp = np.median(lnS[big].std(1)) if big.any() else np.nan
    churn = np.diff(lnS[live], axis=1).std() if live.any() else np.nan
    return mu, sigma, seed, amp, churn


if __name__ == "__main__":
    jobs = [(mu, round(sc + d, 4), s) for mu, sc in SC.items() for d in np.arange(-0.05, 0.301, 0.025) for s in (0, 1)]
    R = np.array(Parallel(n_jobs=10, verbose=2)(delayed(one)(*j) for j in jobs))
    np.savez(OUT, R=R, sc=np.array(list(SC.items())))
    for mu, sc in SC.items():
        print(f"mu={-mu:.2f}  HDMFT sigma_c={sc}")
        for s in np.unique(R[R[:, 0] == mu, 1]):
            x = R[(R[:, 0] == mu) & (R[:, 1] == s)]
            print(f"   sigma {s:.3f} (sc{s - sc:+.3f})  amplitude {np.nanmean(x[:, 3]):.4f}  churn {np.nanmean(x[:, 4]):.4f}")

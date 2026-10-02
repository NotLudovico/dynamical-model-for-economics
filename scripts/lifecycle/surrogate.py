"""Single-firm surrogate: dlnS/dt = Phi - S + lam (1/S - 1), Phi_i(t) = m_i + f_i eta_i(t), eta a unit-variance
smooth Gaussian process with ACF (1 + L/tau) e^{-L/tau} (OU cascade). (m_i, f_i) are the measured
per-firm values of the full model; no network. Same spell estimator. Optional overrides via kwargs."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
DATA = os.path.join(ROOT, "data", "lifecycle")
import numpy as np
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from beta_spells import cq_one, START, DT
from relative_glv.msb import spell_volatility

T, NE, DTI = 480.0, 8000, 0.02

def fit_tau(acf):
    L = DT * np.arange(acf.size)
    return curve_fit(lambda L, tau: (1 + L / tau) * np.exp(-L / tau), L, acf, p0=[3.0])[0][0]

def run(m, f, tau, lam, seed, smooth=True, t_end=START + T):
    rng = np.random.default_rng(seed)
    n = m.size
    y = rng.standard_normal(n)
    eta = y.copy()
    S = np.maximum(m, 0.1)
    rec, every = [], int(round(DT / DTI))
    nsteps = int(round(t_end / DTI))
    a = np.exp(-DTI / tau); b = np.sqrt(1 - a * a)
    for s in range(nsteps + 1):
        if s * DTI >= START - 1e-9 and s % every == 0:
            rec.append(S.copy())
        if smooth:
            eta += (np.sqrt(2) * y - eta) * DTI / tau        # second filter; sqrt(2) keeps unit variance
        y = a * y + b * rng.standard_normal(n)
        Phi = m + f * (eta if smooth else y)
        S = S * np.exp((Phi - S) * DTI) + lam * (1 - S) * DTI
    return np.array(rec).T                                   # (n, n_t)

def analyse(S, lam, min_growth=2):
    t = START + DT * np.arange(S.shape[1])
    sp = spell_volatility(S / NE, t, window=(START, t[-1]), dt=DT, floor=8 * lam / NE, min_growth=min_growth)
    return sp["beta"], cq_one(sp["Sbar"], sp["vol"]), sp

def surrogate(fitfile, lam=None, tau=None, f_scale=1.0, m_shift=0.0, smooth=True, min_growth=2, seeds=8):
    d = np.load(fitfile)
    lam = lam if lam is not None else float(fitfile.rsplit("_l", 1)[1][:-4])
    tau = tau if tau is not None else fit_tau(d["acf"])
    m, f = d["m"] + m_shift, d["f"] * f_scale
    def one(e):
        sl = slice(e * NE, (e + 1) * NE)
        S = run(m[sl], f[sl], tau, lam, seed=100 + e, smooth=smooth)
        b, z, _ = analyse(S, lam, min_growth)
        return b, z
    R = Parallel(n_jobs=8)(delayed(one)(e) for e in range(seeds))
    Z = np.array([z for _, z in R if z is not None])
    return np.median([b for b, _ in R]), Z.mean(0)[1:] / Z.mean(0)[0], tau

if __name__ == "__main__":
    fitfile = sys.argv[1]
    d = np.load(fitfile)
    b, zr, tau = surrogate(fitfile)
    full = d["Z"].mean(0)
    print(f"{fitfile}: tau fit {tau:.2f}")
    print(f"  full model : beta {np.median(d['beta']):.3f} ratios " + " ".join(f"{x:.2f}" for x in full[1:] / full[0]))
    print(f"  surrogate  : beta {b:.3f} ratios " + " ".join(f"{x:.2f}" for x in zr))

"""Degree-resolved law of the field Phi_i(t) = 1 - (aS)_i - fbar at the reference point (reference.py).

Per node over the window: time mean Phibar_i and time standard deviation f_i.  By degree class
(relative degree kappa = k / <k>): mean and spread of Phibar, median f, and power-law fits
f ~ kappa^h (theory h = 1/2: the field is a sum of ~k couplings of size 1/sqrt(C)) and
Var(Phibar | kappa) - its floor ~ kappa.  Also the Gaussianity of the standardised Phibar, the
field's autocorrelation time, and life-mean size against degree.
Run: uv run python scripts/demographic/field_law.py  -> data/demographic/field_law.npz
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
from scipy import stats
import reference as ref

OUT = os.path.join(ROOT, "data", "demographic", "field_law.npz")

if __name__ == "__main__":
    K, PB, F, TAU, SM = [], [], [], [], []
    for seed in ref.SEEDS:
        S, L, deg, _, Phi = ref.economy(seed)
        K.append(deg / deg.mean()); PB.append(Phi.mean(1)); F.append(Phi.std(1)); SM.append(S.mean(1))
        x = Phi - Phi.mean(1, keepdims=True)
        ac = [np.mean(np.sum(x[:, l:] * x[:, :x.shape[1] - l], 1) / np.sum(x * x, 1)) for l in range(0, 21)]
        TAU.append(ac)
    kap, pb, f, sm = map(np.concatenate, (K, PB, F, SM)); ac = np.mean(TAU, 0)
    edges = np.quantile(kap, np.linspace(0, 1, 21))
    b = np.clip(np.digitize(kap, edges[1:-1]), 0, 19)
    kc = np.array([np.median(kap[b == j]) for j in range(20)])
    mpb = np.array([pb[b == j].mean() for j in range(20)]); vpb = np.array([pb[b == j].var() for j in range(20)])
    mf = np.array([np.median(f[b == j]) for j in range(20)]); msm = np.array([sm[b == j].mean() for j in range(20)])
    h = np.polyfit(np.log(kc), np.log(mf), 1)[0]
    hv = np.polyfit(np.log(kc), np.log(vpb), 1)[0]
    print(f"{kap.size} nodes, relative degree {kap.min():.2f}..{kap.max():.1f} (CV {kap.std():.2f})")
    print(f"{'kappa':>7} {'<Phibar>':>9} {'sd Phibar':>9} {'median f':>9} {'<S>':>6}")
    for j in range(0, 20, 2):
        print(f"{kc[j]:7.2f} {mpb[j]:9.3f} {np.sqrt(vpb[j]):9.3f} {mf[j]:9.3f} {msm[j]:6.2f}")
    print(f"amplitude f ~ kappa^{h:.2f} (theory 0.5);  Var(Phibar | kappa) ~ kappa^{hv:.2f} (theory 1)")
    z = np.concatenate([(pb[b == j] - pb[b == j].mean()) / pb[b == j].std() for j in range(20)])
    print(f"standardised Phibar within classes: skewness {stats.skew(z):+.2f}, excess kurtosis {stats.kurtosis(z):+.2f}")
    tau = 0.5 * np.argmax(ac < np.exp(-1)) if (ac < np.exp(-1)).any() else np.nan
    print(f"field autocorrelation at lags 0.5..3: {' '.join(f'{x:.2f}' for x in ac[1:7])};  1/e time {tau:.1f}")
    np.savez(OUT, kc=kc, mpb=mpb, vpb=vpb, mf=mf, msm=msm, h=h, hv=hv, ac=ac)

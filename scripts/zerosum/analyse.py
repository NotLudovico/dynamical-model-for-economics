# Size density, growth tents, volatility vs current size and the MSB life panel for a
# scripts/zerosum/simulate.py run.  Usage: analyse.py run.npz
import sys, numpy as np
d = np.load(sys.argv[1]); U, F, dt = d["U"], d["firm"], float(d["dt"])
U, F = U[len(U) // 3:], F[len(F) // 3:]          # drop transient
lag = int(round(1 / dt))                           # growth over Delta t = 1
x = np.exp(U)
xs = np.sort(x.ravel())[::-1]; k = len(xs) // 100
print(f"sizes: log10 5/50/95% {np.percentile(np.log10(xs), [5, 50, 95]).round(2)}, "
      f"Hill top1% {1/np.mean(np.log(xs[:k]/xs[k])):.2f}, top1% share {xs[:k].sum()/xs.sum():.2f}")
h, e = np.histogram(np.log10(xs), 30)
print("log10 density:", " ".join(f"{c:.2f}:{n}" for c, n in zip((e[1:]+e[:-1])/2, h)))
# growth rates within one firm
same = F[lag:] == F[:-lag]
g = (U[lag:] - U[:-lag])[same]; s0 = U[:-lag][same]
gc = g - g.mean()
print(f"growth: skew {np.mean(gc**3)/gc.std()**3:.2f} kurt {np.mean(gc**4)/gc.var()**2:.2f}")
qs = np.quantile(s0, np.linspace(0, 1, 11)); idx = np.digitize(s0, qs[1:-1])
print("current size  log10x / sd g / mean g:")
for j in range(10):
    m = idx == j; print(f"   {s0[m].mean()/np.log(10):6.2f} {g[m].std():.3f} {g[m].mean():+.3f}")
# per-firm lives (MSB protocol: vol = sd of own growth, Sbar = mean size over life)
T, N = U.shape; rec = []
for i in range(N):
    f = F[:, i]; cut = np.flatnonzero(np.diff(f)) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, T]):
        uu = U[a:b:lag, i]
        if len(uu) >= 11: rec.append((np.exp(uu).mean(), np.diff(uu).std(), len(uu) - 1))
rec = np.array(rec); print(f"lives with >=10 growth rates: {len(rec)}, median length {np.median(rec[:,2]):.0f}")
order = np.argsort(rec[:, 0]); bins = np.array_split(order, 25)
lS = np.log([rec[b, 0].mean() for b in bins])
zeta = []
for q in (1, 2, 3, 4):
    lv = np.log([np.mean(rec[b, 1] ** q) for b in bins]); zeta.append(-np.polyfit(lS, lv, 1)[0])
print("MSB beta", round(zeta[0], 3), "ratios", np.round(np.array(zeta[1:]) / zeta[0], 2),
      "range dec", round((lS[-1] - lS[0]) / np.log(10), 2))
print("bins log10 Sbar / vol:", " ".join(f"{a/np.log(10):.2f}:{rec[b,1].mean():.2f}" for a, b in zip(lS, bins)))

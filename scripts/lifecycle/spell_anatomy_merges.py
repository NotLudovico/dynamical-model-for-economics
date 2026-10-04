"""Spell volatility = humps merged by non-killing dips (step 2 diagnostics, alpha=6 surrogate fitness).
TV = observed range + ~0.35-0.4 * 2 ell per merge (K-1 merges); residual sits in the top size quartile.
Run: uv run python scripts/lifecycle/spell_anatomy_merges.py"""
import sys; sys.path[:0] = ["scripts", "scripts/lifecycle"]
import numpy as np
from surrogate import fit_tau, DT
from kill_rule import simulate
from anatomy import branch
d = np.load("data/lifecycle/fit_a6.0_s1.75_l1e-03.npz"); tau = fit_tau(d["acf"]); lam = 1e-3; thr = 8 * lam
rng = np.random.default_rng(0); idx = rng.choice(d["m"].size, 3000, replace=False)
P, X = simulate(d["m"][idx].astype(float), d["f"][idx].astype(float), tau, lam, 5, t_end=530.0)
step = int(round(DT / 0.02))
out = []
for p, xs, fi in zip(P, X, d['f'][idx]):
    x = np.log(xs[::step]); ph = p[::step]
    on = np.concatenate(([0], (x > np.log(thr)).astype(np.int8), [0])); dd = np.diff(on)
    for a, b in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
        n = b - a - 1
        if n < 2 or a == 0 or b >= x.size: continue
        seg, fs = x[a:b], ph[a:b]
        g = np.diff(seg); TV = np.abs(g).sum(); ell = np.log(np.exp(seg.max()) / thr)
        K = np.sum((fs[:-1] >= 0) & (fs[1:] < 0))            # downcrossings inside the spell (survived dips)
        # drops of ln S between successive local maxima/minima larger than 0.3 (merges)
        out.append((np.exp(seg).mean(), n, TV, ell, K, seg.max()-seg[0]+seg.max()-seg[-1], fi, (1/np.exp(seg)).mean(), (1/np.exp(seg[seg>np.log(1.0)])).sum() if (seg>0).any() else 0.0))
Sb, n, TV, ell, K, rng2, ff, inv, invbig = np.array(out).T
dec = np.concatenate(branch(Sb, np.sqrt(np.pi/2) * TV / n))
print("complete decline-branch spells", dec.size)
print(f"{'K':>3} {'N':>6} {'TV/2ell':>8} {'(TV-2ell)/(2 ell K)':>20} {'L/tau':>6}")
for k in range(0, 6):
    s = dec[K[dec] == k]
    if s.size < 20: continue
    r = TV[s] / (2 * ell[s])
    print(f"{k:3d} {s.size:6d} {np.median(r):8.3f} {np.median((TV[s]-2*ell[s])/(2*ell[s]*max(k,1))):20.3f} {np.median(n[s]*DT/tau):6.1f}")
from anatomy import zq
v = np.sqrt(np.pi / 2) * TV / n
Pb = branch(Sb, v)
ellS = np.zeros_like(Sb)          # ell from Sbar via the parabolic hump: ell - 0.5 ln(4 ell/pi) = ln(Sbar/thr)
from scipy.optimize import brentq
for i, s in enumerate(Sb):
    r = max(np.log(s / thr), 0.727); ellS[i] = brentq(lambda l: l - 0.5 * np.log(4 * l / np.pi) - r, 0.5, 60)
for dlt in (0.0, 0.35):
    for name, L_ in (("ell measured", ell), ("ell from Sbar", ellS)):
        law = np.sqrt(np.pi / 2) * 2 * L_ * (1 + dlt * (K - 1)) / n
        z = zq(Sb, law, Pb)
        print(f"delta={dlt:.2f} {name:14s} zeta_q {' '.join(f'{a:.3f}' for a in z)}  ratios {' '.join(f'{a:.2f}' for a in z[1:]/z[0])}")
z = zq(Sb, v, Pb); print(f"measured v (complete)        zeta_q {' '.join(f'{a:.3f}' for a in z)}  ratios {' '.join(f'{a:.2f}' for a in z[1:]/z[0])}")
print("median |log(law/v)| with delta=.35, ell from Sbar:", np.median(np.abs(np.log(np.sqrt(np.pi/2)*2*ellS*(1+.35*(K-1))/n/v)[np.concatenate(Pb)])))

for dlt in (0.35,):
    law = np.sqrt(np.pi / 2) * (rng2 + 2 * dlt * ell * (K - 1)) / n
    z = zq(Sb, law, Pb)
    print(f"observed range + merges   zeta_q {' '.join(f'{a:.3f}' for a in z)}  ratios {' '.join(f'{a:.2f}' for a in z[1:]/z[0])}")
    print("  median |log(law/v)|:", np.median(np.abs(np.log(law/v)[np.concatenate(Pb)])))
print("median overshoot 2ell - range:", np.median((2*ell-rng2)[np.concatenate(Pb)]), " per spell end in units of one growth step")

rho = (1 + DT / tau) * np.exp(-DT / tau); sD = ff * np.sqrt(2 * (1 - rho))
wig = np.sqrt(2 / np.pi) * sD * inv * n          # adiabatic TV over the whole spell
mer = 2 * ell * (K - 1)
D = np.concatenate(Pb); y = (TV - rng2)[D]
Xm = np.vstack([mer[D], wig[D]]).T
a, b = np.linalg.lstsq(Xm, y, rcond=None)[0]; print(f"fit TV - range = a*2ell(K-1) + b*wiggle:  a={a:.3f}  b={b:.3f}")
for lab, A_, B_ in (("fit", a, b), ("a=.35,b=0", .35, 0.0)):
    law = np.sqrt(np.pi/2) * (rng2 + A_ * mer + B_ * wig) / n
    z = zq(Sb, law, Pb)
    print(f"  {lab:10s} zeta_q {' '.join(f'{q:.3f}' for q in z)}  ratios {' '.join(f'{q:.2f}' for q in z[1:]/z[0])}   med|err| {np.median(np.abs(np.log(law/v)[D])):.3f}")
# where is the residual: by size quartile on the branch
res = (TV - rng2 - .35 * mer)[D] / (2 * ell[D])
qs = np.quantile(Sb[D], [0, .25, .5, .75, 1])
for i in range(4):
    sel = (Sb[D] >= qs[i]) & (Sb[D] <= qs[i+1])
    print(f"  size quartile {i+1}: Sbar~{np.median(Sb[D][sel]):.2f}  residual/(2ell) {np.median(res[sel]):+.3f}   wiggle/(2ell) {np.median((wig[D]/(2*ell[D]))[sel]):.3f}  K-1 mean {np.mean(K[D][sel]-1):.2f}")

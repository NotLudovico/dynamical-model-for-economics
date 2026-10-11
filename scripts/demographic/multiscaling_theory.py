"""Test of the multiscaling mechanism of the demographic-noise model.

Claim.  With a firm's volatility v_i^2 = v_F,i^2 + B(Sbar) (fitness floor plus a demographic term
that is the same function of size for every firm), exactly
    zeta_q(Sbar) = (q/2) <w>_q,   w = B / (v_F^2 + B),
with <.>_q the v^q-weighted average at fixed size.  <w>_q falls with q, so zeta_q/zeta_1 < q, iff
v_F differs across firms; on the network v_F grows with the degree (fitness amplitude ~ sqrt(kappa)).

Tests (alpha 2.5, sigma 1.70, D 0.1, N 16000, C 200, 2 economies; and the same on a random regular
graph, every degree = C, as the ablation):
  T1  in 25 equal-count bins of Sbar (lives >= 20): Var(ln v | Sbar) and the share of it explained by
      ln(degree) (R^2 of a within-bin regression); the claim needs both to rise with size.
  T2  fit v^2 = A kappa^g + B_bin (one B per size bin, common A, g) and predict zeta_q from the fitted
      law averaged over the degrees of the lives in each bin; compare with the measured zeta_q.
  T3  ratio curves: power-law graph against regular graph.
Trajectories in data/demographic/multiscaling/.
Run: uv run python scripts/demographic/multiscaling_theory.py   (simulate if needed, then analyse)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy import optimize

OUT = os.path.join(ROOT, "data", "demographic", "multiscaling")
N, C, ALPHA, SIGMA, D, MU = 16000, 200, 2.5, 1.70, 0.1, 1.76
START, TOBS, DT = 100.0, 480.0, 0.5
CASES = (("powerlaw", 0), ("powerlaw", 1), ("regular", 0), ("regular", 1))


def coupling(kind, seed):
    from relative_glv.model import configuration_coupling
    A = configuration_coupling(N, MU, SIGMA, alpha=ALPHA, mean_degree=C, seed=seed, regular=(kind == "regular"))
    return A, np.diff(A.indptr)


def path(kind, seed):
    return os.path.join(OUT, f"{kind}_seed{seed}.npz")


def simulate(kind, seed):
    f = path(kind, seed)
    if os.path.exists(f):
        return f
    from relative_glv import model
    A, deg = coupling(kind, seed)
    r = model.integrate_demographic(A, D=D, tmax=START + TOBS, t_start=START, dt_save=DT, seed=seed, compact=True)
    np.savez_compressed(f, S=r["S"], L=r["life"], deg=deg)
    return f


def lives(S, L, deg, nmin):
    """(Sbar, v, degree, n) for every life with at least nmin growth rates."""
    from relative_glv.msb import lives as life_panel
    lv = [x for x in life_panel(S, L) if x[2] >= nmin]
    return (np.array([x[0] for x in lv]), np.array([x[1] for x in lv]),
            np.array([deg[x[5]] for x in lv], float), np.array([x[2] for x in lv]))


def zetas(Sb, v, nb=25, qs=(1, 2, 3, 4)):
    P = np.array_split(np.argsort(Sb), nb)
    lx = np.log([Sb[p].mean() for p in P])
    return np.array([-np.polyfit(lx, np.log([np.mean(v[p] ** q) for p in P]), 1)[0] for q in qs]), P, lx


def analyse(kind, nmin=20):
    ds = [np.load(path(kind, s)) for s in (0, 1) if os.path.exists(path(kind, s))]
    parts = [lives(d["S"], d["L"], d["deg"], nmin) for d in ds]
    Sb, v, k, n = (np.concatenate(x) for x in zip(*parts))
    kap = k / k.mean()
    z, P, lx = zetas(Sb, v)
    print(f"\n=== {kind} graph, lives >= {nmin}: {Sb.size} lives, degree CV {kap.std():.2f}")
    print(f"  measured zeta {' '.join(f'{x:.3f}' for x in z)}  ratios {' '.join(f'{x:.2f}' for x in z[1:] / z[0])}")
    # T1: Var(ln v | Sbar) and the share explained by ln kappa, by fifths of the bins
    tot, expl = [], []
    for p in P:
        y = np.log(v[p]); tot.append(y.var())
        if kap[p].std() > 0:
            c = np.polyfit(np.log(kap[p]), y, 1); expl.append(1 - np.var(y - np.polyval(c, np.log(kap[p]))) / y.var())
        else:
            expl.append(0.0)
    tot, expl = np.array(tot), np.array(expl)
    f5 = lambda a: " ".join(f"{a[i:i + 5].mean():.3f}" for i in range(0, 25, 5))
    print(f"  T1 Var(ln v | Sbar) by fifths: {f5(tot)}   share explained by degree: {f5(expl)}")
    out = dict(lx=lx, tot=tot, expl=expl, z=z)
    if kind == "powerlaw":
        # T2: v^2 = A kappa^g + B_bin, least squares on log v^2 residuals
        bi = np.empty(Sb.size, int)
        for j, p in enumerate(P):
            bi[p] = j
        def resid(th):
            A, g, lB = th[0], th[1], th[2:]
            return np.log(v ** 2) - np.log(np.exp(A) * kap ** g + np.exp(lB[bi]))
        th0 = np.r_[np.log(np.mean(v ** 2) / 2), 1.0, np.log([np.mean(v[p] ** 2) / 2 for p in P])]
        sol = optimize.least_squares(resid, th0)
        A, g, lB = np.exp(sol.x[0]), sol.x[1], sol.x[2:]
        r = resid(sol.x)
        print(f"  T2 fit v^2 = A kappa^g + B(Sbar): A = {A:.4f}, g = {g:.2f} (theory: 1), "
              f"B falls from {np.exp(lB[0]):.3f} to {np.exp(lB[-1]):.4f}; residual sd of ln v^2 {r.std():.2f}")
        # predicted moments: law times the empirical residual spread (independent of size by construction)
        eps = np.exp(r / 2)
        zp = []
        for q in (1, 2, 3, 4):
            m = [np.mean((A * kap[p] ** g + np.exp(lB[j])) ** (q / 2)) * np.mean(eps ** q) for j, p in enumerate(P)]
            zp.append(-np.polyfit(lx, np.log(m), 1)[0])
        zp = np.array(zp)
        print(f"  T2 predicted zeta {' '.join(f'{x:.3f}' for x in zp)}  ratios {' '.join(f'{x:.2f}' for x in zp[1:] / zp[0])}")
        # same law with every firm at the mean degree: the factorisation limit
        zh = []
        for q in (1, 2, 3, 4):
            m = [((A + np.exp(lB[j])) ** (q / 2)) * np.mean(eps ** q) for j, p in enumerate(P)]
            zh.append(-np.polyfit(lx, np.log(m), 1)[0])
        zh = np.array(zh)
        print(f"     same law, degree heterogeneity removed: ratios {' '.join(f'{x:.2f}' for x in zh[1:] / zh[0])} (= q exactly)")
        # noise share w by fifths and its v^q-weighted averages
        w = np.exp(lB[bi]) / (A * kap ** g + np.exp(lB[bi]))
        ws = [[np.sum(w[p] * v[p] ** q) / np.sum(v[p] ** q) for q in (1, 2, 3, 4)] for p in P]
        ws = np.array(ws)
        print("  <w>_q by fifths (q = 1..4): " + "  ".join(" ".join(f"{x:.2f}" for x in ws[i:i + 5].mean(0)) for i in range(0, 25, 5)))
        out.update(A=A, g=g, lB=lB, zp=zp, ws=ws)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:2] != ["analyse"]:
        from joblib import Parallel, delayed
        Parallel(n_jobs=4, verbose=5)(delayed(simulate)(*c) for c in CASES)
    res = {k: analyse(k) for k in ("powerlaw", "regular")}
    np.savez(os.path.join(OUT, "results.npz"), **{f"{k}_{kk}": vv for k, r in res.items() for kk, vv in r.items()})
    print("\nCompustat ratios 1.95 2.55 2.90 (zeta .20 .39 .51 .58)")

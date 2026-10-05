"""Robustness batch for the demographic-noise model (no lambda, newborn at S = D).

Points: alpha 6 and 3.5 x D 1e-3..1e-1, plus alpha 3.5/3/2.5 at D 0.1-0.3 (see POINTS); N = 8000, C = 200,
8 economies each, burn-in 100, window 480, sampling 0.5, Euler dt 0.01.
Per economy: lives (birth to death, window-censored included) with mean size, MSB volatility
(sqrt(pi/2) MAD of growth around own mean), number of growth rates; plus a subsample of
one-step (growth, current size) pairs within lives.
Run: uv run python scripts/demographic/batch.py   |   summary: ... batch.py summary
Env: BN, BC (N, mean degree), BPOINT=alpha,sigma,D (single point).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "demographic")
import numpy as np

OUT = os.path.join(DATA, "batch")
POINTS = [(a, s, D) for a, s in ((6.0, 1.75), (3.5, 1.9)) for D in (1e-3, 1e-2, 1e-1)] + \
    [(3.5, 1.9, 0.2), (3.5, 1.9, 0.3), (3.0, 1.9, 0.1), (3.0, 1.9, 0.2), (2.5, 1.75, 0.1), (2.5, 1.75, 0.2)]
N, MU, START, TOBS, DT, dt, SEEDS = int(os.environ.get("BN", 8000)), 1.76, 100.0, 480.0, 0.5, 0.01, 8
C = int(os.environ.get("BC", N // 40))
if "BPOINT" in os.environ:                      # e.g. BPOINT=2.5,1.75,0.1
    POINTS = [tuple(float(x) for x in os.environ["BPOINT"].split(","))]


def economy(alpha, sigma, D, seed):
    from relative_glv import model
    model._ALPHA_PL = alpha
    a = model.coupling(N, MU, sigma, kind="powerlaw", seed=seed, mean_degree=C)
    r = model.integrate_demographic(a, D=D, tmax=START + TOBS, t_start=START, dt_save=DT, dt=dt, seed=seed)
    S, L = r["S"], r["life"]
    rng = np.random.default_rng(seed)
    lnS, T = np.log(S), S.shape[1]
    Sb, v, n = [], [], []
    for i in range(N):
        for seg in np.split(np.arange(T), np.flatnonzero(np.diff(L[i])) + 1):
            if seg.size < 3:
                continue
            g = np.diff(lnS[i, seg])
            Sb.append(S[i, seg].mean()); v.append(np.sqrt(np.pi / 2) * np.abs(g - g.mean()).mean()); n.append(g.size)
    same = L[:, 1:] == L[:, :-1]
    g1, s0 = (lnS[:, 1:] - lnS[:, :-1])[same], S[:, :-1][same]
    pick = rng.choice(g1.size, min(g1.size, 300_000), replace=False)
    never = L[:, -1] == L[:, 0]
    diag = np.array([(L[:, -1] - L[:, 0]).sum() / N / TOBS, never.mean(), S[never].sum() / S.sum()])
    return dict(Sb=np.array(Sb), v=np.array(v), n=np.array(n), g1=g1[pick], s0=s0[pick], diag=diag)


def path(a, s, D):
    tag = "" if (N, C) == (8000, 200) else f"_N{N}_C{C}"
    return os.path.join(OUT, f"a{a}_s{s}_D{D:g}{tag}.npz")


def curve(x, y, nb=25):
    P = np.array_split(np.argsort(x), nb)
    return np.log([x[p].mean() for p in P]), [np.log([np.mean(y[p] ** q) for p in P]) for q in (1, 2, 3, 4)]


def zetas(x, y):
    lx, ly = curve(x, y)
    z = np.array([-np.polyfit(lx, l, 1)[0] for l in ly])
    fifths = [-np.polyfit(lx[i:i + 5], ly[0][i:i + 5], 1)[0] for i in range(0, 25, 5)]
    return z, np.array(fifths), (lx[-1] - lx[0]) / np.log(10)


def summary():
    print(f"{'point':>20} | {'min g':>5} | {'beta (sd)':>13} | {'zeta_q/zeta_1':>16} | {'local beta by fifths':>32} | dec | lives/econ")
    for p in POINTS:
        if not os.path.exists(path(*p)):
            continue
        d = np.load(path(*p)); e = d["econ"]
        for mg in (2, 5, 10, 20):
            R = [zetas(d["Sb"][(e == i) & (d["n"] >= mg)], d["v"][(e == i) & (d["n"] >= mg)]) for i in np.unique(e)]
            Z = np.array([r[0] for r in R]); zm = Z.mean(0)
            F = np.mean([r[1] for r in R], 0)
            print(f"a={p[0]} D={p[2]:g}".rjust(20) + f" | {mg:5d} | {np.mean(Z[:, 0]):.3f} ({np.std(Z[:, 0]):.3f}) | "
                  + " ".join(f"{x:5.2f}" for x in zm[1:] / zm[0]) + " | " + " ".join(f"{x:+.2f}" for x in F)
                  + f" | {np.mean([r[2] for r in R]):.1f} | {np.sum((d['n'] >= mg)) / len(R):.0f}")
        R = [zetas(d["s0"][d["ge"] == i], np.abs(d["g1"][d["ge"] == i] - d["g1"][d["ge"] == i].mean()) * np.sqrt(np.pi / 2))
             for i in np.unique(d["ge"])]
        Z = np.array([r[0] for r in R]); F = np.mean([r[1] for r in R], 0)
        print(f"{'current size':>20} | {'-':>5} | {np.mean(Z[:, 0]):.3f} ({np.std(Z[:, 0]):.3f}) | "
              + " ".join(f"{x:5.2f}" for x in Z.mean(0)[1:] / Z.mean(0)[0]) + " | " + " ".join(f"{x:+.2f}" for x in F)
              + f" | {np.mean([r[2] for r in R]):.1f} |")
        dg = d["diag"].mean(0)
        print(f"{'':>20}   births/firm/time {dg[0]:.2f}, nodes never dying {dg[1]:.3f} holding {dg[2]:.2f} of size")
    print("data: beta 0.15-0.20, ratios 2.0 2.6 2.9")


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary(); sys.exit()
    from joblib import Parallel, delayed
    os.makedirs(OUT, exist_ok=True)
    for p in POINTS:
        if os.path.exists(path(*p)):
            continue
        R = Parallel(n_jobs=8)(delayed(economy)(*p, s) for s in range(SEEDS))
        np.savez(path(*p), econ=np.concatenate([np.full(r["Sb"].size, i) for i, r in enumerate(R)]),
                 ge=np.concatenate([np.full(r["g1"].size, i) for i, r in enumerate(R)]),
                 diag=np.array([r["diag"] for r in R]),
                 **{k: np.concatenate([r[k] for r in R]) for k in ("Sb", "v", "n", "g1", "s0")})
        print("done", p, flush=True)
    summary()

"""Low-N points (N=2000, 4000) for the appendix topology-scaling figure.

The 8k/16k/32k points already exist (scattered, published in tab:scaling); this only runs
the cheap small-N end so the reader sees the finite-size dropdown. Same operating point,
estimator, and mean-degree normalization as notebooks/meandeg_topology.ipynb, so the new
points sit on the existing curves. Saves data/scaling_lowN.npz.

  P_SEEDS (20), P_NJOBS (6), P_SMOKE=1
"""
import os, time
import numpy as np
import networkx as nx
from scipy import sparse
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate
from relative_glv.msb import size_volatility

SMOKE = os.environ.get("P_SMOKE", "0") == "1"
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
TMAX, N_EVAL, LATE, DT, NB = 250.0, 700, (180.0, 240.0), 0.5, 20
SEEDS = 3 if SMOKE else int(os.environ.get("P_SEEDS", 20))
NJOBS = int(os.environ.get("P_NJOBS", 6))
NS = [2000, 4000]
TOPOS = os.environ.get("P_TOPOS", "fc,regular,pl3.5,pl2.5,pl2.2").split(",")


def degree_seq(topo, N, C, rng):
    if topo == "regular":
        return np.full(N, C, float)
    a = float(topo[2:])                                   # "pl2.2" -> 2.2
    kmin = C * (a - 2) / (a - 1)
    return np.maximum(kmin * (1 - rng.uniform(size=N)) ** (-1 / (a - 1)), 1.0)


def build(topo, N, C, seed):
    if topo == "fc":
        return coupling(N, MU, SIGMA, kind="fc", seed=seed)
    rng = np.random.default_rng(seed)
    deg = np.maximum(np.round(degree_seq(topo, N, C, rng)).astype(int), 1)
    if deg.sum() % 2:
        deg[deg.argmin()] += 1
    G = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(seed)))
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    ki = np.diff(A.indptr).astype(float); ki[ki == 0] = 1.0
    C_eff = float(ki.mean())
    coo = A.tocoo(); z = rng.normal(size=coo.row.size)
    vals = MU / C_eff + (SIGMA / np.sqrt(C_eff)) * z
    return sparse.csr_array((vals, (coo.row, coo.col)), shape=(N, N))


def cq_one(S, v, nb=NB):
    """Quenched zeta_1..4 on one economy's own decline branch (matches meandeg_topology.ipynb)."""
    live = (v > 0) & np.isfinite(v) & (S > 0)
    S, v = S[live], v[live]
    if S.size < 200:
        return None
    o = np.argsort(S); P = np.array_split(np.arange(o.size), nb)
    bx = np.array([S[o][p].mean() for p in P]); by = np.array([np.median(v[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]
    pk = int(np.argmax(by)); dec = S > bx[pk]; nfit = max(6, nb - pk)
    if dec.sum() < 200:
        return None
    Sx, vx = S[dec], v[dec]
    ox = np.argsort(Sx); Px = np.array_split(np.arange(ox.size), nfit)
    bxx = np.array([Sx[ox][p].mean() for p in Px]); out = []
    for q in (1, 2, 3, 4):
        byy = np.array([np.mean(vx[ox][p] ** q) for p in Px])
        mm = (bxx > 0) & (byy > 0)
        out.append(-np.polyfit(np.log10(bxx[mm]), np.log10(byy[mm]), 1)[0])
    return np.array(out)


def economy(topo, N, seed):
    C = max(2, round(N / 40))
    a = build(topo, N, C, seed)
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                  method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT, n_bins=NB)
    if sv["growth"].size == 0 or not np.isfinite(sv["beta"]):
        return None
    return dict(topo=topo, N=N, beta=float(sv["beta"]), cq=cq_one(sv["Sbar"], sv["vol"]))


jobs = [(t, N, s) for N in NS for t in TOPOS for s in range(SEEDS)]
print(f"{'SMOKE ' if SMOKE else ''}low-N splice  Ns={NS}  {SEEDS} seeds x {TOPOS}  ({len(jobs)} economies)")
t0 = time.time()
res = [x for x in Parallel(n_jobs=NJOBS, backend="loky")(
    delayed(economy)(t, N, s) for t, N, s in jobs) if x is not None]
print(f"integrated {len(res)}/{len(jobs)} in {(time.time()-t0)/60:.1f} min")

out = {}
for N in NS:
    for topo in TOPOS:
        rows = [x for x in res if x["topo"] == topo and x["N"] == N]
        be = np.array([x["beta"] for x in rows])
        cq = np.array([x["cq"] for x in rows if x["cq"] is not None])
        n = len(be)
        if n == 0:
            print(f"  N={N} {topo}: 0 survived"); continue
        b_med, b_err = float(np.median(be)), float(be.std() / max(1, np.sqrt(n)))
        if len(cq):
            r4 = cq[:, 3] / cq[:, 0]
            r4_med, r4_err = float(np.median(r4)), float(r4.std() / max(1, np.sqrt(len(cq))))
        else:
            r4_med = r4_err = np.nan
        out[f"{topo}_N{N}"] = np.array([n, b_med, b_err, r4_med, r4_err])
        print(f"  N={N:5d} {topo:8s} n={n:2d}  beta={b_med:.3f}+-{b_err:.3f}  z4/1={r4_med:.3f}+-{r4_err:.3f}")

np.savez("data/scaling_lowN.npz", **out)
print("saved data/scaling_lowN.npz")

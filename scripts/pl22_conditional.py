"""MSB conditional tests D1 and D3 on the power-law-2.2 graph.

pl2.2 has the converged beta inside the empirical band; this checks whether it also
passes the two conditional tests the thesis so far verifies only at pl2.5:
  D1: rescaled volatility sigma_i/sigmabar_e(S) collapses with a bounded tail
      (per-economy rescale, tail = P99/median)
  D3: size-unmixed growth (g-gbar)/sigma_i does NOT Gaussianize with size
      (per-economy kurtosis-vs-size trend, positive = MSB)
Quenched beta, zeta ratios and tent stats come along for free from the same sims.
Estimators mirror notebooks/meandeg_characterize.ipynb exactly.

Run from repo root: python3 scripts/pl22_conditional.py
Env: P22_SEEDS (20), P22_N (16000), P22_NJOBS (6), P22_SMOKE=1 (code-path check).
"""
import os, sys
while not os.path.isdir("relative_glv") and os.path.dirname(os.getcwd()) != os.getcwd():
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.stats import kurtosis
from joblib import Parallel, delayed
from relative_glv.model import integrate
from relative_glv.msb import size_volatility, tent_stats, rescale

SMOKE = os.environ.get("P22_SMOKE", "0") == "1"
MU, SIGMA, LAM, ALPHA_PL = 1.76, 1.75, 1e-3, 2.2
N = 1500 if SMOKE else int(os.environ.get("P22_N", 16000))
Cdeg = max(2, round(100 * N / 4000))               # C ∝ N anchor
SEEDS = 2 if SMOKE else int(os.environ.get("P22_SEEDS", 20))
TMAX, N_EVAL = (100.0, 400) if SMOKE else (250.0, 700)
LATE, DT = ((75.0, 98.0) if SMOKE else (180.0, 240.0)), 0.5
NJOBS = int(os.environ.get("P22_NJOBS", 6))
NB = 20

print(f"{'SMOKE ' if SMOKE else ''}pl{ALPHA_PL} conditional tests  N={N} C={Cdeg}  {SEEDS} seeds  "
      f"mu={MU} sigma={SIGMA} lam={LAM}")


def pl_alpha(N, mu, sigma, seed, a=ALPHA_PL):
    """Mean-degree couplings on a power-law-a configuration graph (meandeg_alpha of the
    topology notebook, exponent generalized)."""
    rng = np.random.default_rng(seed)
    kmin = Cdeg * (a - 2) / (a - 1)
    deg = np.maximum(np.round(
        kmin * (1 - rng.uniform(size=N)) ** (-1 / (a - 1))).astype(int), 1)
    if deg.sum() % 2:
        deg[deg.argmin()] += 1
    G = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(seed)))
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    ki = np.diff(A.indptr).astype(float); ki[ki == 0] = 1.0
    C_eff = float(ki.mean())
    coo = A.tocoo()
    z = rng.normal(size=coo.row.size)
    vals = mu / C_eff + (sigma / np.sqrt(C_eff)) * z
    return sparse.csr_array((vals, (coo.row, coo.col)), shape=(N, N))


def cq_one(S, v, nb=NB):
    """zeta_1..4 for ONE economy on its OWN decline branch (pooling-free estimator)."""
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
        if mm.sum() < 3:
            return None
        out.append(-np.polyfit(np.log10(bxx[mm]), np.log10(byy[mm]), 1)[0])
    c = np.array(out)
    return c if c[0] > 0 else None


def simulate(seed):
    a = pl_alpha(N, MU, SIGMA, seed)
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                  method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT, n_bins=NB)
    if sv["growth"].size == 0 or not np.isfinite(sv["beta"]):
        return None
    return dict(Sbar=sv["Sbar"], vol=sv["vol"], growth=sv["growth"], beta=float(sv["beta"]))


recs = [x for x in Parallel(n_jobs=NJOBS)(delayed(simulate)(s) for s in range(SEEDS))
        if x is not None]
assert recs, "no economies survived"
print(f"survived {len(recs)}/{SEEDS}")

Sbar = np.concatenate([x["Sbar"] for x in recs])
vol = np.concatenate([x["vol"] for x in recs])
growth = np.vstack([x["growth"] for x in recs])
econ = np.concatenate([np.full(x["Sbar"].size, i) for i, x in enumerate(recs)])
live = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
Sbar, vol, growth, econ = Sbar[live], vol[live], growth[live], econ[live]

beta_med = float(np.median([x["beta"] for x in recs]))
beta_sd = float(np.std([x["beta"] for x in recs]))

# quenched zeta ratios
Z = np.array([c for c in (cq_one(Sbar[econ == e], vol[econ == e])
                          for e in np.unique(econ)) if c is not None])
ratio_q = Z.mean(0) / Z.mean(0)[0] if Z.size else np.full(4, np.nan)
ratio_q_err = (Z / Z[:, :1]).std(0) / np.sqrt(len(Z)) if Z.size else np.full(4, np.nan)

# D1 + D3 on PER-ECONOMY decline branches (a pooled Scut mixes economy scales and, at
# pl2.2's wide scale spread, silently drops most economies from the tests)
gres = growth - growth.mean(1, keepdims=True)
ghat = gres / vol[:, None]
NBK = 6
rr_all, rr_rank, tr_e, K_e = [], [], [], []
for e in np.unique(econ):
    m = econ == e
    S_e, v_e, gh_e = Sbar[m], vol[m], ghat[m]
    o = np.argsort(S_e); P = np.array_split(np.arange(o.size), NB)
    bx = np.array([S_e[o][p].mean() for p in P])
    by = np.array([np.mean(v_e[o][p]) for p in P])
    dece = S_e > bx[int(np.argmax(by))]
    if dece.sum() < 8 * NBK:
        continue
    Se, ve = S_e[dece], v_e[dece]
    oe = np.argsort(Se); Pe = np.array_split(np.arange(Se.size), max(6, NB // 2))
    bxe = np.array([Se[oe][p].mean() for p in Pe])
    bye = np.array([np.mean(ve[oe][p]) for p in Pe])
    rr_all.append(ve / np.interp(Se, bxe, bye))
    rr_rank.append(np.argsort(np.argsort(Se)) / (Se.size - 1))   # within-economy size rank
    Qe = np.array_split(oe, NBK)
    bxk = np.array([Se[q].mean() for q in Qe])
    bke = np.array([kurtosis(gh_e[dece][q].ravel()) for q in Qe])
    K_e.append(bke)
    tr_e.append(np.polyfit(np.log10(bxk), bke, 1)[0])
rr_q = np.concatenate(rr_all)
rank = np.concatenate(rr_rank)
d1_tail_q = float(np.percentile(rr_q, 99) / np.median(rr_q))
ter = [rank < 1 / 3, (rank >= 1 / 3) & (rank < 2 / 3), rank >= 2 / 3]
d1_med3 = [float(np.median(rr_q[i])) for i in ter]
d1_p99_3 = [float(np.percentile(rr_q[i], 99)) for i in ter]
tr_e = np.array(tr_e)
d3_trend_q = float(np.median(tr_e))
d3_trend_q_err = float(tr_e.std() / np.sqrt(tr_e.size))

ts = tent_stats(growth)

print(f"\nbeta (median per economy)      {beta_med:.3f} ± {beta_sd:.3f}")
print(f"quenched ratios zeta_q/zeta_1  " + "  ".join(f"{v:.2f}±{e:.2f}" for v, e in zip(ratio_q, ratio_q_err)))
print(f"tent: bowley {ts['bowley']:+.2f}   excess kurtosis {ts['exkurt']:.1f}")
print(f"D1  tail P99/median (per-economy rescale) = {d1_tail_q:.2f}")
print(f"    median by size tercile {['%.2f' % v for v in d1_med3]}   P99 {['%.2f' % v for v in d1_p99_3]}")
print(f"D3  per-economy kurtosis trend d(exkurt)/d(logS) = {d3_trend_q:+.2f} ± {d3_trend_q_err:.2f}"
      f"   positive in {int((tr_e > 0).sum())}/{tr_e.size} economies")

tag = "_smoke" if SMOKE else ""
np.savez(f"data/pl22_conditional{tag}.npz",
         N=N, C=Cdeg, seeds=len(recs), alpha_pl=ALPHA_PL,
         beta=beta_med, beta_sd=beta_sd,
         zeta_per_econ=Z, ratio_quenched=ratio_q, ratio_quenched_err=ratio_q_err,
         d1_tail_quenched=d1_tail_q, d1_med3=d1_med3, d1_p99_3=d1_p99_3,
         d3_trend_quenched=d3_trend_q, d3_trend_quenched_err=d3_trend_q_err,
         d3_trend_per_econ=tr_e, d3_kurt_per_econ=np.array(K_e),
         bowley=ts["bowley"], exk=ts["exkurt"],
         tent_z=rescale(growth)[::max(1, growth.size // 40000)].astype(np.float32))
print(f"saved data/pl22_conditional{tag}.npz")

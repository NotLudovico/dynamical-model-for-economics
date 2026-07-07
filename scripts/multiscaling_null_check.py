"""Artifact check on the multiscaling concavity (c_q/c_1 < q, the anti-granular signal).

Does the D2 pipeline manufacture concavity, or is it in the data? Compare the real c_q to a
SINGLE-SCALE null surrogate: keep the real sigma(S) trend but make the volatility-distribution
SHAPE size-independent (permute the rescaled vols u = sigma/sigmabar(S) across all decline
firms). By construction the null has c_q = q*c_1 (no multiscaling). Run the identical estimator
on both:
  - null c_q/c_1 ~ [1,2,3,4]  => pipeline is clean, real concavity is GENUINE
  - null c_q/c_1 concave too  => pipeline manufactures it, real result is an ARTIFACT
Bootstrap the real over economies for its own error bars.

    uv run python growing_glv/multiscaling_null_check.py [--smoke]
"""
import sys, os
import numpy as np
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate
from relative_glv.msb import size_volatility

SMOKE = "--smoke" in sys.argv
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N, SEEDS = (800, 6) if SMOKE else (4000, 40)
TMAX, N_EVAL = (120.0, 600) if SMOKE else (400.0, 800)
LATE, DT = ((95.0, 118.0) if SMOKE else (320.0, 390.0)), 0.5
NB = 20
QS = (1, 2, 3, 4)
NDRAW = 20 if SMOKE else 300


def sim(seed):
    a = coupling(N, MU, SIGMA, kind="powerlaw_owndeg", seed=seed)
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed, method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT, n_bins=NB)
    if sv["growth"].size == 0 or not np.isfinite(sv["beta"]):
        return None
    return sv["Sbar"], sv["vol"], seed


def cq(S, v, nbins):
    """D2 estimator: equal-count size bins, raw moment E[sigma^q]=mean per bin, c_q=-slope."""
    o = np.argsort(S); P = np.array_split(np.arange(o.size), nbins)
    bx = np.array([S[o][p].mean() for p in P])
    out = []
    for q in QS:
        byq = np.array([np.mean(v[o][p] ** q) for p in P])
        m = (bx > 0) & (byq > 0)
        out.append(-np.polyfit(np.log10(bx[m]), np.log10(byq[m]), 1)[0])
    return np.array(out)


if __name__ == "__main__":
    recs = [r for r in Parallel(n_jobs=6, backend="loky")(delayed(sim)(s) for s in range(SEEDS)) if r]
    Sbar = np.concatenate([r[0] for r in recs])
    vol = np.concatenate([r[1] for r in recs])
    eid = np.concatenate([np.full(r[0].size, i) for i, r in enumerate(recs)])
    live = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
    Sbar, vol, eid = Sbar[live], vol[live], eid[live]

    # decline branch on the pooled binned-median sigma(S) (matches msb_conditional)
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), NB)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    pk = int(np.argmax(by)); Scut = bx[pk]
    dec = Sbar > Scut
    Sd, vd, ed = Sbar[dec], vol[dec], eid[dec]
    nbins = max(6, NB - pk)
    print(f"economies={len(recs)}  decline firms={Sd.size}  bins={nbins}  Scut={Scut:.3g}", flush=True)

    c_real = cq(Sd, vd, nbins)
    print(f"\nREAL   c_q = {np.round(c_real,3)}   c_q/c_1 = {np.round(c_real/c_real[0],2)}   (linear = [1 2 3 4])")

    # sigma_bar(S) mean trend -> rescaled u = sigma/sigmabar (size-independent under the null)
    ob = np.argsort(Sd); Pb = np.array_split(np.arange(ob.size), nbins)
    bxm = np.array([Sd[ob][p].mean() for p in Pb]); bmean = np.array([np.mean(vd[ob][p]) for p in Pb])
    sbar_i = np.interp(Sd, bxm, bmean)
    u = vd / sbar_i

    rng = np.random.default_rng(0)
    null = np.array([cq(Sd, sbar_i * rng.permutation(u), nbins) for _ in range(NDRAW)])
    nr = null / null[:, [0]]
    ue = np.unique(ed)
    boot = []
    for _ in range(NDRAW):
        pick = rng.choice(ue, ue.size, replace=True)
        idx = np.concatenate([np.where(ed == e)[0] for e in pick])
        boot.append(cq(Sd[idx], vd[idx], nbins))
    br = np.array(boot); br = br / br[:, [0]]

    # ABSOLUTE exponent error bars
    print(f"\nZETA_q (pooled point)                 = {np.round(c_real,3)}")
    print(f"ZETA_q +/- bootstrap-over-economies   = {np.round(np.array(boot).mean(0),3)} +/- {np.round(np.array(boot).std(0),3)}")
    # per-economy scatter (like the beta=0.8+/-0.1 error): each economy's own decline firms
    pe = []
    for e in ue:
        m = ed == e
        if m.sum() > 300:
            cc = cq(Sd[m], vd[m], max(6, nbins - 4))
            if np.all(np.isfinite(cc)):
                pe.append(cc)
    pe = np.array(pe)
    print(f"ZETA_q per-economy (n={len(pe)}) mean +/- sd  = {np.round(pe.mean(0),3)} +/- {np.round(pe.std(0),3)}\n")
    print(f"NULL   c_q/c_1 = {np.round(nr.mean(0),2)} +/- {np.round(nr.std(0),2)}   (single-scale surrogate)")
    print(f"REAL*  c_q/c_1 = {np.round(br.mean(0),2)} +/- {np.round(br.std(0),2)}   (bootstrap over economies)\n")
    print(" q | linear |   null (should be q)   |  real (bootstrap)   | verdict")
    for i, q in enumerate(QS):
        z = (nr[:, i].mean() - br[:, i].mean()) / np.sqrt(nr[:, i].var() + br[:, i].var() + 1e-12)
        v = "GENUINE (real << null)" if z > 2 else ("artifact-like (real ~ null)" if abs(z) <= 2 else "?")
        print(f" {q} |   {q}    |  {nr[:,i].mean():.2f} +/- {nr[:,i].std():.2f}         |  "
              f"{br[:,i].mean():.2f} +/- {br[:,i].std():.2f}       | {v if q>1 else '-'}  (z={z:+.1f})")
    print("\nread: null~[1,2,3,4] => pipeline clean, real concavity genuine;  null concave => artifact")

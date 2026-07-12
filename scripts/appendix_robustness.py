"""Appendix robustness checks of the measurement protocol.

Two axes, same headline observables (decline-branch beta, quenched zeta ratios,
tent excess kurtosis) on the thesis graph (power-law 2.5, mean-degree, C = N/40):

  1. Observation lag: each economy is integrated ONCE at fine output resolution,
     then every Delta t in DTS is measured on the SAME trajectory, so any
     difference is a pure estimator effect.
  2. Solver tolerance: a subset of seeds is re-integrated at rtol 1e-6 / atol 1e-9
     (production: 1e-4 / 1e-7) and measured at the production Delta t = 0.5.

Run from repo root: .venv/bin/python scripts/appendix_robustness.py
Env: ROB_SMOKE=1 (code-path check only), ROB_SEEDS (10), ROB_N (8000), ROB_NJOBS (6).
"""
import os, sys
while not os.path.isdir("relative_glv") and os.path.dirname(os.getcwd()) != os.getcwd():
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate
from relative_glv.msb import size_volatility, tent_stats

SMOKE = os.environ.get("ROB_SMOKE", "0") == "1"
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N = 4000 if SMOKE else int(os.environ.get("ROB_N", 8000))  # <4k collapses below the 50-survivor fit floor
Cdeg = max(2, round(N / 40))
SEEDS = 2 if SMOKE else int(os.environ.get("ROB_SEEDS", 10))
TMAX, N_EVAL = (100.0, 800) if SMOKE else (250.0, 2000)
LATE = (75.0, 98.0) if SMOKE else (180.0, 240.0)
DTS = [0.25, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
DT_PROD = 0.5
NJOBS = int(os.environ.get("ROB_NJOBS", 6))
NB = 20
TOL_SEEDS = 1 if SMOKE else 3

print(f"{'SMOKE ' if SMOKE else ''}robustness checks  N={N} C={Cdeg}  {SEEDS} seeds  "
      f"mu={MU} sigma={SIGMA} lam={LAM}  dts={DTS}")


def cq_one(S, v, nb=NB):
    """zeta_1..4 for ONE economy on its OWN decline branch (pl22_conditional.py)."""
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


def measure(W, t):
    """All observables at every Delta t from one trajectory. None if any dt fails."""
    rec = {}
    for dt in DTS:
        sv = size_volatility(W, t, window=LATE, dt=dt, n_bins=NB)
        if sv["growth"].size == 0 or not np.isfinite(sv["beta"]):
            return None
        z = cq_one(sv["Sbar"], sv["vol"])
        rec[dt] = dict(beta=float(sv["beta"]), zeta=z,
                       exk=tent_stats(sv["growth"])["exkurt"],
                       medvol=float(np.median(sv["vol"][sv["vol"] > 0])))
    return rec


def acorr_tau(W, t, dt=0.25, nlag=40):
    """Firm-averaged autocorrelation of ln S over the late window, and its decay
    times: half-life, 1/e time, integrated time (sum to the first zero crossing)."""
    N = W.shape[0]
    win = (t >= LATE[0]) & (t <= LATE[1])
    live = W[:, win].min(1) > 1e-6
    tg = np.arange(LATE[0], LATE[1] + 1e-9, dt)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12)))
                    for i in np.where(live)[0]])
    x = lnS - lnS.mean(1, keepdims=True)
    T = x.shape[1]
    c = np.array([(x[:, :T - k] * x[:, k:]).mean(1) for k in range(nlag)])
    cn = (c / c[0]).mean(1)
    lag = np.arange(nlag) * dt
    z = int(np.argmax(cn < 0))
    return dict(cn=cn, tau_half=float(lag[np.argmax(cn < 0.5)]),
                tau_e=float(lag[np.argmax(cn < 1 / np.e)]),
                tau_int=float(dt * (0.5 + cn[1:z if z > 0 else None].sum())))


def simulate(seed, rtol=1e-4, atol=1e-7):
    a = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=Cdeg)
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                  method="RK45", rtol=rtol, atol=atol)
    rec = measure(r["W"], r["t"]) if r["success"] else None
    if rec is not None:
        rec["acorr"] = acorr_tau(r["W"], r["t"])
    return rec


def plot_fig(d, tag):
    """Three panels, observables vs Delta t, from the saved npz."""
    dts = d["dts"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].errorbar(dts, d["beta_med"], yerr=d["beta_sd"], fmt="o-", ms=5, lw=1.2,
                   color="#1d3557", capsize=3)
    ax[0].set(xlabel=r"observation lag $\Delta t$", ylabel=r"$\beta$")
    QCOL = ["#457b9d", "#e76f51", "#2a9d8f"]
    for i, q in enumerate((2, 3, 4)):
        ax[1].errorbar(dts, d["ratio"][:, q - 1], yerr=d["ratio_err"][:, q - 1], fmt="o-",
                       ms=5, lw=1.2, color=QCOL[i], capsize=3, label=rf"$\zeta_{q}/\zeta_1$")
    ax[1].set(xlabel=r"observation lag $\Delta t$", ylabel=r"$\zeta_q/\zeta_1$")
    ax[1].legend(frameon=False, fontsize=9, loc="center", bbox_to_anchor=(0.7, 0.32))
    ax[2].errorbar(dts, d["exk_med"], yerr=d["exk_sd"], fmt="o-", ms=5, lw=1.2,
                   color="#e76f51", capsize=3)
    ax[2].axhline(3.0, color="0.5", ls="--", lw=1)
    ax[2].text(dts[0], 3.3, "Laplace", fontsize=8, color="0.4")
    ax[2].set(xlabel=r"observation lag $\Delta t$", ylabel="excess kurtosis")
    for a in ax:
        a.axvline(DT_PROD, color="0.6", ls=":", lw=1)
        a.set_xscale("log")
        a.set_xticks(dts)
        a.set_xticklabels([f"{v:g}" if v in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0) else ""
                           for v in dts])
        a.minorticks_off()
    fig.tight_layout()
    fig.savefig(f"data/appendix_robustness{tag}.png", dpi=150)
    print(f"saved data/appendix_robustness{tag}.png")


tag = "_smoke" if SMOKE else ""
if "--replot" in sys.argv:                       # figure-only pass from the saved npz
    d = np.load(f"data/appendix_robustness{tag}.npz")
    plot_fig(d, tag)
    sys.exit(0)

recs = [x for x in Parallel(n_jobs=NJOBS)(delayed(simulate)(s) for s in range(SEEDS))
        if x is not None]
assert recs, "no economies survived"
print(f"survived {len(recs)}/{SEEDS}")

# per-dt aggregates: beta median+sd, quenched ratios mean+sem, exkurt median, vol median
agg = {}
for dt in DTS:
    beta = np.array([r[dt]["beta"] for r in recs])
    Z = np.array([r[dt]["zeta"] for r in recs if r[dt]["zeta"] is not None])
    ratio = Z.mean(0) / Z.mean(0)[0] if Z.size else np.full(4, np.nan)
    ratio_err = (Z / Z[:, :1]).std(0) / np.sqrt(len(Z)) if Z.size else np.full(4, np.nan)
    exk = np.array([r[dt]["exk"] for r in recs])
    agg[dt] = dict(beta_med=float(np.median(beta)), beta_sd=float(beta.std()),
                   ratio=ratio, ratio_err=ratio_err,
                   exk_med=float(np.median(exk)), exk_sd=float(exk.std()),
                   medvol=float(np.median([r[dt]["medvol"] for r in recs])))
    print(f"dt={dt:<5} beta {agg[dt]['beta_med']:.3f}±{agg[dt]['beta_sd']:.3f}  "
          f"ratios " + " ".join(f"{v:.2f}" for v in ratio) +
          f"  exkurt {agg[dt]['exk_med']:.1f}  medvol {agg[dt]['medvol']:.4f}")

# ln S autocorrelation time over the economies
cn = np.mean([r["acorr"]["cn"] for r in recs], axis=0)
taus = {k: np.array([r["acorr"][k] for r in recs]) for k in ("tau_half", "tau_e", "tau_int")}
alag = np.arange(cn.size) * 0.25
print("lnS autocorr  " + "  ".join(f"C({alag[i]:g})={cn[i]:.2f}" for i in (2, 4, 8, 16)) +
      "   " + "  ".join(f"{k} {np.median(v):.2f}±{v.std():.2f}" for k, v in taus.items()))

# solver-tolerance check: tight reruns of the first TOL_SEEDS seeds, production dt
tight = [x for x in Parallel(n_jobs=NJOBS)(
    delayed(simulate)(s, rtol=1e-6, atol=1e-9) for s in range(TOL_SEEDS)) if x is not None]
tol = {}
if tight:
    for name, sel in (("loose", recs[:len(tight)]), ("tight", tight)):
        beta = np.array([r[DT_PROD]["beta"] for r in sel])
        Z = np.array([r[DT_PROD]["zeta"] for r in sel if r[DT_PROD]["zeta"] is not None])
        tol[name] = dict(beta=beta, ratio=Z / Z[:, :1] if Z.size else np.full((0, 4), np.nan),
                         exk=np.array([r[DT_PROD]["exk"] for r in sel]))
    db = np.abs(tol["tight"]["beta"] - tol["loose"]["beta"])
    print(f"tolerance check ({len(tight)} seeds, dt={DT_PROD}): "
          f"max |dbeta| = {db.max():.3f}   "
          f"beta loose {tol['loose']['beta'].round(3)} tight {tol['tight']['beta'].round(3)}")

np.savez(f"data/appendix_robustness{tag}.npz",
         N=N, C=Cdeg, seeds=len(recs), dts=np.array(DTS),
         beta_med=np.array([agg[dt]["beta_med"] for dt in DTS]),
         beta_sd=np.array([agg[dt]["beta_sd"] for dt in DTS]),
         ratio=np.array([agg[dt]["ratio"] for dt in DTS]),
         ratio_err=np.array([agg[dt]["ratio_err"] for dt in DTS]),
         exk_med=np.array([agg[dt]["exk_med"] for dt in DTS]),
         exk_sd=np.array([agg[dt]["exk_sd"] for dt in DTS]),
         medvol=np.array([agg[dt]["medvol"] for dt in DTS]),
         acorr_lag=alag, acorr_cn=cn,
         tau_half=taus["tau_half"], tau_e=taus["tau_e"], tau_int=taus["tau_int"],
         tol_beta_loose=tol.get("loose", {}).get("beta", np.array([])),
         tol_beta_tight=tol.get("tight", {}).get("beta", np.array([])),
         tol_exk_loose=tol.get("loose", {}).get("exk", np.array([])),
         tol_exk_tight=tol.get("tight", {}).get("exk", np.array([])))
print(f"saved data/appendix_robustness{tag}.npz")

plot_fig(np.load(f"data/appendix_robustness{tag}.npz"), tag)

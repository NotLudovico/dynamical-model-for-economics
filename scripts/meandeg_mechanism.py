"""Why hubs are large under competition: quenched luck + selection, parameter-free.

Balance law: time-avg of d lnS_i/dt = 0 for survivors gives (code sign convention:
h here is MINUS the thesis field (WS)_i, so competition means positive h)
    S̄_i = (1 - ḡ) - mean_h_i,   mean_h_i = (alpha @ S̄)_i  (exact, linearity)
Decompose mean_h = penalty + luck:
    penalty_i = (mu/C_eff) * sum_{j in ∂i} S̄_j   (coherent, ∝ k)
    luck_i    = mean_h_i - penalty_i             (disorder draw, std ∝ √k)
Closures per degree bin, each a prediction of survival(k) and survivor size(k):
    Gaussian:  m = F - <penalty>, s = std(luck), P = Φ(m/s), E[S̄] = m + s φ/Φ
    empirical: margin = F - mean_h over ALL firms, no distributional assumption
    dynamic:   survive iff margin > c * std_t(h), c calibrated to aggregate rate only
Ablation mu=0: no competitive mean -> the economy condenses (survival collapses).

Run from repo root: python3 scripts/meandeg_mechanism.py
Env: MM_SEEDS (default 10), MM_N (default 8000), MM_NJOBS (default 6).
"""
import os, sys
while not os.path.isdir("relative_glv") and os.path.dirname(os.getcwd()) != os.getcwd():
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy.stats import norm, kurtosis
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate

MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N = int(os.environ.get("MM_N", 8000))
Cdeg = max(2, round(100 * N / 4000))               # C ∝ N anchor
SEEDS = int(os.environ.get("MM_SEEDS", 10))
MU0_SEEDS = 3
TMAX, N_EVAL, LATE = 250.0, 700, (180.0, 240.0)
NJOBS = int(os.environ.get("MM_NJOBS", 6))
NBIN = 8

print(f"mechanism  mean-degree pl2.5  N={N} C={Cdeg}  {SEEDS} seeds (+{MU0_SEEDS} mu=0)  "
      f"mu={MU} sigma={SIGMA} lam={LAM}")


def run(mu, seed):
    a = coupling(N, mu, SIGMA, kind="powerlaw", seed=seed, mean_degree=Cdeg)
    k = np.diff(a.indptr).astype(float)
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                  method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    t, W = r["t"], r["W"]
    win = (t >= LATE[0]) & (t <= LATE[1])
    Wl = W[:, win]
    live = Wl.min(1) > 1e-6
    Sbar = (N * Wl).mean(1)
    mean_h = a @ Sbar                                  # exact time-avg field
    adj = a.copy(); adj.data = np.ones_like(adj.data)
    C_eff = k.mean()
    penalty = (mu / C_eff) * (adj @ Sbar)
    luck = mean_h - penalty
    gbar = np.gradient(r["lnM"], t)[win].mean()        # ḡ = time-avg <f>
    F = 1.0 - gbar

    # 1. balance law on survivors: S̄ vs F - mean_h, slope 1, intercept 0
    x, y = (F - mean_h)[live], Sbar[live]
    if x.size < 10:
        return dict(mu=mu, seed=seed, live_frac=float(live.mean()))
    sl, ic = np.polyfit(x, y, 1)
    r2 = 1 - np.var(y - (sl * x + ic)) / np.var(y)

    # dynamic culling: survive iff margin > thr + c * std_t(h); one global c
    # calibrated to the AGGREGATE survival rate, k-shape stays a prediction
    margin = F - mean_h                                # balance-law size of EVERY firm
    thr = 0.008                                        # live cut w>1e-6 in S units
    st = (N * (a @ Wl)).std(1)
    lo, hi = 0.0, 10.0
    for _ in range(40):
        c = (lo + hi) / 2
        if (margin > thr + c * st).mean() > live.mean():
            lo = c
        else:
            hi = c
    dyn = margin > thr + c * st
    jac = (dyn & live).sum() / (dyn | live).sum()

    # 2-3. per-degree-bin channels + the three closures
    edges = np.logspace(np.log10(k.min()), np.log10(k.max() + 1), NBIN + 1)
    mid = np.sqrt(edges[:-1] * edges[1:])
    rows = []
    for b in range(NBIN):
        sel = (np.digitize(k, edges) - 1) == b
        if sel.sum() < 30:
            rows.append([np.nan] * 11); continue
        m = F - penalty[sel].mean()
        s = luck[sel].std()
        a_ = m / s
        emp = margin[sel] > thr
        dy = dyn[sel]
        rows.append([penalty[sel].mean(), s, live[sel].mean(), norm.cdf(a_),
                     Sbar[sel & live].mean() if (sel & live).sum() else np.nan,
                     m + s * norm.pdf(a_) / norm.cdf(a_),
                     emp.mean(),
                     margin[sel][emp].mean() if emp.sum() else np.nan,
                     kurtosis(luck[sel]),
                     dy.mean(),
                     margin[sel][dy].mean() if dy.sum() else np.nan])
    rows = np.array(rows)  # pen, luckstd, surv, surv_gauss, S_meas, S_gauss, surv_emp, S_emp, exkurt, surv_dyn, S_dyn
    ok = np.isfinite(rows[:, :8]).all(1)
    lsl = lambda ycol: np.polyfit(np.log(mid[ok]), np.log(rows[ok, ycol]), 1)[0]
    sub = np.random.default_rng(0).choice(x.size, min(x.size, 3000), replace=False)
    return dict(mu=mu, seed=seed, live_frac=float(live.mean()), slope=sl, ic=ic, r2=r2,
                F=F, c=c, jac=jac, bal_x=x[sub], bal_y=y[sub],
                pen_slope=lsl(0), luck_slope=lsl(1),
                mid=mid, rows=rows, ok=ok,
                delta_meas=lsl(4), delta_gauss=lsl(5), delta_emp=lsl(7), delta_dyn=lsl(10),
                surv_lo=rows[ok, 2][0], surv_hi=rows[ok, 2][-1])


jobs = [(MU, s) for s in range(SEEDS)] + [(0.0, s) for s in range(MU0_SEEDS)]
res = [x for x in Parallel(n_jobs=NJOBS)(delayed(run)(m, s) for m, s in jobs) if x]
comp = [x for x in res if x["mu"] == MU and "slope" in x]
mu0 = [x for x in res if x["mu"] == 0.0]

for x in comp:
    print(f"mu={x['mu']:4.2f} seed {x['seed']}:  balance slope={x['slope']:+.3f} ic={x['ic']:+.3f} "
          f"R2={x['r2']:.3f}  |  penalty~k^{x['pen_slope']:.2f}  luck~k^{x['luck_slope']:.2f}  |  "
          f"surv {x['surv_lo']:.2f}->{x['surv_hi']:.2f}  |  delta meas={x['delta_meas']:+.2f}  "
          f"gauss={x['delta_gauss']:+.2f}  emp={x['delta_emp']:+.2f}  dyn={x['delta_dyn']:+.2f}  "
          f"(c={x['c']:.2f}, jaccard={x['jac']:.2f})")
for x in mu0:
    print(f"mu=0.00 seed {x['seed']}:  survival {x['live_frac']:.4f} (condensed)")

med = lambda key: (np.median([x[key] for x in comp]), np.std([x[key] for x in comp]))
print("\nmedians over competitive seeds:")
for key in ["slope", "r2", "pen_slope", "luck_slope", "surv_lo", "surv_hi",
            "delta_meas", "delta_gauss", "delta_emp", "delta_dyn", "c", "jac"]:
    m_, s_ = med(key)
    print(f"  {key:12s} {m_:+.3f} ± {s_:.3f}")

np.savez("data/meandeg_mechanism.npz",
         N=N, C=Cdeg, mu=MU, sigma=SIGMA, lam=LAM, seeds=len(comp),
         mid=comp[0]["mid"], rows=np.stack([x["rows"] for x in comp]),
         mu0_live_frac=[x["live_frac"] for x in mu0],
         **{key: [x[key] for x in comp] for key in
            ["slope", "ic", "r2", "F", "c", "jac", "pen_slope", "luck_slope",
             "delta_meas", "delta_gauss", "delta_emp", "delta_dyn", "surv_lo", "surv_hi"]})

# ---- diagnostic figure (data record, not a thesis figure): seed 0 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM, CG, CE, CD = "#222222", "#9aa0a6", "#4c8cbf", "#c0504d"   # measured, gauss, empirical, dynamic
x = comp[0]
mid, rows, ok = x["mid"], x["rows"], x["ok"]
m, r_ = mid[ok], rows[ok]

fig, ax = plt.subplots(2, 2, figsize=(9.5, 8), constrained_layout=True)

a0 = ax[0, 0]
a0.loglog(x["bal_x"], x["bal_y"], ".", ms=3, alpha=0.3, color=CG)
lim = [x["bal_y"].min() * 0.7, x["bal_y"].max() * 1.5]
a0.loglog(lim, lim, "-", lw=1, color=CM)
a0.text(0.05, 0.9, r"$\bar S_i=(1-\bar g)-\langle h_i\rangle$" + f"\nslope {x['slope']:.3f},  $R^2$={x['r2']:.3f}",
        transform=a0.transAxes, fontsize=9, va="top")
a0.set(xlabel=r"balance prediction  $(1-\bar g)-\langle h_i\rangle$", ylabel=r"measured size $\bar S_i$",
       title="a)  balance law (survivors)")

a1 = ax[0, 1]
a1.loglog(m, r_[:, 0], "o-", color=CD, label=r"penalty $\langle$coherent field$\rangle$")
a1.loglog(m, r_[:, 1], "s-", color=CE, label="luck spread (std of disorder field)")
a1.loglog(m, r_[0, 0] * (m / m[0]), "--", lw=1, color=CG)
a1.loglog(m, r_[0, 1] * (m / m[0]) ** 0.5, "--", lw=1, color=CG)
a1.text(m[-1], r_[0, 0] * (m[-1] / m[0]) * 1.25, r"$\propto k$", fontsize=9, color=CG, ha="right")
a1.text(m[-1], r_[0, 1] * (m[-1] / m[0]) ** 0.5 * 0.7, r"$\propto \sqrt{k}$", fontsize=9, color=CG, ha="right")
a1.set(xlabel="degree $k$", ylabel="field component", title="b)  what degree does to the field")
a1.legend(fontsize=8, loc="upper left")

a2 = ax[1, 0]
a2.semilogx(m, r_[:, 3], "--", color=CG, label="Gaussian closure")
a2.semilogx(m, r_[:, 6], ":", lw=2, color=CE, label="+ fat-tailed luck (empirical)")
a2.semilogx(m, r_[:, 9], "-", color=CD, label=rf"+ churn culling ($c$={x['c']:.2f})")
a2.semilogx(m, r_[:, 2], "o", color=CM, label="measured", zorder=5)
a2.set(xlabel="degree $k$", ylabel="survival probability", title="c)  survival: competition kills hubs twice")
a2.legend(fontsize=8)

a3 = ax[1, 1]
a3.loglog(m, r_[:, 5], "--", color=CG, label=rf"Gaussian ($\delta$={x['delta_gauss']:.2f})")
a3.loglog(m, r_[:, 7], ":", lw=2, color=CE, label=rf"+ fat tails ($\delta$={x['delta_emp']:.2f})")
a3.loglog(m, r_[:, 10], "-", color=CD, label=rf"+ churn culling ($\delta$={x['delta_dyn']:.2f})")
a3.loglog(m, r_[:, 4], "o", color=CM, label=rf"measured ($\delta$={x['delta_meas']:.2f})", zorder=5)
a3.set(xlabel="degree $k$", ylabel=r"survivor mean size $\bar S$", title="d)  size of survivors: the lucky tail")
a3.legend(fontsize=8, loc="upper left")

fig.suptitle("Why hubs are large under competition — quenched luck + selection "
             f"(mean-degree pl2.5, N={N}, seed {x['seed']})", fontsize=11)
fig.savefig("data/meandeg_mechanism.png", dpi=150)
print("saved data/meandeg_mechanism.npz/.png")

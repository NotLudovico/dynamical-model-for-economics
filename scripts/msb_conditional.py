"""MSB's three DISCRIMINATING tests on the OWN-DEGREE relative GLV.

size_volatility() + tent_stats() already cover beta and the unconditional tent -- bars that
granular models also clear. The three facts that actually FALSIFY granular models
(Moran-Santos-Bouchaud 2024, Figs 3-7) are conditional-on-size; this script measures them on the
own-degree model (kind='powerlaw_owndeg') at the locked operating point (mu=1.76, sigma=1.75):

  D1  rescaled-volatility collapse + tail: sigma_i/sigmabar(S) should be size-INDEPENDENT with a
      THIN tail (MSB data: Inverse-Gamma, tail exponent ~ -(1+4.6)).
  D2  q-dependent moment scaling E[sigma^q|S] ~ S^-c_q. Granular: c_q EQUAL for all q>=2.
      MSB data: c_1..c_4 ~ (0.20, 0.39, 0.51, 0.58) -- q-dependent (thin-tail signature).
  D3  no Gaussianization with size: un-mix growth by each firm's own vol, ghat=(g-gbar)/sigma_i.
      Granular/CLT: large firms -> Gaussian (excess kurtosis of ghat -> 0 with size).
      MSB data: ghat stays fat-tailed at ALL sizes; large firms are if anything LESS Gaussian.

Everything is measured on the DECLINE branch (S past the sigma(S) peak), the same branch beta is
defined on -- the small-S plateau is a survival-floor artefact. Pools several economies.

    uv run python scripts/msb_conditional.py [--smoke]
"""
import sys, os, time, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate, growth_rate
from relative_glv.msb import size_volatility, tent_stats

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SMOKE = "--smoke" in sys.argv
MU, SIGMA, LAM = 1.76, 1.75, 1e-3                 # locked operating point (own-degree phase diagram)
N, SEEDS = (800, 3) if SMOKE else (4000, 40)
TMAX, N_EVAL = (120.0, 600) if SMOKE else (400.0, 800)
LATE, DT = ((95.0, 118.0) if SMOKE else (320.0, 390.0)), 0.5
N_JOBS, TIMEOUT = 6, 120
NB = 20                                            # size bins for the sigma(S) curve


def _alarm(_s, _f):
    raise TimeoutError()


def simulate(seed):
    """One own-degree economy: per-firm size, vol, growth rows (survivors), + anchors."""
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(TIMEOUT)
    try:
        a = coupling(N, MU, SIGMA, kind="powerlaw_owndeg", seed=seed)
        r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                      method="RK45", rtol=1e-4, atol=1e-7)
        signal.alarm(0)
    except TimeoutError:
        return None
    if not r["success"]:
        return None
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT, n_bins=NB)
    if sv["growth"].size == 0 or not np.isfinite(sv["beta"]):
        return None
    g_eff = growth_rate(r["t"], r["lnM"], frac=0.3)
    return dict(Sbar=sv["Sbar"], vol=sv["vol"], growth=sv["growth"], beta=sv["beta"],
                r2=sv["r2"], g_eff=g_eff, bowley=tent_stats(sv["growth"])["bowley"],
                exk=tent_stats(sv["growth"])["exkurt"])


def decline_peak(Sbar, vol, nb=NB):
    """Bin by size, median vol per bin, return (bin_S, bin_vol_median, peak_index)."""
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), nb)
    bx = np.array([Sbar[o][p].mean() for p in P])
    by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]
    return bx, by, int(np.argmax(by))


if __name__ == "__main__":
    t0 = time.time()
    print(f"[msb_cond] own-degree  mu={MU} sigma={SIGMA} lam={LAM}  N={N} x {SEEDS} seeds  "
          f"window={LATE} dt={DT}", flush=True)
    recs = [r for r in Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(simulate)(s) for s in range(SEEDS)) if r is not None]
    assert recs, "no economies survived integration"
    Sbar = np.concatenate([r["Sbar"] for r in recs])
    vol = np.concatenate([r["vol"] for r in recs])
    growth = np.vstack([r["growth"] for r in recs])           # (firms, increments) aligned to Sbar/vol
    live = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
    Sbar, vol, growth = Sbar[live], vol[live], growth[live]
    beta_med = float(np.median([r["beta"] for r in recs]))
    geff_med = float(np.median([r["g_eff"] for r in recs]))
    bowley_med = float(np.median([r["bowley"] for r in recs]))
    exk_med = float(np.median([r["exk"] for r in recs]))
    print(f"  pooled firms={Sbar.size}  beta(median per-economy)={beta_med:.3f}"
          f"  g_eff~{geff_med:+.2f}  bowley~{bowley_med:+.2f}  exk~{exk_med:.1f}\n", flush=True)

    bx, by, pk = decline_peak(Sbar, vol)
    Scut = bx[pk]
    dec = Sbar > Scut                                          # decline-branch firms
    print(f"sigma(S) peak at S={Scut:.3g} (bin {pk}/{len(bx)}); decline-branch firms={int(dec.sum())}\n")

    # ---- D2: q-dependent moment scaling on the decline branch ---------------------------------
    print("D2  E[sigma^q|S] ~ S^-c_q on decline branch   (granular: c_q EQUAL for q>=2; "
          "MSB data 0.20,0.39,0.51,0.58)")
    o = np.argsort(Sbar[dec]); Sd, vd = Sbar[dec][o], vol[dec][o]
    P = np.array_split(np.arange(Sd.size), max(6, NB - pk))
    bxd = np.array([Sd[p].mean() for p in P])
    cq, D2M = [], []
    for q in (1, 2, 3, 4):
        byq = np.array([np.mean(vd[p] ** q) for p in P]); D2M.append(byq)
        m = (bxd > 0) & (byq > 0)
        c = -np.polyfit(np.log10(bxd[m]), np.log10(byq[m]), 1)[0]
        cq.append(c)
        print(f"     q={q}:  c_q = {c:+.3f}    (proportional -q*beta = {q*cq[0]:+.3f})")
    D2M = np.array(D2M)

    # ---- D1: rescaled-volatility collapse + tail (decline branch) ------------------------------
    sbar_S = np.interp(Sbar[dec], bxd, np.array([np.mean(vd[p]) for p in P]))
    rr = vol[dec] / sbar_S
    lo, mid, hi = np.array_split(np.argsort(Sbar[dec]), 3)
    print("\nD1  rescaled vol sigma/sigmabar(S) (decline branch): size-collapse + tail")
    for nm, idx in (("small", lo), ("mid", mid), ("large", hi)):
        print(f"     {nm:>5} S: median={np.median(rr[idx]):.2f}  P90={np.percentile(rr[idx],90):.2f}  "
              f"P99={np.percentile(rr[idx],99):.2f}")
    print(f"     pooled tail P99/median = {np.percentile(rr,99)/np.median(rr):.2f}  "
          f"(thin/Inverse-Gamma if collapse holds and tail bounded)")

    # ---- D3: un-mix growth by own vol; Gaussianize with size? ----------------------------------
    gres = growth - growth.mean(1, keepdims=True)
    ghat = gres / vol[:, None]                                 # (g - gbar)/sigma_i per firm
    pooled = ghat[dec].ravel(); pooled = pooled[np.isfinite(pooled)]
    raw = gres[dec].ravel(); raw = raw[np.isfinite(raw)]
    raw = raw / (np.sqrt(np.pi / 2) * np.abs(raw).mean())      # homogeneous rescale
    k_homog, k_unmix = float(kurtosis(raw)), float(kurtosis(pooled))
    print("\nD3  un-mixed growth ghat=(g-gbar)/sigma_i (decline branch)   (Gaussian -> exkurt 0)")
    print(f"     pooled excess kurtosis: homogeneous-rescale={k_homog:.2f}  "
          f"per-firm-unmixed={k_unmix:.2f}")
    od = np.argsort(Sbar[dec]); Q = np.array_split(od, 8)
    bxk = np.array([Sbar[dec][q].mean() for q in Q])
    bk = np.array([kurtosis(ghat[dec][q].ravel()) for q in Q])
    trend = np.polyfit(np.log10(bxk), bk, 1)[0]
    print("     excess kurtosis of ghat vs size (8 bins, small->large):")
    print("     " + "  ".join(f"{v:.1f}" for v in bk))
    print(f"     trend d(exkurt)/d(log S) = {trend:+.2f}   "
          f"(<0 = MORE Gaussian with size [granular/CLT]; >=0 = MSB: stays/gets fatter)")

    # ---- save raw arrays so the notebook can replot (dataset-oriented) -------------------------
    grp = np.empty(rr.size, np.int8); grp[lo] = 0; grp[mid] = 1; grp[hi] = 2
    rng_ss = np.random.default_rng(0)
    def _sub(x, n=50000):
        x = x[np.isfinite(x)]
        return (x if x.size <= n else rng_ss.choice(x, n, replace=False)).astype(np.float32)
    np.savez(os.path.join(DATA, "msb_conditional.npz"),
             N=N, seeds=SEEDS, n_firms=int(Sbar.size), Scut=float(Scut),
             beta=beta_med, g_eff=geff_med, bowley=bowley_med, exk=exk_med,
             d2_S=bxd, d2_M=D2M, cq=np.array(cq),
             d1_r=rr.astype(np.float32), d1_group=grp,
             d1_p99_med=float(np.percentile(rr, 99) / np.median(rr)),
             d3_S=bxk, d3_kurt=bk, d3_trend=float(trend),
             d3_k_homog=k_homog, d3_k_unmix=k_unmix,
             ghat_small=_sub(ghat[dec][lo].ravel()),
             ghat_mid=_sub(ghat[dec][mid].ravel()),
             ghat_large=_sub(ghat[dec][hi].ravel()))
    print(f"saved {os.path.join(DATA, 'msb_conditional.npz')}")

    # ---- figure -------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for q, col in zip((1, 2, 3, 4), ("#1d3557", "#457b9d", "#e76f51", "#2a9d8f")):
        ax[0].loglog(bxd, D2M[q - 1], "o-", ms=4, color=col, label=f"q={q} (c={cq[q-1]:.2f})")
    ax[0].set(xlabel=r"$\bar S$", ylabel=r"$E[\sigma^q|S]$", title="D2 moment scaling (decline branch)")
    ax[0].legend(fontsize=8)
    for nm, idx, col in (("small", lo, "#1d3557"), ("mid", mid, "#457b9d"), ("large", hi, "#e76f51")):
        h, e = np.histogram(rr[idx], bins=50, range=(0, 5), density=True); m = h > 0
        ax[1].semilogy(0.5 * (e[:-1] + e[1:])[m], h[m], color=col, label=nm)
    ax[1].set(xlabel=r"$\sigma/\bar\sigma(S)$", ylabel="PDF", title="D1 rescaled-vol collapse")
    ax[1].legend(fontsize=8)
    for nm, idx, col in (("small", lo, "#1d3557"), ("mid", mid, "#457b9d"), ("large", hi, "#e76f51")):
        gg = ghat[dec][idx].ravel(); gg = gg[np.isfinite(gg)]
        h, e = np.histogram(gg, bins=70, range=(-8, 8), density=True); m = h > 0
        ax[2].semilogy(0.5 * (e[:-1] + e[1:])[m], h[m], color=col, label=nm)
    zz = np.linspace(-8, 8, 200)
    ax[2].semilogy(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="Gaussian")
    ax[2].set(xlabel=r"$\hat g=(g-\bar g)/\sigma_i$", ylabel="PDF", title="D3 un-mixed growth by size")
    ax[2].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(DATA, "msb_conditional.png")
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}   ({(time.time()-t0)/60:.1f} min)")

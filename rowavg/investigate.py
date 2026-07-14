"""Investigate the ROW-AVERAGE own-degree variant  (kind='powerlaw_rowavg').

Exploratory model: the interaction is the AVERAGE of O(1) couplings over each firm's
k_i neighbours, a_ij = (mu + sigma z)/k_i. Mean field mu (like own-degree) but the
disorder field self-averages as sigma/sqrt(k_i) -- hubs are QUIETER, so node degree
DRIVES the size-volatility law (unlike the banked own-degree model, whose field is
degree-independent and whose beta is pure self-limitation).

At the locked operating point (mu=1.76, sigma=1.75, lam=1e-3), with the SAME definitions
as scripts/msb_conditional.py so numbers are directly comparable:
  beta   size-volatility exponent  sigma(S) ~ S^-beta
  tent   growth-rate distribution g=Dln S: Bowley skew + excess kurtosis (Laplace=3, Gaussian=0)
  D1     rescaled-vol collapse sigma_i/sigmabar(S): size-independent? tail bounded?
  D2     q-moment scaling E[sigma^q|S] ~ S^-c_q: c_q q-dependent (thin tail) or equal (granular)?
  D3     un-mixed growth ghat=(g-gbar)/sigma_i: stays fat-tailed with size (MSB) or Gaussianises?
Prints the banked own-degree numbers (data/msb_conditional.npz) alongside, where present.

    uv run python rowavg/investigate.py [--smoke]
"""
import sys, os, time, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import kurtosis, laplace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate, growth_rate
from relative_glv.msb import size_volatility, tent_stats, rescale

HERE = os.path.dirname(os.path.abspath(__file__))
BANKED = os.path.join(os.path.dirname(HERE), "data", "msb_conditional.npz")   # own-degree reference
SMOKE = "--smoke" in sys.argv


def _arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


KIND = _arg("--kind", "powerlaw_rowavg", str)   # also 'fc', 'powerlaw' for comparison


# NB: own-degree's locked point (sigma=1.75) is FROZEN for this variant -- the row-average
# noise is sqrt(k)-suppressed, so the fluctuating band sits at much larger sigma. sigma~12 is
# the only band with genuine fluctuation (and it is still partly frozen + marginally
# shrinking, g_eff<0). Override with --mu/--sigma/--N/--seeds to explore.
MU = _arg("--mu", 1.76)
SIGMA = _arg("--sigma", 12.0)
LAM = 1e-3
N = int(_arg("--N", 800 if SMOKE else 4000))
SEEDS = int(_arg("--seeds", 3 if SMOKE else 8))
TMAX, N_EVAL = (120.0, 600) if SMOKE else (400.0, 800)
LATE, DT = ((95.0, 118.0) if SMOKE else (320.0, 390.0)), 0.5
N_JOBS, TIMEOUT, NB = 6, 300, 20


def _alarm(_s, _f):
    raise TimeoutError()


def simulate(seed):
    """One row-average economy: per-survivor size, vol, growth rows, + anchors."""
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(TIMEOUT)
    try:
        a = coupling(N, MU, SIGMA, kind=KIND, seed=seed)
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
    ts = tent_stats(sv["growth"])
    return dict(Sbar=sv["Sbar"], vol=sv["vol"], growth=sv["growth"], beta=sv["beta"],
                r2=sv["r2"], g_eff=growth_rate(r["t"], r["lnM"], frac=0.3),
                bowley=ts["bowley"], exk=ts["exkurt"])


def main():
    t0 = time.time()
    print(f"[rowavg] kind={KIND}  mu={MU} sigma={SIGMA} lam={LAM}  N={N} x {SEEDS} seeds  "
          f"window={LATE} dt={DT}", flush=True)
    recs = [r for r in Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(simulate)(s) for s in range(SEEDS)) if r is not None]
    assert recs, "no economies survived integration"
    Sbar = np.concatenate([r["Sbar"] for r in recs])
    vol = np.concatenate([r["vol"] for r in recs])
    growth = np.vstack([r["growth"] for r in recs])            # (firms, increments), aligned to Sbar/vol
    live = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
    Sbar, vol, growth = Sbar[live], vol[live], growth[live]
    beta_med = float(np.median([r["beta"] for r in recs]))
    geff_med = float(np.median([r["g_eff"] for r in recs]))
    bowley_med = float(np.median([r["bowley"] for r in recs]))
    exk_med = float(np.median([r["exk"] for r in recs]))
    print(f"  pooled firms={Sbar.size}  beta(median per-economy)={beta_med:.3f}  "
          f"g_eff~{geff_med:+.2f}  bowley~{bowley_med:+.2f}  exk~{exk_med:.1f}\n", flush=True)

    # decline branch (S past the sigma(S) peak), the branch beta is defined on
    o = np.argsort(Sbar); P0 = np.array_split(np.arange(o.size), NB)
    bx0 = np.array([Sbar[o][p].mean() for p in P0])
    by0 = np.array([np.median(vol[o][p]) for p in P0])
    m0 = (bx0 > 0) & (by0 > 0); bx0, by0 = bx0[m0], by0[m0]
    pk = int(np.argmax(by0)); Scut = bx0[pk]
    dec = Sbar > Scut
    print(f"sigma(S) peak at S={Scut:.3g} (bin {pk}/{len(bx0)}); decline-branch firms={int(dec.sum())}\n")

    # ---- D2: q-dependent moment scaling on the decline branch -----------------------------------
    o = np.argsort(Sbar[dec]); Sd, vd = Sbar[dec][o], vol[dec][o]
    P = np.array_split(np.arange(Sd.size), max(6, NB - pk))
    bxd = np.array([Sd[p].mean() for p in P])
    cq, D2M = [], []
    print("D2  E[sigma^q|S] ~ S^-c_q (decline)  (granular: c_q EQUAL for q>=2; MSB 0.20,0.39,0.51,0.58)")
    for q in (1, 2, 3, 4):
        byq = np.array([np.mean(vd[p] ** q) for p in P]); D2M.append(byq)
        mm = (bxd > 0) & (byq > 0)
        c = -np.polyfit(np.log10(bxd[mm]), np.log10(byq[mm]), 1)[0]; cq.append(c)
        print(f"     q={q}:  c_q = {c:+.3f}    (proportional q*c_1 = {q * cq[0]:+.3f})")
    D2M = np.array(D2M)

    # ---- D1: rescaled-volatility collapse + tail (decline branch) -------------------------------
    sbar_S = np.interp(Sbar[dec], bxd, np.array([np.mean(vd[p]) for p in P]))
    rr = vol[dec] / sbar_S
    lo, mid, hi = np.array_split(np.argsort(Sbar[dec]), 3)
    print("\nD1  rescaled vol sigma/sigmabar(S) (decline): size-collapse + tail")
    for nm, idx in (("small", lo), ("mid", mid), ("large", hi)):
        print(f"     {nm:>5} S: median={np.median(rr[idx]):.2f}  P90={np.percentile(rr[idx], 90):.2f}  "
              f"P99={np.percentile(rr[idx], 99):.2f}")
    p99med = float(np.percentile(rr, 99) / np.median(rr))
    print(f"     pooled tail P99/median = {p99med:.2f}")

    # ---- D3: un-mix growth by own vol; Gaussianize with size? -----------------------------------
    gres = growth - growth.mean(1, keepdims=True)
    ghat = gres / vol[:, None]
    raw = gres[dec].ravel(); raw = raw[np.isfinite(raw)]
    raw = raw / (np.sqrt(np.pi / 2) * np.abs(raw).mean())
    gh_pool = ghat[dec].ravel(); gh_pool = gh_pool[np.isfinite(gh_pool)]
    k_homog, k_unmix = float(kurtosis(raw)), float(kurtosis(gh_pool))
    od = np.argsort(Sbar[dec]); Q = np.array_split(od, 8)
    bxk = np.array([Sbar[dec][q].mean() for q in Q])
    bk = np.array([kurtosis(ghat[dec][q].ravel()) for q in Q])
    trend = float(np.polyfit(np.log10(bxk), bk, 1)[0])
    print("\nD3  un-mixed growth ghat=(g-gbar)/sigma_i (decline)   (Gaussian -> exkurt 0)")
    print(f"     pooled excess kurtosis: homogeneous-rescale={k_homog:.2f}  per-firm-unmixed={k_unmix:.2f}")
    print("     ghat exkurt vs size (8 bins, small->large): " + "  ".join(f"{v:.1f}" for v in bk))
    print(f"     trend d(exkurt)/d(log S) = {trend:+.2f}   "
          f"(<0 = MORE Gaussian with size [granular]; >=0 = MSB: stays/gets fatter)")

    # ---- banked own-degree reference, side by side ----------------------------------------------
    if os.path.exists(BANKED):
        b = np.load(BANKED, allow_pickle=True)
        print("\n--- banked OWN-DEGREE reference (data/msb_conditional.npz) ---")
        print(f"     beta={float(b['beta']):.3f}  exk={float(b['exk']):.1f}  "
              f"c_q={np.array(b['cq']).round(2).tolist()}  D1 P99/med={float(b['d1_p99_med']):.2f}  "
              f"D3 trend={float(b['d3_trend']):+.2f}")

    # ---- save raw arrays (dataset-oriented) -----------------------------------------------------
    np.savez(os.path.join(HERE, "investigate.npz"),
             kind=KIND, N=N, seeds=SEEDS, n_firms=int(Sbar.size), Scut=float(Scut),
             beta=beta_med, g_eff=geff_med, bowley=bowley_med, exk=exk_med,
             bin_S=bx0, bin_vol=by0, d2_S=bxd, d2_M=D2M, cq=np.array(cq),
             d1_r=rr.astype(np.float32), d1_p99_med=p99med,
             d3_S=bxk, d3_kurt=bk, d3_trend=trend, d3_k_homog=k_homog, d3_k_unmix=k_unmix,
             tent_g=rescale(growth).astype(np.float32))

    # ---- figure: size-vol | tent | D2 | D1 | D3 -------------------------------------------------
    zz = np.linspace(-8, 8, 200)
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.3))
    ax[0].loglog(bx0, by0, "o-", ms=4); ax[0].axvline(Scut, color="grey", ls=":")
    ax[0].set(xlabel=r"$\bar S$", ylabel=r"$\sigma(S)$", title=f"size-volatility  $\\beta$={beta_med:.2f}")
    z = rescale(growth)
    h, e = np.histogram(z, bins=120, range=(-8, 8), density=True); mm = h > 0
    ax[1].semilogy(0.5 * (e[:-1] + e[1:])[mm], h[mm], lw=1, label="model")
    ax[1].semilogy(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="Gaussian")
    ax[1].semilogy(zz, laplace.pdf(zz, scale=np.sqrt(2 / np.pi)), "r--", lw=1, label="Laplace")
    ax[1].set(xlabel=r"rescaled $g$", ylabel="PDF", title=f"growth tent  exk={exk_med:.1f}",
              ylim=(1e-4, 1)); ax[1].legend(fontsize=8)
    for q, col in zip((1, 2, 3, 4), ("#1d3557", "#457b9d", "#e76f51", "#2a9d8f")):
        ax[2].loglog(bxd, D2M[q - 1], "o-", ms=3, color=col, label=f"q={q} (c={cq[q - 1]:.2f})")
    ax[2].set(xlabel=r"$\bar S$", ylabel=r"$E[\sigma^q|S]$", title="D2 moment scaling"); ax[2].legend(fontsize=7)
    for nm, idx, col in (("small", lo, "#1d3557"), ("mid", mid, "#457b9d"), ("large", hi, "#e76f51")):
        h, e = np.histogram(rr[idx], bins=50, range=(0, 5), density=True); mm = h > 0
        ax[3].semilogy(0.5 * (e[:-1] + e[1:])[mm], h[mm], color=col, label=nm)
    ax[3].set(xlabel=r"$\sigma/\bar\sigma(S)$", ylabel="PDF", title="D1 rescaled-vol collapse")
    ax[3].legend(fontsize=8)
    for nm, idx, col in (("small", lo, "#1d3557"), ("mid", mid, "#457b9d"), ("large", hi, "#e76f51")):
        gg = ghat[dec][idx].ravel(); gg = gg[np.isfinite(gg)]
        h, e = np.histogram(gg, bins=70, range=(-8, 8), density=True); mm = h > 0
        ax[4].semilogy(0.5 * (e[:-1] + e[1:])[mm], h[mm], color=col, label=nm)
    ax[4].semilogy(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="Gaussian")
    ax[4].set(xlabel=r"$\hat g=(g-\bar g)/\sigma_i$", ylabel="PDF", title="D3 un-mixed growth by size")
    ax[4].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(HERE, "investigate.png")
    plt.savefig(out, dpi=120)
    print(f"\nsaved {os.path.join(HERE, 'investigate.npz')}\n      {out}   ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

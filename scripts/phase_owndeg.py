"""(sigma, mu) phase diagram of the own-degree relative GLV (the model we bank).

Maps the four regimes on the disorder x competition plane at fixed immigration
lambda: FROZEN (shares relax to a fixed point), FLUCTUATING+GROWING (the MSB
region -- coloured by the size-volatility exponent beta), SHRINKING (g_eff<0), and
DIVERGENT (too violent to integrate). Compute once -> data/phase_owndeg.npz, then
replot freely.

    uv run python scripts/phase_owndeg.py            # compute + plot
    uv run python scripts/phase_owndeg.py --plot     # replot from the npz
    uv run python scripts/phase_owndeg.py --smoke
"""
import sys, os, time, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate, growth_rate
from relative_glv.msb import size_volatility

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N, LAM = 4000, 1e-3
TMAX, N_EVAL, LATE, DT = 400.0, 800, (320.0, 390.0), 0.5
SIGMAS = np.round(np.linspace(1.40, 1.95, 8), 3)
MUS = np.round(np.linspace(0.0, 2.5, 8), 3)
SEEDS, N_JOBS, TIMEOUT = 6, 6, 120
OP = (1.76, 1.75)                                    # operating point (mu, sigma)
FREEZE_CHURN = 0.02                                  # churn below this = frozen


class _TO(Exception):
    pass


def _alarm(_s, _f):
    raise _TO()


def run_one(sigma, mu, seed):
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(TIMEOUT)
    try:
        a = coupling(N, mu, sigma, kind="powerlaw_owndeg", seed=seed)
        r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                      method="RK45", rtol=1e-4, atol=1e-7)
        signal.alarm(0)
    except _TO:
        return [np.nan] * 5
    if not r["success"]:
        return [np.nan] * 5
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT)
    return [float(sv["growth"].std()), growth_rate(r["t"], r["lnM"], frac=0.3),
            sv["vol"].size / N, float(sv["beta"]), float(sv["r2"])]


def compute():
    t0 = time.time()
    nS, nM = len(SIGMAS), len(MUS)
    # raw[i,j,seed,metric], metrics = churn, g_eff, surv, beta, r2
    raw = np.full((nS, nM, SEEDS, 5), np.nan)
    print(f"[phase] own-degree N={N} lam={LAM}  {nS}x{nM} grid x {SEEDS} seeds", flush=True)
    for i, s in enumerate(SIGMAS):
        for j, m in enumerate(MUS):
            res = Parallel(n_jobs=N_JOBS)(delayed(run_one)(s, m, k) for k in range(SEEDS))
            raw[i, j] = np.array(res)
        print(f"  sigma={s:.3f} done  ({(time.time()-t0)/60:.1f} min)", flush=True)
    np.savez(os.path.join(DATA, "phase_owndeg.npz"),
             sigmas=SIGMAS, mus=MUS, seeds=SEEDS, N=N, lam=LAM, op=np.array(OP), raw=raw)
    print(f"[phase] saved data/phase_owndeg.npz  ({(time.time()-t0)/60:.1f} min)", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    d = np.load(os.path.join(DATA, "phase_owndeg.npz"))
    sig, mus, raw, op = d["sigmas"], d["mus"], d["raw"], d["op"]
    nS, nM = len(sig), len(mus)
    okfrac = np.isfinite(raw[..., 3]).mean(2)
    med = np.nanmedian(raw, axis=2)                  # (nS,nM,5)
    churn, geff, beta = med[..., 0], med[..., 1], med[..., 3]

    REG = {"frz": "#cfd8e3", "shr": "#e8a0a0", "div": "#3a3a3a"}
    gmap = matplotlib.colormaps["viridis_r"]
    bmin, bmax = 0.4, 0.95
    rgba = np.ones((nS, nM, 3))
    cat = np.empty((nS, nM), dtype=object)
    for i in range(nS):
        for j in range(nM):
            if okfrac[i, j] < 0.5 or not np.isfinite(beta[i, j]):
                rgba[i, j] = matplotlib.colors.to_rgb(REG["div"]); cat[i, j] = "div"
            elif np.isfinite(churn[i, j]) and churn[i, j] < FREEZE_CHURN:
                rgba[i, j] = matplotlib.colors.to_rgb(REG["frz"]); cat[i, j] = "frz"
            elif geff[i, j] < 0:
                rgba[i, j] = matplotlib.colors.to_rgb(REG["shr"]); cat[i, j] = "shr"
            else:
                t = np.clip((beta[i, j] - bmin) / (bmax - bmin), 0, 1)
                rgba[i, j] = gmap(t)[:3]; cat[i, j] = "flu"

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    dmu = (mus[1] - mus[0]) / 2; dsg = (sig[1] - sig[0]) / 2
    ax.imshow(rgba, origin="lower", aspect="auto",
              extent=[mus[0]-dmu, mus[-1]+dmu, sig[0]-dsg, sig[-1]+dsg])
    for i in range(nS):
        for j in range(nM):
            if cat[i, j] == "flu":
                ax.text(mus[j], sig[i], f"{beta[i,j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if beta[i, j] > 0.7 else "black")
    ax.scatter([op[0]], [op[1]], marker="*", s=460, edgecolor="gold",
               facecolor="none", linewidth=2.4, zorder=6)
    ax.annotate("operating\npoint", (op[0], op[1]), (op[0]-0.33, op[1]+0.045),
                fontsize=8, color="#7a5c00", ha="center")
    # colorbar for beta
    sm = plt.cm.ScalarMappable(cmap=gmap, norm=plt.Normalize(bmin, bmax))
    cb = fig.colorbar(sm, ax=ax, pad=0.13, fraction=0.045)
    cb.set_label(r"size-volatility exponent $\beta$ (fluctuating+growing)")
    ax.set_xlabel(r"$\mu$  (mean competition; $\mu>0$ competitive)")
    ax.set_ylabel(r"$\sigma$  (interaction disorder)")
    ax.set_title(f"Own-degree relative GLV phase diagram  (N={int(d['N'])}, "
                 r"$\lambda$=" + f"{float(d['lam']):.0e}, {int(d['seeds'])} seeds)")
    leg = [Patch(facecolor=REG["frz"], label="frozen (shares relax)"),
           Patch(facecolor=gmap(0.5)[:3], label=r"fluctuating + growing (MSB; $\beta$ shown)"),
           Patch(facecolor=REG["shr"], label=r"shrinking ($g_{\rm eff}<0$)"),
           Patch(facecolor=REG["div"], label="divergent (unintegrable)")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.18, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(DATA, "phase_owndeg.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[phase] figure -> {out}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        SIGMAS = np.array([1.5, 1.8]); MUS = np.array([0.5, 1.76]); SEEDS = 2
    if "--plot" not in sys.argv:
        compute()
    plot()

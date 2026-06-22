"""Fine-resolution PHASE diagram of the own-degree relative GLV (regime-coloured,
not beta-coloured; mu<0 = competitive, thesis convention).

Regimes: DIVERGENT (integration fails/explodes) / FROZEN (shares relax) /
SHRINKING (g_eff<0) / FLUCTUATING+GROWING (the MSB regime). Robust to exploding
sims: per-seed wall-clock cap + finite-check + broad except -> bad runs become
NaN (flagged divergent), never crash the job; per-row checkpoint to the npz.

    uv run python scripts/phase_owndeg_fine.py            # compute (~2-3h) + plot
    uv run python scripts/phase_owndeg_fine.py --plot     # replot from the npz
    uv run python scripts/phase_owndeg_fine.py --smoke
"""
import sys, os, time, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate, growth_rate
from relative_glv.msb import size_volatility

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NPZ = os.path.join(DATA, "phase_owndeg_fine.npz")
N, LAM = 4000, 1e-3
TMAX, N_EVAL, LATE, DT = 400.0, 800, (320.0, 390.0), 0.5
SIGMAS = np.round(np.linspace(1.40, 1.95, 14), 4)
MUS = np.round(np.linspace(0.0, 2.5, 14), 4)        # code convention (mu>0 competitive)
SEEDS, N_JOBS, TIMEOUT = 5, 6, 90
OP = (1.76, 1.75)                                    # operating point (mu_code, sigma)
FREEZE_CHURN = 0.035


class _TO(Exception):
    pass


def _alarm(_s, _f):
    raise _TO()


def run_one(sigma, mu, seed):
    """Explosion-safe: wall-clock cap + finite check + broad except -> NaN on any
    failure (flagged divergent later), never raises."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        a = coupling(N, mu, sigma, kind="powerlaw_owndeg", seed=seed)
        r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                      method="RK45", rtol=1e-4, atol=1e-7)
        signal.alarm(0)
        if (not r["success"]) or (not np.isfinite(r["W"]).all()) or (not np.isfinite(r["lnM"]).all()):
            return [np.nan] * 5
        sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT)
        return [float(sv["growth"].std()), growth_rate(r["t"], r["lnM"], frac=0.3),
                sv["vol"].size / N, float(sv["beta"]), float(sv["r2"])]
    except Exception:
        signal.alarm(0)
        return [np.nan] * 5


def compute():
    t0 = time.time()
    nS, nM = len(SIGMAS), len(MUS)
    raw = np.full((nS, nM, SEEDS, 5), np.nan)
    print(f"[phase-fine] own-degree N={N} lam={LAM}  {nS}x{nM} grid x {SEEDS} seeds  "
          f"(cap {TIMEOUT}s/seed, explosion-safe, per-row checkpoint)", flush=True)
    for i, s in enumerate(SIGMAS):
        for j, m in enumerate(MUS):
            raw[i, j] = np.array(Parallel(n_jobs=N_JOBS)(
                delayed(run_one)(s, m, k) for k in range(SEEDS)))
        np.savez(NPZ, sigmas=SIGMAS, mus=MUS, seeds=SEEDS, N=N, lam=LAM,
                 op=np.array(OP), raw=raw)                          # checkpoint each row
        nd = int((np.isfinite(raw[i, :, :, 0]).mean(1) < 0.5).sum())
        print(f"  sigma={s:.3f} done  (divergent cols this row: {nd}/{nM})  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[phase-fine] saved {os.path.basename(NPZ)}  ({(time.time()-t0)/60:.1f} min)", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.interpolate import RegularGridInterpolator
    d = np.load(NPZ)
    sig, mus, raw, op = d["sigmas"], d["mus"], d["raw"], d["op"]
    integ = np.isfinite(raw[..., 0]).mean(2)
    med = np.nanmedian(raw, axis=2)
    churn, geff = med[..., 0], med[..., 1]
    for F in (churn, geff):
        F[~np.isfinite(F)] = 0.0                       # divergent cells flagged via integ

    # interpolate the regime-defining fields onto a fine mesh for smooth boundaries
    fch = RegularGridInterpolator((sig, mus), churn, bounds_error=False, fill_value=None)
    fge = RegularGridInterpolator((sig, mus), geff, bounds_error=False, fill_value=None)
    fin = RegularGridInterpolator((sig, mus), integ, bounds_error=False, fill_value=None)
    MU = np.linspace(mus[0], mus[-1], 400)
    SG = np.linspace(sig[0], sig[-1], 400)
    MM, SS = np.meshgrid(MU, SG)
    pts = np.column_stack([SS.ravel(), MM.ravel()])
    CH = fch(pts).reshape(SS.shape); GE = fge(pts).reshape(SS.shape); IN = fin(pts).reshape(SS.shape)

    # phase colours (regime, NOT beta)
    C = {"div": "#2f2f2f", "frz": "#aebfd4", "shr": "#e2948c", "flu": "#3f9b6e"}
    img = np.ones((*SS.shape, 3))
    for a in range(SS.shape[0]):
        for b in range(SS.shape[1]):
            if IN[a, b] < 0.5:
                k = "div"
            elif CH[a, b] < FREEZE_CHURN:
                k = "frz"
            elif GE[a, b] < 0:
                k = "shr"
            else:
                k = "flu"
            img[a, b] = matplotlib.colors.to_rgb(C[k])

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    # plot in mu<0 convention: x = -mu
    ax.imshow(img, origin="lower", aspect="auto",
              extent=[-mus[-1], -mus[0], sig[0], sig[-1]], interpolation="bilinear")
    ax.contour(-MM, SS, CH, levels=[FREEZE_CHURN], colors="0.15", linewidths=1.5)
    ax.contour(-MM, SS, GE, levels=[0.0], colors="#6e1414", linewidths=1.7)
    ax.contour(-MM, SS, IN, levels=[0.5], colors="white", linewidths=1.3, linestyles=":")
    ax.axhline(np.sqrt(2), color="#1f6f8b", ls="--", lw=1.2)
    ax.text(-mus[-1] + 0.05, np.sqrt(2) + 0.012, r"$\sigma_c=\sqrt{2}$ (DMFT)",
            color="#1f6f8b", fontsize=8.5, va="bottom")
    ax.scatter([-op[0]], [op[1]], marker="*", s=520, edgecolor="gold",
               facecolor="none", linewidth=2.6, zorder=6)
    ax.annotate("operating point", (-op[0], op[1]), (-op[0], op[1] - 0.09),
                fontsize=9, color="gold", ha="center", weight="bold")
    ax.set_xlabel(r"mean interaction  $\mu$   ($\mu<0$ competitive)", fontsize=12)
    ax.set_ylabel(r"interaction disorder  $\sigma$", fontsize=12)
    ax.set_title(f"Phase diagram of the own-degree relative GLV  (N={int(d['N'])})", fontsize=12)
    leg = [Patch(facecolor=C["flu"], label="fluctuating + growing (MSB regime)"),
           Patch(facecolor=C["frz"], label="frozen (shares relax)"),
           Patch(facecolor=C["shr"], label=r"shrinking ($g_{\rm eff}<0$)"),
           Patch(facecolor=C["div"], label="divergent (explodes)")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    fig.tight_layout()
    out = os.path.join(DATA, "phase_owndeg_fine.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[phase-fine] figure -> {out}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        SIGMAS = np.array([1.45, 1.7, 1.9]); MUS = np.array([0.0, 1.0, 2.0]); SEEDS = 2
    if "--plot" not in sys.argv:
        compute()
    plot()

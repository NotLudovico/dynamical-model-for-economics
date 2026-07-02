"""How own-degree departs from fully-connected as the mean degree C falls.

Own-degree keeps the single-site field variance ~sigma^2 M2 regardless of C, so reducing
C does NOT move the phase -- it only makes each firm's field a sum of FEWER neighbours,
i.e. less Gaussian => fatter growth tails. This sweeps C at the locked own-degree point
(mu=1.76, sigma=1.75) and tracks beta, survival, the fluctuating fraction, the tent
fatness (excess kurtosis), and -- the genuine-tent test -- the excess kurtosis of the
per-firm UN-MIXED growth ghat=(g-gbar)/sigma_i. FC family: unmix ~ 0 (a real fat tail
stays >0; a frozen+volatile mixing artefact goes <0). Large C -> FC limit (exk~7).

    uv run python rowavg/c_sweep.py [--smoke]
"""
import sys, os, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate, growth_rate, survivors
from relative_glv.msb import size_volatility, tent_stats

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = "--smoke" in sys.argv
MU, SIGMA, LAM, KIND = 1.76, 1.75, 1e-3, "powerlaw_owndeg"
N, SEEDS = (800, 3) if SMOKE else (1500, 4)
TMAX, NEV, LATE, DT = ((120.0, 600, (95.0, 118.0), 0.5) if SMOKE
                       else (250.0, 1000, (200.0, 245.0), 0.5))
CLIST = [100, 60, 40, 28, 20, 14, 10, 7]
TIMEOUT = 420


def _alarm(_s, _f):
    raise TimeoutError()


def run(C, seed):
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(TIMEOUT)
    try:
        a = coupling(N, MU, SIGMA, kind=KIND, seed=seed, mean_degree=C)
        r = integrate(a, tmax=TMAX, n_eval=NEV, lam=LAM, seed=seed,
                      method="LSODA", rtol=1e-5, atol=1e-8)   # stiff-capable: RK45 fails at intermediate C
        signal.alarm(0)
    except TimeoutError:
        return None
    if not r["success"]:
        return None
    deg = np.diff(a.indptr)
    W = r["W"]
    sv = size_volatility(W, r["t"], window=LATE, dt=DT, n_bins=20)
    if sv["growth"].size == 0 or not np.isfinite(sv["beta"]):
        return None
    g, vol = sv["growth"], sv["vol"]
    gh = ((g - g.mean(1, keepdims=True)) / vol[:, None]).ravel()
    gh = gh[np.isfinite(gh)]
    return dict(C=float(C), kmean=float(deg.mean()), kmin=int(deg.min()), beta=float(sv["beta"]),
                surv=int(survivors(W, 1e-6).sum()), exk=float(tent_stats(g)["exkurt"]),
                unmix=float(kurtosis(gh)), g_eff=float(growth_rate(r["t"], r["lnM"], 0.3)),
                fluc=float(np.mean(vol > 1e-3)))


def main():
    print(f"own-degree C-sweep  mu={MU} sigma={SIGMA}  N={N}x{SEEDS}   (FC ref: beta~0.84 exk~7 unmix~+0.4)\n")
    print(f'{"C":>5}{"<k>":>6}{"kmin":>6}{"surv":>6}{"fluc":>6}{"g_eff":>7}{"beta":>6}{"exk":>7}{"unmix_exk":>11}')
    rows = []
    for C in CLIST:
        recs = [x for x in Parallel(n_jobs=4, backend="loky")(
            delayed(run)(C, s) for s in range(SEEDS)) if x]
        if not recs:
            print(f'{C:>5}   all seeds failed/froze', flush=True); continue
        med = {k: float(np.median([r[k] for r in recs])) for k in recs[0]}
        rows.append(med)
        print(f'{C:>5.0f}{med["kmean"]:>6.0f}{med["kmin"]:>6.0f}{med["surv"]:>6.0f}{med["fluc"]:>6.2f}'
              f'{med["g_eff"]:>+7.2f}{med["beta"]:>6.2f}{med["exk"]:>7.1f}{med["unmix"]:>+11.2f}', flush=True)
    if not rows:
        print("nothing survived"); return
    C = [r["C"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].semilogx(C, [r["exk"] for r in rows], "o-"); ax[0].axhline(7, color="grey", ls=":", label="FC")
    ax[0].set(xlabel="mean degree C", ylabel="tent excess kurtosis", title="tent fattens as C falls")
    ax[0].legend()
    ax[1].semilogx(C, [r["unmix"] for r in rows], "o-"); ax[1].axhline(0, color="grey", ls=":")
    ax[1].set(xlabel="mean degree C", ylabel="un-mixed excess kurtosis",
              title="genuine (>0) vs frozen-artefact (<0)")
    ax[2].semilogx(C, [r["beta"] for r in rows], "o-")
    ax[2].axhspan(0.15, 0.20, color="green", alpha=0.15, label="empirical")
    ax[2].set(xlabel="mean degree C", ylabel="beta", title="size-volatility exponent"); ax[2].legend()
    plt.tight_layout()
    out = os.path.join(HERE, "c_sweep.png")
    plt.savefig(out, dpi=120)
    np.savez(os.path.join(HERE, "c_sweep.npz"), **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"\nsaved {out}\n      {os.path.join(HERE, 'c_sweep.npz')}")


if __name__ == "__main__":
    main()

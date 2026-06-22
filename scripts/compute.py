"""Regenerate the heavy datasets the story notebook embeds:

  data/msb.npz            MSB at the locked regime (sigma=1.75, mu=1.76, power-law):
                          single-run churn, size-volatility + tent, beta-vs-N + extrapolation,
                          freeze-vs-persist trajectories.
  data/phase_diagram.npz  (mu, sigma) grid of growth rate g_eff and chaos amplitude.
  data/dmft_validation.npz  matched fully-connected sim vs DMFT (growth, survival, chaos onset).

Dataset-oriented: compute once, store raw-enough results, replot freely in the notebook.

    uv run python scripts/compute.py [--smoke]
"""
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relative_glv.model import coupling, integrate, growth_rate, survivors
from relative_glv import msb as msb_mod
from relative_glv.dmft import solve_fixed_point, sigma_c

SMOKE = "--smoke" in sys.argv
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Locked regime
SIGMA, MU, LAM = 1.75, 1.76, 1e-3
N_JOBS = 6

# Full-run parameters
TMAX, N_EVAL = 400.0, 1600
MID_WIN = (240.0, 310.0)
LATE_WIN = (320.0, 390.0)
DT = 0.5

N_SHOWCASE = 6400
MSB_N, MSB_SEEDS = 6400, 8
NSCAN, NSCAN_SEEDS = [800, 1600, 3200, 6400, 12800], 8
TRAJ_N, TRAJ_SEEDS = 600, 12
SHARE_SAMPLE = 30

# Smoke-run overrides
if SMOKE:
    N_SHOWCASE = 400
    MSB_N, MSB_SEEDS = 400, 3
    NSCAN, NSCAN_SEEDS = [400, 800], 2
    TRAJ_N, TRAJ_SEEDS = 300, 3
    TMAX, N_EVAL = 120.0, 600
    MID_WIN = (60.0, 90.0)
    LATE_WIN = (95.0, 118.0)


# ---------------------------------------------------------------------------
# MSB helpers
# ---------------------------------------------------------------------------

def _simulate_msb(N, seed, store_shares=False):
    """Integrate one trajectory and extract MSB statistics.

    Returns a dict with the relevant stats; sets success=False if integration
    failed and skips that seed gracefully.
    """
    rec = dict(N=N, seed=int(seed), success=True, g_eff=np.nan, surv=np.nan,
               gstd_late=0.0, persistent=False, beta=np.nan, r2=np.nan)

    alpha = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed)
    result = integrate(alpha, tmax=TMAX, n_eval=N_EVAL, lam=LAM,
                       seed=seed, method="RK45", rtol=1e-4, atol=1e-7)

    if not result["success"]:
        rec["success"] = False
        return rec

    t = result["t"]
    W = result["W"]
    lnM = result["lnM"]

    # Aggregate growth rate from the late half
    lh = t > 0.5 * t[-1]
    rec["g_eff"] = float(np.polyfit(t[lh], lnM[lh], 1)[0])

    # Mid-window persistent survivors (for gstd_mid)
    mid_mask = (t >= MID_WIN[0]) & (t <= MID_WIN[1])
    sm = np.where(W[:, mid_mask].min(1) > 1e-6)[0]
    if sm.size:
        tgm = np.arange(MID_WIN[0], MID_WIN[1] + 1e-9, DT)
        gm = np.diff(
            np.array([np.interp(tgm, t, np.log(np.maximum(N * W[i], 1e-12))) for i in sm]),
            axis=1,
        )
        rec["gstd_mid"] = float(gm.std())

    # Late-window survivors
    late_mask = (t >= LATE_WIN[0]) & (t <= LATE_WIN[1])
    surv_idx = np.where(W[:, late_mask].min(1) > 1e-6)[0]
    rec["surv"] = float(surv_idx.size / N)

    if surv_idx.size >= 50:
        tg = np.arange(LATE_WIN[0], LATE_WIN[1] + 1e-9, DT)
        lnS = np.array([
            np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv_idx
        ])
        g = np.diff(lnS, axis=1)
        rec["gstd_late"] = float(g.std())
        rec["persistent"] = bool(g.std() > 0.05)

        # Size-volatility via the package function
        sv = msb_mod.size_volatility(W, t, window=LATE_WIN, dt=DT)
        rec["beta"] = sv["beta"]
        rec["r2"] = sv["r2"]
        rec["bin_S"] = sv["bin_S"]
        rec["bin_vol"] = sv["bin_vol"]
        rec["pk"] = sv["pk"]
        live_mask = (sv["vol"] > 0) & np.isfinite(sv["vol"]) & (sv["Sbar"] > 0)
        rec["sv_S"] = sv["Sbar"][live_mask].astype(np.float32)
        rec["sv_vol"] = sv["vol"][live_mask].astype(np.float32)

        # Growth rates rescaled via msb.rescale (sqrt(pi/2)*MAD, NOT std)
        flat = g.ravel()
        flat = flat[np.isfinite(flat)]
        z = msb_mod.rescale(flat)
        rng = np.random.default_rng(seed + 9)
        rec["growth_z"] = (
            rng.choice(z, 8000, replace=False) if z.size > 8000 else z
        ).astype(np.float32)

    if store_shares:
        end = W[:, -1]
        rng2 = np.random.default_rng(seed + 7)
        order = np.argsort(end)
        top = order[-(SHARE_SAMPLE // 2):]
        rnd = rng2.choice(order[:-(SHARE_SAMPLE // 2)], SHARE_SAMPLE - top.size, replace=False)
        idx = np.concatenate([top, rnd])
        rec["t"] = t.astype(np.float32)
        rec["lnM"] = lnM.astype(np.float32)
        rec["share_S"] = (N * W[idx]).astype(np.float32)

    return rec


# ---------------------------------------------------------------------------
# Block 1: MSB
# ---------------------------------------------------------------------------

def compute_msb():
    print(
        f"[compute_msb] smoke={SMOKE} | regime sigma={SIGMA} mu={MU} (chapter mu={-MU}) "
        f"| jobs={N_JOBS}",
        flush=True,
    )
    t0 = time.time()
    out = {}

    # (1/4) showcase single run
    print("[compute_msb] (1/4) showcase single run ...", flush=True)
    sc = _simulate_msb(N_SHOWCASE, 0, store_shares=True)
    for k in ("t", "lnM", "share_S"):
        out["sc_" + k] = sc.get(k, np.array([]))
    out["sc_g_eff"] = sc["g_eff"]
    out["sc_surv"] = sc["surv"]
    out["sc_N"] = N_SHOWCASE

    # (2/4) MSB size-volatility + tent
    print("[compute_msb] (2/4) MSB size-volatility + tent ...", flush=True)
    msb_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_simulate_msb)(MSB_N, s) for s in range(MSB_SEEDS)
    )
    msb_results = [
        r for r in msb_results
        if r["success"] and r["persistent"] and "sv_S" in r and np.isfinite(r["beta"])
    ]

    out["msb_z"] = (
        np.concatenate([r["growth_z"] for r in msb_results]) if msb_results else np.array([])
    )

    if msb_results:
        ts = msb_mod.tent_stats(out["msb_z"])
        out["msb_bowley"] = ts["bowley"]
        out["msb_exk"] = ts["exkurt"]
    else:
        out["msb_bowley"] = np.nan
        out["msb_exk"] = np.nan

    out["msb_betas"] = np.array([r["beta"] for r in msb_results])

    # Representative single economy (best survival) for size-volatility figure
    if msb_results:
        best = max(msb_results, key=lambda r: r["surv"])
        out["msb_sv_S"] = best["sv_S"]
        out["msb_sv_vol"] = best["sv_vol"]
        out["msb_bin_S"] = best["bin_S"]
        out["msb_bin_vol"] = best["bin_vol"]
        out["msb_pk"] = best["pk"]
        out["msb_beta"] = best["beta"]
        out["msb_r2"] = best["r2"]
        out["msb_surv"] = best["surv"]
    else:
        for k in ("msb_sv_S", "msb_sv_vol", "msb_bin_S", "msb_bin_vol"):
            out[k] = np.array([])
        out["msb_beta"] = np.nan
        out["msb_r2"] = np.nan
        out["msb_pk"] = 0
        out["msb_surv"] = np.nan
    out["msb_N"] = MSB_N

    # (3/4) N-scan (persistence + beta)
    print("[compute_msb] (3/4) N-scan (persistence + beta) ...", flush=True)
    jobs = [(N, s) for N in NSCAN for s in range(NSCAN_SEEDS)]
    ns = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(_simulate_msb)(N, s) for (N, s) in jobs
    )
    Ns, pf, bmed, blo, bhi = [], [], [], [], []
    for N in NSCAN:
        g = [r for r in ns if r["N"] == N and r["success"]]
        if not g:
            continue
        persist = [r for r in g if r["persistent"]]
        clean = [r["beta"] for r in persist if np.isfinite(r["beta"]) and r.get("r2", 0) > 0.9]
        Ns.append(N)
        pf.append(len(persist) / len(g))
        bmed.append(np.median(clean) if clean else np.nan)
        blo.append(np.percentile(clean, 25) if clean else np.nan)
        bhi.append(np.percentile(clean, 75) if clean else np.nan)
    Ns = np.array(Ns, float)
    bmed = np.array(bmed)
    out["ns_N"] = Ns
    out["ns_persistf"] = np.array(pf)
    out["ns_beta"] = bmed
    out["ns_blo"] = np.array(blo)
    out["ns_bhi"] = np.array(bhi)

    # Extrapolation beta = beta_inf + c / sqrt(N)
    fin = np.isfinite(bmed)
    if fin.sum() >= 3:
        x = 1.0 / np.sqrt(Ns[fin])
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, bmed[fin], rcond=None)
        out["ns_beta_inf"] = float(coef[0])
        out["ns_beta_slope"] = float(coef[1])
    else:
        out["ns_beta_inf"] = np.nan
        out["ns_beta_slope"] = np.nan

    # (4/4) freeze-vs-persist trajectories
    print("[compute_msb] (4/4) freeze-vs-persist trajectories ...", flush=True)
    tr = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_simulate_msb)(TRAJ_N, s, True) for s in range(TRAJ_SEEDS)
    )
    tr = [r for r in tr if r["success"] and "share_S" in r]
    out["traj_t"] = tr[0]["t"] if tr else np.array([])
    out["traj_shares"] = np.stack([r["share_S"] for r in tr]) if tr else np.array([])
    out["traj_persistent"] = np.array([r["persistent"] for r in tr])
    out["traj_gstd"] = np.array([r["gstd_late"] for r in tr])
    out["traj_N"] = TRAJ_N

    np.savez(os.path.join(DATA, "msb.npz"), **out)
    elapsed = (time.time() - t0) / 60.0
    print(
        f"[compute_msb] N-scan beta(N): "
        + ", ".join(f"N={int(n)}:{b:+.2f}" for n, b in zip(Ns, bmed))
        + f"  -> beta_inf={out['ns_beta_inf']:+.3f}",
        flush=True,
    )
    print(
        f"[compute_msb] MSB pooled beta={out['msb_beta']:+.2f} (r2={out['msb_r2']:.2f}), "
        f"bowley={out['msb_bowley']:+.2f}, exk={out['msb_exk']:.1f}; "
        f"showcase g_eff={sc['g_eff']:+.2f}",
        flush=True,
    )
    print(f"[compute_msb] saved -> data/msb.npz  ({elapsed:.1f} min)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Block 2: Phase diagram
# ---------------------------------------------------------------------------

# Phase grid parameters
if SMOKE:
    PHASE_MUS = [0.0, 2.0, 4.0]
    PHASE_SIGS = [0.5, 1.5, 3.0]
    PHASE_N = 300
    PHASE_TMAX = 120.0
    PHASE_N_EVAL = 600
    PHASE_LATE_START = 60.0
else:
    PHASE_MUS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    PHASE_SIGS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    PHASE_N = 2000
    PHASE_TMAX = 200.0
    PHASE_N_EVAL = 1200
    PHASE_LATE_START = 100.0

PHASE_SEEDS = 2
PHASE_LAM = 1e-3


def _phase_cell(seed, mu, sigma):
    """Compute (g_eff, fluctuation_std) for one (seed, mu, sigma) combination."""
    N = PHASE_N
    alpha = coupling(N, mu, sigma, kind="powerlaw", seed=seed)
    result = integrate(
        alpha,
        tmax=PHASE_TMAX,
        n_eval=PHASE_N_EVAL,
        lam=PHASE_LAM,
        seed=seed,
        method="RK45",
        rtol=1e-4,
        atol=1e-7,
    )

    if not result["success"]:
        return np.nan, 0.0

    t = result["t"]
    W = result["W"]
    lnM = result["lnM"]

    late = t > PHASE_LATE_START
    g_eff = float(np.polyfit(t[late], lnM[late], 1)[0])

    win_start = PHASE_LATE_START
    win_end = PHASE_TMAX * 0.95
    surv_mask = (t >= win_start) & (t <= win_end)
    surv_idx = np.where(W[:, surv_mask].min(1) > 1e-6)[0]

    if surv_idx.size < 30:
        return g_eff, 0.0

    tg = np.arange(win_start, win_end + 1e-9, 0.5)
    lnS = np.array([
        np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv_idx
    ])
    fluct = float(np.diff(lnS, axis=1).ravel().std())
    return g_eff, fluct


def compute_phase():
    print(
        f"[compute_phase] smoke={SMOKE} | N={PHASE_N}, lam={PHASE_LAM:g}, "
        f"{PHASE_SEEDS} seeds, {len(PHASE_MUS)}x{len(PHASE_SIGS)} grid",
        flush=True,
    )
    t0 = time.time()

    seeds = np.random.default_rng(7).integers(0, 2**31 - 1, size=PHASE_SEEDS)

    G = np.zeros((len(PHASE_SIGS), len(PHASE_MUS)))
    F = np.zeros_like(G)

    # Build all jobs: (j_sig, i_mu, seed_idx) -> (g, f)
    jobs = [
        (j, i, int(s))
        for j in range(len(PHASE_SIGS))
        for i in range(len(PHASE_MUS))
        for s in seeds
    ]

    def _job(j, i, s):
        return j, i, _phase_cell(s, PHASE_MUS[i], PHASE_SIGS[j])

    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(_job)(j, i, s) for (j, i, s) in jobs
    )

    # Accumulate: average over seeds
    counts = np.zeros_like(G)
    for j, i, (g, f) in results:
        if np.isfinite(g):
            G[j, i] += g
            F[j, i] += f
            counts[j, i] += 1

    valid = counts > 0
    G[valid] /= counts[valid]
    F[valid] /= counts[valid]

    for j, sig in enumerate(PHASE_SIGS):
        row = []
        for i, mu in enumerate(PHASE_MUS):
            row.append(f"({G[j,i]:+.1f},{F[j,i]:.2f})")
        print(f"sig={sig}: " + " ".join(row), flush=True)

    np.savez(
        os.path.join(DATA, "phase_diagram.npz"),
        G=G, F=F,
        MUS=np.array(PHASE_MUS),
        SIGS=np.array(PHASE_SIGS),
    )
    elapsed = (time.time() - t0) / 60.0
    print(f"[compute_phase] saved -> data/phase_diagram.npz  ({elapsed:.1f} min)", flush=True)
    return G, F


# ---------------------------------------------------------------------------
# Block 3: DMFT validation
# ---------------------------------------------------------------------------

if SMOKE:
    DMFT_SIGMAS = np.array([0.4, 1.0, 2.0])
    DMFT_MUI_MUS = np.array([0.0, 1.0])
    DMFT_N = 400
    DMFT_TMAX = 80.0
    DMFT_N_EVAL = 400
    DMFT_LATE = (50.0, 75.0)
    DMFT_SEEDS = 2
else:
    DMFT_SIGMAS = np.array([0.4, 0.7, 1.0, 1.2, 1.35, 1.6, 2.0, 2.5])
    DMFT_MUI_MUS = np.array([0.0, 1.0, 2.0])
    DMFT_N = 1500
    DMFT_TMAX = 160.0
    DMFT_N_EVAL = 800
    DMFT_LATE = (100.0, 150.0)
    DMFT_SEEDS = 3

DMFT_MU = 0.5
DMFT_LAM = 0.0   # match the lambda=0 DMFT


def _dmft_sim(seed, mu, sigma):
    """Simulate one FC trajectory and return (g_eff, survival, fluct)."""
    N = DMFT_N
    alpha = coupling(N, mu, sigma, kind="fc", seed=seed)
    result = integrate(
        alpha,
        tmax=DMFT_TMAX,
        n_eval=DMFT_N_EVAL,
        lam=DMFT_LAM,
        seed=seed,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
    )

    if not result["success"]:
        return np.nan, np.nan, np.nan

    t = result["t"]
    W = result["W"]
    lnM = result["lnM"]

    late = t > DMFT_LATE[0]
    g_eff = float(np.polyfit(t[late], lnM[late], 1)[0])

    win = (t >= DMFT_LATE[0]) & (t <= DMFT_LATE[1])
    surv_mask = W[:, win].min(1) > 1e-6
    surv = float(surv_mask.mean())

    if surv_mask.sum() > 20:
        tg = np.arange(DMFT_LATE[0], DMFT_LATE[1] + 1e-9, 0.5)
        lnS = np.array([
            np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12)))
            for i in np.where(surv_mask)[0]
        ])
        fluct = float(np.diff(lnS, axis=1).std())
    else:
        fluct = 0.0

    return g_eff, surv, fluct


def compute_dmft_validation():
    print(
        f"[compute_dmft] smoke={SMOKE} | N={DMFT_N}, fully-connected Gaussian, "
        f"gamma=0, lambda={DMFT_LAM}, sigma_c={sigma_c(0.0):.3f}",
        flush=True,
    )
    t0 = time.time()

    seeds = np.random.default_rng(11).integers(0, 2**31 - 1, size=DMFT_SEEDS)

    # Sweep over sigma at fixed mu
    print(f"sweep over sigma at mu={DMFT_MU:.2f}:", flush=True)

    def _sweep_job(sig, s):
        return sig, s, _dmft_sim(int(s), DMFT_MU, float(sig))

    sweep_jobs = [(sig, s) for sig in DMFT_SIGMAS for s in seeds]
    sweep_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_sweep_job)(sig, s) for (sig, s) in sweep_jobs
    )

    # Average over seeds per sigma
    sim_arr = np.zeros((len(DMFT_SIGMAS), 3))
    for k, sig in enumerate(DMFT_SIGMAS):
        rows = np.array([res for (s, _, res) in sweep_results if s == sig and np.isfinite(res[0])])
        if rows.size:
            sim_arr[k] = rows.mean(0)
        else:
            sim_arr[k] = np.nan
        print(
            f"  mu={DMFT_MU:+.2f} sigma={sig:.3f}: g_eff={sim_arr[k,0]:+.3f} "
            f"surv={sim_arr[k,1]:.3f} fluct={sim_arr[k,2]:.4f}",
            flush=True,
        )

    # mu-independence at sigma=1.0
    print("mu-independence at sigma=1.0 (shape fixed, g_eff shifts by -mu):", flush=True)
    mui_arr = np.zeros((len(DMFT_MUI_MUS), 3))
    for k, mu in enumerate(DMFT_MUI_MUS):
        rows = np.array([
            _dmft_sim(int(seeds[0]), float(mu), 1.0)
        ])
        valid = rows[np.isfinite(rows[:, 0])]
        if valid.size:
            mui_arr[k] = valid.mean(0)
        else:
            mui_arr[k] = np.nan
        print(
            f"  mu={mu:+.2f}: g_eff={mui_arr[k,0]:+.3f} "
            f"surv={mui_arr[k,1]:.3f} fluct={mui_arr[k,2]:.4f}",
            flush=True,
        )

    np.savez(
        os.path.join(DATA, "dmft_validation.npz"),
        sigmas=DMFT_SIGMAS,
        sim=sim_arr,
        mui_mus=DMFT_MUI_MUS,
        mui=mui_arr,
    )
    elapsed = (time.time() - t0) / 60.0
    print(f"[compute_dmft] saved -> data/dmft_validation.npz  ({elapsed:.1f} min)", flush=True)
    return sim_arr, mui_arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    print(
        f"[compute] smoke={SMOKE} | sigma={SIGMA}, mu={MU}, lam={LAM} | jobs={N_JOBS}",
        flush=True,
    )

    msb_out = compute_msb()
    G, F = compute_phase()
    sim_arr, mui_arr = compute_dmft_validation()

    total = (time.time() - t_start) / 60.0
    print("\n=== SUMMARY ===", flush=True)
    print(
        f"  beta_inf      = {float(msb_out['ns_beta_inf']):+.3f}",
        flush=True,
    )
    print(
        f"  MSB pooled beta = {float(msb_out['msb_beta']):+.2f}  "
        f"(r2={float(msb_out['msb_r2']):.2f})",
        flush=True,
    )
    print(
        f"  showcase g_eff  = {float(msb_out['sc_g_eff']):+.2f}",
        flush=True,
    )
    print(f"  total wall-clock: {total:.1f} min", flush=True)
    print("=== DONE ===", flush=True)

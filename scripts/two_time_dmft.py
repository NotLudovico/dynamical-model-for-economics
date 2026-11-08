"""Two-time HDMFT of the relative-size model at the reference point, and the life panel it predicts.

gamma = 0 (z_ij and z_ji independent), so the mean-field node has no response term.  A node of degree class p feels
    Phi_p(t) = a - mu B_p + sigma eta_p(t),    <eta_p(t) eta_p(t')> = Q_p(t - t') = sum_r Kc_pr E_k[S_r(t) S_r(t')],
with B_p = sum_r Kc_pr E_k S_r, Kc the soft-configuration kernel (expected neighbours in class r, over C) and
a = 1 - fbar fixed by E S = 1.  Each sampled node carries its own degree k, drawn from its class, and feels the
class field scaled by r = k/k_p (mean part r B_p, variance r Q_p); E_k is the degree-weighted class average, since
neighbours are drawn in proportion to degree.  Q_p(inf) > 0 is the frozen part (the spread of time-mean fields at equal degree);
Q_p(tau) - Q_p(inf) moves.  Solved by sampling (Eissfeller-Opper): draw eta from Q, integrate
dS = S(Phi - S)dt + sqrt(D S)dW with exit at S <= 0 and a newborn at D, re-measure Q, mix, repeat.
No network, no measured field: the inputs are (mu, sigma, D) and the degree sequence.

    uv run python scripts/two_time_dmft.py            # sim field stats, solve, panel -> data/two_time_dmft.npz
    uv run python scripts/two_time_dmft.py measured   # same, with the temporal field taken from simulation
    uv run python scripts/two_time_dmft.py measured3  # and the frozen spread too
"""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from economy import reference_economy, simulate_reference, lives, zeta_bins, DATA, SIGMA, D, MU, N, DT, SEEDS

OUT = DATA / "two_time_dmft.npz"
P, DTF, TAU = 32, 0.1, 80.0                 # degree classes, field grid, longest lag kept
NLAG = int(round(TAU / DTF)) + 1
TAIL = int(round(10 / DTF))                 # lags averaged for the frozen part Q(inf)


def degree_classes(deg, n=P):
    """Degree classes (n/2 equal-population edges merged with n/2 equal-degree-mass edges, so the hubs are resolved):
    node lists, weights, kernel Kc (rows sum to kappa)."""
    from scipy.optimize import least_squares
    srt = np.argsort(deg, kind="stable")
    mass = np.cumsum(deg[srt]) / deg.sum()
    cuts = np.unique(np.r_[np.linspace(0, deg.size, n // 2 + 1).astype(int),
                           np.searchsorted(mass, np.linspace(0, 1, n // 2 + 1)[1:-1]), deg.size])
    order = [srt[i:j] for i, j in zip(cuts[:-1], cuts[1:]) if j > i]
    w = np.array([o.size for o in order]) / deg.size
    frac = np.array([deg[o].mean() for o in order]) / deg.size
    c = deg.mean() / deg.size

    def K(lh):
        prod = np.exp(lh)[:, None] * np.exp(lh)[None, :]
        return prod / (prod + c)
    sol = least_squares(lambda lh: (K(lh) @ w - frac) / frac, np.log(frac), bounds=(-50.0, 20.0),
                        xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=20_000)
    assert np.max(np.abs(K(sol.x) @ w - frac) / frac) < 1e-8, "hidden-degree inversion did not converge"
    return order, w, K(sol.x) * w[None, :] / c


def gaussian_paths(cov, n_t, m, rng):
    """m stationary Gaussian paths of n_t grid points with autocovariance cov (zero beyond), circulant embedding."""
    L = 2 * n_t
    row = np.zeros(L)
    k = min(cov.size, n_t)
    row[:k] = cov[:k]
    row[L - k + 1:] = cov[1:k][::-1]
    lam = np.clip(np.fft.fft(row).real, 0.0, None)        # ponytail: clipped negative modes, tiny for a decaying cov
    xi = rng.standard_normal((m, L)) + 1j * rng.standard_normal((m, L))
    return np.fft.fft(np.sqrt(lam / L) * xi, axis=1).real[:, :n_t]


def integrate(phi, rng, dt=0.01, every=1):
    """One-node SDE along fields phi (m, n_t) on the DTF grid (held between points); exit at S <= 0, newborn at D.
    Returns S and the death counter every `every` grid points."""
    m, n_t = phi.shape
    per = int(round(DTF / dt))
    S, life = np.ones(m), np.zeros(m, np.int32)
    keep = range(0, n_t, every)
    S_out, L_out = np.empty((m, len(keep)), np.float32), np.empty((m, len(keep)), np.int32)
    for k in range(n_t):
        if k % every == 0:
            S_out[:, k // every], L_out[:, k // every] = S, life
        f = phi[:, k]
        for _ in range(per):
            S = S + S * (f - S) * dt + np.sqrt(D * S * dt) * rng.standard_normal(m)
            dead = S <= 0
            if dead.any():
                S[dead] = D
                life[dead] += 1
    return S_out, L_out


def autocorr(X, nlag, wt=None):
    """E[X(t) X(t + tau)] for tau = 0 .. nlag-1 grid steps, over t and rows (weighted by wt; zero-padded FFT)."""
    n = X.shape[1]
    nfft = 1 << int(np.ceil(np.log2(n + nlag)))
    F = np.fft.rfft(X, nfft, axis=1)
    P2 = (F * F.conj()).real
    avg = P2.mean(0) if wt is None else wt @ P2 / wt.sum()
    return np.fft.irfft(avg, nfft)[:nlag] / (n - np.arange(nlag))


def node_fields(Q, B, a, r, n_t, rng, moving=None, frozen=None):
    """Fields of the sampled nodes, class by class: r B_p mean, sqrt(r) times a frozen Gaussian part plus the moving
    part of Q_p (or the given moving covariance and frozen variance).  r is a list of relative-degree arrays, one per class."""
    out = np.empty((sum(x.size for x in r), n_t), np.float32)
    i = 0
    for p, rp in enumerate(r):
        qinf = Q[p, -TAIL:].mean()
        cov = Q[p] - qinf if moving is None else moving[p]
        qinf = qinf if frozen is None else frozen[p]
        eta = np.sqrt(qinf) * rng.standard_normal((rp.size, 1)) + gaussian_paths(cov, n_t, rp.size, rng)
        out[i:i + rp.size] = a - MU * rp[:, None] * B[p] + SIGMA * np.sqrt(rp)[:, None] * eta
        i += rp.size
    return out


def draw_r(kdeg, m, rng):
    return [rng.choice(k, m) / k.mean() for k in kdeg]


def solve(Kc, w, kdeg, iters=80, m=250, burn=100.0, T=250.0, seed=0, moving=None, frozen=None):
    """Stationary self-consistency for the degree-weighted C_p(tau), E_k S_p, and a (node-average E S = 1).  Mixing
    0.5 for the first half, then a running average of the iterates (the sampled C is noisy: E S^2 sits in the tail).
    With `moving` (and `frozen`) the temporal (and frozen) part of the field is that fixed input and only the rest
    is solved for.  Returns the weighted C and E_k S, the node-average C and E S, and a."""
    rng = np.random.default_rng(seed)
    n = len(w)
    lag = np.arange(NLAG) * DTF
    C = np.tile(1.0 + np.exp(-lag / 5.0), (n, 1))
    Cn, Ew, Es, a = C.copy(), np.ones(n), np.ones(n), 1.0 + MU
    nb, nt = int(round(burn / DTF)), int(round((burn + T) / DTF))
    for it in range(iters):
        t0 = time.time()
        r = draw_r(kdeg, m, rng)
        S, _ = integrate(node_fields(Kc @ C, Kc @ Ew, a, r, nt, rng, moving, frozen), rng)
        S = S[:, nb:].reshape(n, m, -1)
        newC = np.array([autocorr(S[p], NLAG, r[p]) for p in range(n)])
        newCn = np.array([autocorr(S[p], NLAG) for p in range(n)])
        newEw = np.array([r[p] @ S[p].mean(1) / r[p].sum() for p in range(n)])
        newEs = S.mean((1, 2))
        mix = 0.5 if it < iters // 2 else 1.0 / (it - iters // 2 + 2)
        C, Cn = (1 - mix) * C + mix * newC, (1 - mix) * Cn + mix * newCn
        Ew, Es = (1 - mix) * Ew + mix * newEw, (1 - mix) * Es + mix * newEs
        a += 1.0 * (1.0 - w @ newEs) * (1.0 if it < iters // 2 else 0.3)
        if it % 5 == 0 or it == iters - 1:
            Q = Kc @ C
            print(f"it {it:2d}  a {a:.4f}  E S {w @ newEs:.4f}  class-0 moving sd "
                  f"{SIGMA * np.sqrt(Q[0, 0] - Q[0, -TAIL:].mean()):.3f}  frozen sd {SIGMA * np.sqrt(Q[0, -TAIL:].mean()):.3f}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
    return C, Ew, Cn, Es, a


def sim_stats(seed, nlag=int(TAU / DT) + 1):
    """Per degree class of a stored reference economy: mean field, frozen and moving field variance, the moving
    field's autocovariance, E S, E[S(t)S(t+tau)] (lags on the DT grid) and turnover per node per unit time."""
    S, L, deg, _, Phi = reference_economy(seed)
    order, _, _ = degree_classes(deg)
    pm = Phi.mean(1)
    dev = Phi - pm[:, None]
    rows = []
    for o in order:
        rows.append(dict(mean=pm[o].mean(), frozen=pm[o].var(), moving=dev[o].var(1).mean(),
                         acov=autocorr(dev[o], nlag),
                         common=autocorr(dev[o].mean(0, keepdims=True), nlag), ES=S[o].mean(), CS=autocorr(S[o], nlag),
                         turnover=(L[o, -1] - L[o, 0]).mean() / (DT * (S.shape[1] - 1))))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def panel(Q, B, a, kdeg, m_class, seed=1, burn=100.0, T=480.0, nmin=20, moving=None, frozen=None):
    """DMFT life panel: m_class nodes per class (degrees drawn from the class) over a window T after burn."""
    rng = np.random.default_rng(seed)
    nt, every = int(round((burn + T) / DTF)), int(round(DT / DTF))
    nb = int(round(burn / DT))
    rows = []
    for p in range(len(Q)):
        m = max(int(round(m_class[p])), 1)
        r = [np.zeros(0)] * p + [rng.choice(kdeg[p], m) / kdeg[p].mean()]
        mv = None if moving is None else moving[:p + 1]
        fz = None if frozen is None else frozen[:p + 1]
        S, L = integrate(node_fields(Q[:p + 1], B[:p + 1], a, r, nt, rng, mv, fz), rng, every=every)
        rows += [(x[0], x[1], x[2]) for x in lives(S[:, nb:].astype(float), L[:, nb:])]
    out = np.array(rows)
    return out[out[:, 2] >= nmin]


def sim_panel(nmin=20):
    """Life panel of the two stored reference economies (Sbar, v, n), lives with >= nmin growth rates."""
    rows = []
    for s in SEEDS:
        d = np.load(simulate_reference(s))
        rows += [(x[0], x[1], x[2]) for x in lives(d["S"].astype(float), d["L"])]
    out = np.array(rows)
    return out[out[:, 2] >= nmin]


def measured_temporal(with_frozen=False):
    """Theory for the mean and the frozen field, with only the temporal field taken from simulation: its amplitude
    c sqrt(kappa) (one number) and its correlation, either the measured shape or exp(-tau/tau_c) (one more).
    with_frozen: the frozen spread at equal degree, f sqrt(kappa), is measured too (a third number)."""
    deg = np.load(simulate_reference(SEEDS[0]))["deg"]
    order, w, Kc = degree_classes(deg)
    kdeg = [deg[o].astype(float) for o in order]
    kappa = Kc.sum(1)
    sims = [sim_stats(s) for s in SEEDS]
    sim = {k: np.mean([x[k] for x in sims], axis=0) for k in sims[0]}
    c = float(np.exp(w @ (0.5 * np.log(sim["moving"] / kappa))))
    lag = np.arange(NLAG) * DTF
    rho05 = (w @ sim["acov"]) / (w @ sim["acov"][:, 0])
    rho = np.interp(lag, np.arange(rho05.size) * DT, rho05)
    tau_c = float(lag[np.argmax(rho < np.exp(-1))])
    var_r = np.array([(k / k.mean()).var() for k in kdeg])
    frz = sim["frozen"] - (MU * (Kc @ sim["ES"])) ** 2 * var_r          # less the spread of degrees within a class
    ok = frz > 0                                     # the top class's field is not linear in its wide degree range
    f = float(np.exp(w[ok] @ (0.5 * np.log(frz[ok] / kappa[ok])) / w[ok].sum()))
    frozen = (f ** 2 / SIGMA ** 2) * kappa if with_frozen else None
    print(f"measured: moving field sd = {c:.3f} sqrt(kappa), 1/e time {tau_c:.2f}, frozen sd = {f:.3f} sqrt(kappa)"
          f" ({'used' if with_frozen else 'not used'})")
    sp = sim_panel()
    zs = zeta_bins(sp[:, 0], sp[:, 1])
    print(f"simulation panel: zeta {np.round(zs[0], 3)}  ratios {np.round(zs[0] / zs[0][0], 2)}  "
          f"vol smallest/largest bin {zs[2]:.3f} {zs[3]:.3f}  decades {zs[4]:.2f}  local slopes {np.round(zs[5], 2)}")
    out = {}
    for name, shape in (("measured shape", rho), ("exponential", np.exp(-lag / tau_c))):
        moving = (c ** 2 / SIGMA ** 2) * kappa[:, None] * shape[None, :]
        C, Ew, Cn, Es, a = solve(Kc, w, kdeg, iters=50, moving=moving, frozen=frozen)
        Q, B = Kc @ C, Kc @ Ew
        qinf = Q[:, -TAIL:].mean(1) if frozen is None else frozen
        rows = panel(Q, B, a, kdeg, m_class=2 * N * w, moving=moving, frozen=frozen)
        z = zeta_bins(rows[:, 0], rows[:, 1])
        print(f"\n[{name}]  a {a:.3f}")
        print("class kappa | mean field th sim | frozen sd th sim | E S th sim")
        for p in (0, 10, 20, 24, 28):
            print(f"{p:3d} {kappa[p]:5.2f} | {a - MU * B[p]:6.2f} {sim['mean'][p]:6.2f} | {SIGMA * np.sqrt(qinf[p]):5.2f} "
                  f"{np.sqrt(sim['frozen'][p]):5.2f} | {Es[p]:5.2f} {sim['ES'][p]:5.2f}")
        print(f"panel: zeta {np.round(z[0], 3)}  ratios {np.round(z[0] / z[0][0], 2)}  vol smallest/largest bin "
              f"{z[2]:.3f} {z[3]:.3f}  decades {z[4]:.2f}  local slopes {np.round(z[5], 2)}", flush=True)
        out[name] = dict(zeta=z[0], a=a, B=B, qinf=qinf, Es=Es)
    np.savez(DATA / ("two_time_dmft_measured_three.npz" if with_frozen else "two_time_dmft_measured_temporal.npz"),
             c=c, tau_c=tau_c, f=f, sim_zeta=zs[0],
             **{f"{k.split()[0]}_{q}": v for k, d in out.items() for q, v in d.items()})


if __name__ == "__main__" and sys.argv[1:] in (["measured"], ["measured3"]):
    measured_temporal(with_frozen=sys.argv[1] == "measured3")
elif __name__ == "__main__":
    deg = np.load(simulate_reference(SEEDS[0]))["deg"]
    order, w, Kc = degree_classes(deg)
    kdeg = [deg[o].astype(float) for o in order]
    kappa, var_r = Kc.sum(1), np.array([(k / k.mean()).var() for k in kdeg])
    print("kappa by class:", np.round(kappa, 2), "\nnodes by class:", np.round(w * deg.size).astype(int))
    sims = [sim_stats(s) for s in SEEDS]
    sim = {k: np.mean([s[k] for s in sims], axis=0) for k in sims[0]}
    C, Ew, Cn, Es, a = solve(Kc, w, kdeg)
    Q, B = Kc @ C, Kc @ Ew
    qinf = Q[:, -TAIL:].mean(1)
    sub = int(round(DT / DTF))
    th = dict(mean=a - MU * B, frozen=SIGMA ** 2 * qinf + (MU * B) ** 2 * var_r, moving=SIGMA ** 2 * (Q[:, 0] - qinf),
              acov=SIGMA ** 2 * (Q - qinf[:, None])[:, ::sub], ES=Es, CS=Cn[:, ::sub])
    print("\nclass kappa | mean field th sim | frozen sd th sim | moving sd th sim | E S th sim | size moving var th sim")
    mv = lambda c: c[:, 0] - c[:, -20:].mean(1)
    for p in range(len(w)):
        print(f"{p:3d} {kappa[p]:5.2f} | {th['mean'][p]:6.2f} {sim['mean'][p]:6.2f} | "
              f"{np.sqrt(th['frozen'][p]):5.2f} {np.sqrt(sim['frozen'][p]):5.2f} | "
              f"{np.sqrt(th['moving'][p]):5.2f} {np.sqrt(sim['moving'][p]):5.2f} | {th['ES'][p]:5.2f} {sim['ES'][p]:5.2f} | "
              f"{mv(th['CS'])[p]:5.2f} {mv(sim['CS'])[p]:5.2f}")
    rho = lambda ac: (w @ ac) / (w @ ac[:, 0])
    ix = [1, 2, 4, 8, 16, 32, 64]
    print("moving-field correlation at lags 0.5 1 2 4 8 16 32:")
    print("  th ", np.round(rho(th["acov"])[ix], 3), "\n  sim", np.round(rho(sim["acov"])[ix], 3))
    rows = panel(Q, B, a, kdeg, m_class=2 * N * w)
    z = zeta_bins(rows[:, 0], rows[:, 1])
    print(f"\nDMFT panel, lives >= 20 growth rates: {len(rows)} lives, zeta {np.round(z[0], 3)}, "
          f"ratios {np.round(z[0] / z[0][0], 2)}, decades {z[4]:.2f}")
    print("full model on the same economies: zeta_1 0.25, ratios 1.79 2.37 2.80")
    np.savez(OUT, kappa=kappa, w=w, C=C, Ew=Ew, Cn=Cn, Es=Es, a=a, Q=Q, B=B,
             **{f"th_{k}": v for k, v in th.items()}, **{f"sim_{k}": v for k, v in sim.items()})

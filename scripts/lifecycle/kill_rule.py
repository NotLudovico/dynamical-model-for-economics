"""Kill rule for spells of the one-firm process dlnS/dt = Phi - S + lam (1/S - 1), Phi = m + f eta.

At lam = 0 the process is linear in 1/S:  1/S(t) = int_{-inf}^t exp(-[A(t) - A(t')]) dt',  A = int Phi.
Laplace at the downcrossing t0 of Phi (slope -s) gives, during a dip, S(t) ~ sqrt(s/2pi) exp(-[A(t0) - A(t)]).
So a negative excursion of Phi ends a spell iff its area exceeds ell_c(s) = ln(sqrt(s/2pi) / S_thr).
This script measures, per dip of a listed firm, the area, the crossing slope and whether S fell below
S_thr, and scores the rule.  Run: uv run python scripts/lifecycle/kill_rule.py
"""
import numpy as np

F, TAU, LAM = 1.82, 3.51, 1e-3          # alpha = 6 fitness (Table fitness of the theory note)
THR, DTI, T, BURN = 8 * LAM, 0.02, 1500.0, 50.0


def simulate(m, f, tau, lam, seed, t_end=T + BURN, dt=DTI):
    """Same scheme as surrogate.run, but returns Phi and S on the fine grid after the burn-in."""
    rng = np.random.default_rng(seed)
    n = m.size
    y = rng.standard_normal(n); eta = y.copy(); S = np.maximum(m, 0.1)
    a = np.exp(-dt / tau); b = np.sqrt(1 - a * a)
    nb, nt = int(BURN / dt), int(t_end / dt)
    P = np.empty((n, nt - nb), np.float32); X = np.empty((n, nt - nb), np.float32)
    for s in range(nt):
        eta += (np.sqrt(2) * y - eta) * dt / tau
        y = a * y + b * rng.standard_normal(n)
        Phi = m + f * eta
        S = S * np.exp((Phi - S) * dt) + lam * (1 - S) * dt
        if s >= nb:
            P[:, s - nb] = Phi; X[:, s - nb] = S
    return P, X


def dips(P, X, thr=THR, dt=DTI):
    """Per complete negative excursion of Phi that starts while listed: slope at downcrossing, area,
    depth, duration, size at downcrossing, killed?"""
    out = []
    neg = P < 0
    for i in range(P.shape[0]):
        d = np.diff(neg[i].astype(np.int8))
        st, en = np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1
        if en.size and st.size and en[0] < st[0]:
            en = en[1:]
        k = min(st.size, en.size); st, en = st[:k], en[:k]
        for a, b in zip(st, en):
            if a < 2 or X[i, a - 1] < thr:
                continue
            s = (P[i, a - 1] - P[i, a + 1]) / (2 * dt)
            seg = P[i, a:b]
            out.append((s, -seg.sum() * dt, -seg.min(), (b - a) * dt, X[i, a], X[i, a:b + 1].min() < thr))
    return np.array(out)


def lc(s, thr=THR):
    return np.log(np.sqrt(s / (2 * np.pi)) / thr)


if __name__ == "__main__":
    from joblib import Parallel, delayed
    us = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    R = Parallel(n_jobs=6)(delayed(lambda u, e: dips(*simulate(np.full(300, u * F), F, TAU, LAM, 7 + e)))(u, e)
                           for e, u in enumerate(us))
    print(f"f={F} tau={TAU} lam={LAM} thr={THR}")
    print(f"{'u':>5} {'dips':>6} {'P(kill)':>8} | {'rule acc':>8} {'rule P':>7} | S_dc/sqrt(2s/pi) median   area/ell_c at kill edge")
    for u, D in zip(us, R):
        s, A, h, L, Sdc, kill = D.T
        rule = A > lc(s)
        x = A / lc(s)
        edge = np.median(x[kill.astype(bool)]) if kill.any() else np.nan
        print(f"{u:5.2f} {len(D):6d} {kill.mean():8.3f} | {np.mean(rule == kill):8.3f} {rule.mean():7.3f} | "
              f"{np.median(Sdc / np.sqrt(2 * s / np.pi)):6.3f}                    {edge:.2f}")
    # logistic fit of P(kill | area - ell_c) to locate the empirical threshold and its width
    D = np.vstack(R); s, A, h, L, Sdc, kill = D.T
    z = A - lc(s)
    for lo in np.arange(-2, 2.01, 0.5):
        sel = (z >= lo) & (z < lo + 0.5)
        print(f"  area - ell_c in [{lo:+.1f},{lo + 0.5:+.1f}): n={sel.sum():6d}  P(kill)={kill[sel].mean():.3f}")


def spells_ode(X, thr=THR, dt=DTI):
    """Complete spells S > thr: lifetimes."""
    out = []
    on = X > thr
    d = np.diff(np.pad(on.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    for i in range(X.shape[0]):
        a, b = np.where(d[i] == 1)[0], np.where(d[i] == -1)[0]
        c = (a > 0) & (b < X.shape[1])
        out.append((b - a)[c] * dt)
    return np.concatenate(out), on.mean()


def spells_rule(P, lam=LAM, thr=THR, dt=DTI):
    """Spells from the fitness path alone: enter once A has risen by ell_e(s_up) since the upcrossing,
    exit once a dip's area reaches ell_c(s_dc).  No ODE."""
    n, nt = P.shape
    A = np.zeros(n); A0 = np.zeros(n); lim = np.full(n, np.inf)
    listed = np.zeros(n, bool); t_in = np.full(n, -1.0)
    out = []; time_on = 0
    for k in range(1, nt - 1):
        p0, p1 = P[:, k - 1], P[:, k]
        A += p1 * dt
        up, dn = (p0 < 0) & (p1 >= 0), (p0 >= 0) & (p1 < 0)
        s = np.abs(P[:, k + 1] - p0) / (2 * dt) + 1e-9
        # a crossing resets the reference point of the integrated fitness and the threshold
        reset_e = up & ~listed; reset_c = dn & listed
        A0[reset_e | reset_c] = A[reset_e | reset_c]
        lim[reset_e] = np.log(thr * np.sqrt(s[reset_e] / (2 * np.pi)) / lam)
        lim[reset_c] = np.log(np.sqrt(s[reset_c] / (2 * np.pi)) / thr)
        enter = ~listed & (p1 > 0) & (A - A0 >= lim)
        leave = listed & (p1 < 0) & (A0 - A >= lim)
        for i in np.where(leave)[0]:
            if t_in[i] >= 0:
                out.append((k * dt - t_in[i]))
        t_in[enter] = k * dt; t_in[leave] = -1
        listed |= enter; listed &= ~leave
        lim[enter | leave] = np.inf
        time_on += listed.sum()
    return np.array(out), time_on / (n * (nt - 2))


def pkill_slepian(u, f=F, tau=TAU, thr=THR, n=20000, tmax=12.0, dt=0.05, seed=0):
    """P(a dip kills | downcrossing at level -u) from the Slepian model: eta conditioned on eta(0) = -u,
    eta'(0) = -sdot, with sdot drawn from the crossing (Rayleigh) law.  No ODE, no rare-event sampling."""
    rng = np.random.default_rng(seed)
    t = dt * np.arange(1, int(tmax * tau / dt) + 1)
    rho = lambda L: (1 + np.abs(L) / tau) * np.exp(-np.abs(L) / tau)
    drho = lambda L: -(L / tau ** 2) * np.exp(-np.abs(L) / tau)
    C = rho(t[:, None] - t[None, :]) - np.outer(rho(t), rho(t)) - tau ** 2 * np.outer(drho(t), drho(t))
    Lc = np.linalg.cholesky(C + 1e-9 * np.eye(t.size))
    sdot = np.sqrt(-2 * np.log(rng.random(n))) / tau            # Rayleigh with variance parameter 1/tau^2
    eta = -u * rho(t)[None, :] + tau ** 2 * sdot[:, None] * drho(t)[None, :] + rng.standard_normal((n, t.size)) @ Lc.T
    phi = f * (u + eta)
    ret = np.argmax(phi >= 0, axis=1); never = ~(phi >= 0).any(1)
    area = -np.cumsum(np.minimum(phi, 0) * (np.arange(t.size)[None, :] < np.where(never, t.size, ret)[:, None]), 1)[:, -1] * dt
    return np.mean(never | (area > lc(f * sdot, thr)))


def pkill_asymptotic(u, f=F, tau=TAU, thr=THR):
    """Large-u limit: parabolic dips, kill iff (2/3) f sdot^3 tau^4 / u^2 > ell_c(f sdot); solved for sdot*."""
    from scipy.optimize import brentq
    g = lambda sd: (2 / 3) * f * sd ** 3 * tau ** 4 / u ** 2 - lc(f * sd, thr)
    sd = brentq(g, 1e-3 / tau, 50 / tau)
    return np.exp(-(sd * tau) ** 2 / 2)

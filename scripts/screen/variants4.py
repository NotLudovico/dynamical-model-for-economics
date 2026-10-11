"""Round 4: the sales-network model.  Absolute sizes x_i (sales), m = <x>.

  x_i = sum_j pi_ij a x_j + f_i            (Leontief: i sells a share pi_ij of buyer j's purchases)

  - every firm j buys inputs a x_j through k supplier slots of equal value a x_j / k;
  - final demand F = (1 - a) N m is spent through kh N household slots of equal value;
  - pi_ij = (slots of j held by i) / k; f_i = (household slots held by i) F / (kh N).

Total sales are fixed by final demand, sum_i x_i = F / (1 - a), so m needs no normalisation.  Only the
slots move:
  - switching: each slot drops its supplier at rate omega and re-sources with probability
    proportional to x (visibility);
  - exit: each firm exits at rate delta, and a firm left with no customers dies; its customers
    re-source the same way.  Its node is reborn as an entrant holding one household slot taken from
    a random holder, entry size x0 ~ (1 - a)/kh, and draws its k input suppliers afresh.

Growth: x_i'/x_i = sum_j w_ij x_j'/x_j + (customers won - customers lost)/x_i, with w_ij the share of
i's sales that goes to j.  The first term propagates buyers' growth upstream; the second is the
customer Herfindahl term.  Zipf: exits hand survivors a proportional drift nu = delta (1 - phi), so
the tail is zeta = delta / nu = 1 / (1 - phi), with phi the entrant mass share (~ x0 / m).
"""
import numpy as np


def _solve(x, a, k, sup, f, tol=1e-10, itmax=500):
    """Leontief fixed point x = a Pi x + f by iteration from the warm start x."""
    N = x.size
    for _ in range(itmax):
        xn = a * np.bincount(sup.ravel(), np.repeat(x / k, k), N) + f
        if np.max(np.abs(xn - x)) < tol * xn.mean():
            return xn
        x = xn
    return x


def simulate(*, a=0.5, k=10, kh=20, omega=0.2, delta=0.05, N=4000, T=300.0, burn=300.0, dt=0.05,
             rec=1.0, seed=0, **_):
    """dict(X, fid, lm, t, births, H, npart, deg): X[r, i] = x_i/m, H[r, i] the Herfindahl index of
    firm i's customers (household slots and firm buyers), npart the mean number of customers."""
    rng = np.random.default_rng(seed + 4000)
    Mh = kh * N
    F = (1.0 - a) * N
    hv = F / Mh                                            # value of one household slot
    w0 = 1.0 / (1.0 - rng.random(N))                       # Zipf start (the tail relaxes on ~1/(delta x0))
    p0 = w0 / w0.sum()
    sup = rng.choice(N, size=(N, k), p=p0)                 # sup[j, l]: supplier of buyer j's slot l
    self_ = sup == np.arange(N)[:, None]
    sup[self_] = (sup[self_] + 1) % N
    hh = rng.choice(N, size=Mh, p=p0)                      # supplier of each household slot
    f = np.bincount(hh, minlength=N) * hv
    x = _solve(np.ones(N), a, k, sup, f)
    fid, nxt = np.arange(N), N
    every, nstep, nburn = int(round(rec / dt)), int(round(T / dt)), int(round(burn / dt))
    X, Fid, tt, Hs, NP = [], [], [], [], []
    pk = 1.0 - np.exp(-omega * dt)

    def draw(n, excl=None):
        c = np.cumsum(x); s = rng.random(n) * c[-1]
        out = np.minimum(np.searchsorted(c, s, side="right"), N - 1)
        if excl is not None:                                # no self-supply: shift to the next node
            bad = out == excl
            out[bad] = (out[bad] + 1) % N
        return out

    for s in range(nstep + nburn):
        # switching (firm slots, then household slots)
        m1 = rng.random((N, k)) < pk
        if m1.any():
            jj, ll = np.nonzero(m1)
            sup[jj, ll] = draw(jj.size, jj)
        m2 = rng.random(Mh) < pk
        if m2.any():
            hh[m2] = draw(int(m2.sum()))
        f = np.bincount(hh, minlength=N) * hv
        x = _solve(x, a, k, sup, f)
        # exit: hazard delta, or no customers left
        dead = (rng.random(N) < delta * dt) | (x < 1e-12)
        if dead.any():
            d = np.flatnonzero(dead)
            x[d] = 0.0                                      # dead nodes cannot be chosen
            ms = np.isin(sup, d)
            if ms.any():
                jj, ll = np.nonzero(ms)
                sup[jj, ll] = draw(jj.size, jj)
            mh = np.isin(hh, d)
            if mh.any():
                hh[mh] = draw(int(mh.sum()))
            for i in d:                                     # entrants: one household slot, fresh inputs
                hh[rng.integers(Mh)] = i
                c = np.cumsum(x); sup[i] = np.minimum(np.searchsorted(c, rng.random(k) * c[-1], side="right"), N - 1)
                sup[i][sup[i] == i] = (i + 1) % N
            fid[d] = np.arange(nxt, nxt + d.size); nxt += d.size
            f = np.bincount(hh, minlength=N) * hv
            x[d] = hv
            x = _solve(x, a, k, sup, f)
        if s >= nburn and (s - nburn) % every == 0:
            m = x.mean()
            # customer Herfindahl: chunks a x_j / k per firm slot, hv per household slot
            flat = sup.ravel(); val = np.repeat(a * x / k, k)
            key = flat.astype(np.int64) * N + np.repeat(np.arange(N), k)
            u, inv = np.unique(key, return_inverse=True)
            cv = np.bincount(inv, val)                      # sales of supplier u // N to buyer u % N
            sq = np.bincount(u // N, cv ** 2, N) + np.bincount(hh, minlength=N) * hv ** 2
            ncust = np.bincount(u // N, minlength=N) + (np.bincount(hh, minlength=N) > 0)
            X.append((x / m).astype(np.float32)); Fid.append(fid.copy()); tt.append(s * dt)
            Hs.append((sq / np.maximum(x, 1e-12) ** 2).astype(np.float32)); NP.append(ncust.mean())
    return dict(X=np.array(X), fid=np.array(Fid, dtype=np.int32), lm=np.zeros(len(X)), t=np.array(tt),
                births=(nxt - N) / N / (T + burn), H=np.array(Hs), npart=float(np.mean(NP)),
                deg=np.zeros(N, int))

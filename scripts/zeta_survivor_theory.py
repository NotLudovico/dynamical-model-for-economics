"""Multiscaling from the survivor ensemble (thesis/standalone/beta_theory, Sec. 7 remark).

With v = sqrt(kappa) A(S) zeta, the q-th size-conditioned moment splits as
    zeta_q = zeta_q^A - e_q,   e_q = dln E[kappa^(q/2) | S] / dln S,
where zeta_q^A is the q-th exponent of the degree-normalised volatility v/sqrt(kappa).
Multiscaling (zeta_q/zeta_1 < q) needs e_q > q e_1. The log-Gaussian closure gives
e_q = q e_1 exactly; the survivor joint law (eq. joint: Pareto degrees, Gaussian static
fitness, extreme-value survival) does not.

Per degree exponent alpha, all on each ensemble's own decline branch with the thesis
estimator (bin means of y^q, cq_one of pl22_conditional.py):
  EV   e_q from the survivor ensemble, order parameters from the two-time HDMFT
       (interpolated in kappa, drawn on the realised degree sequence), nothing fitted
  HD   zeta_q, zeta_q^A, e_q on the HDMFT's own surviving paths
  sim  zeta_q, zeta_q^A, e_q in simulation (N=8000, eight economies, medians)
  th   zeta_q^A(HD) - e_q(EV): the leading-order prediction

Run: uv run python scripts/zeta_survivor_theory.py   (-> data/zeta_survivor_theory.npz, ~15 min)
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from relative_glv import model
from beta_survivor_theory import (graph, solve_hdmft, order_parameters, mad, MU, SIGMA, LAM, N,
                                  C, DELTA, T0, FLOOR)

ALPHAS, SEEDS, QS, NB = (2.2, 2.5, 3.5, 6.0), 8, np.arange(1, 5), 20
OUT = os.path.join(ROOT, "data", "zeta_survivor_theory.npz")


def decline_start(S, v, nb=NB):
    """Decline branch as in cq_one: lower edge at the peak of the binned median volatility,
    and max(6, nb - peak) fit bins."""
    o = np.argsort(S)
    P = np.array_split(o, nb)
    bx = np.array([S[p].mean() for p in P])
    pk = int(np.argmax([np.median(v[p]) for p in P]))
    return bx[pk], max(6, nb - pk)


def exponents(S, ys, dec_branch):
    """-dln E[y^q | S]/dln S, q=1..4, for each y in ys, on the decline branch (cq_one binning).
    None if the branch holds fewer than 200 firms."""
    S0, nfit = dec_branch
    dec = S > S0
    if dec.sum() < 200:
        return None
    Sx = S[dec]
    o = np.argsort(Sx)
    P = np.array_split(o, nfit)
    bx = np.log(np.array([Sx[p].mean() for p in P]))
    return [np.array([-np.polyfit(bx, np.log([np.mean(y[dec][p] ** q) for p in P]), 1)[0]
                      for q in QS]) for y in ys]


def ev_ensemble(op, kap, rng, M=2_000_000):
    """Survivor ensemble on the realised degrees; class order parameters interpolated in ln kappa."""
    lk, lc = np.log(kap), np.log(op["kap_c"])
    qT = np.exp(np.interp(lk, lc, np.log(op["qT"])))
    q0 = np.exp(np.interp(lk, lc, np.log(op["q0"])))
    r = np.interp(lk, lc, op["r"])
    i = rng.integers(0, kap.size, M)
    a, f = SIGMA * np.sqrt(qT[i]), SIGMA * np.sqrt(np.maximum(q0[i] - qT[i], 1e-9))
    S = r[i] + a * rng.standard_normal(M)
    live = (S > 0) & (rng.uniform(size=M) < norm.cdf(S / f) ** (T0 / (2 * op["tau"])))
    return S[live], kap[i][live]


def hdmft_side(r, op):
    t, U = r["t"], r["U"].astype(float)
    st = int(round(DELTA / op["dt"]))
    ncls, npc, _ = U.shape
    kap = np.repeat(op["kap_c"], npc)
    U = U.reshape(-1, t.size)[:, op["late"]]
    live = U.min(1) > FLOOR
    S, kap = U.mean(1)[live], kap[live]
    v = mad(np.diff(np.log(U[live][:, ::st]), axis=1))
    br = decline_start(S, v)
    z, zA, zk = exponents(S, (v, v / np.sqrt(kap), np.sqrt(kap)), br)
    return dict(z=z, zA=zA, e=-zk, br=br)


def economy(alpha, seed):
    a = graph(alpha, seed)
    res = model.integrate(a, tmax=250.0, n_eval=1001, lam=LAM, seed=seed,
                          method="RK45", rtol=1e-4, atol=1e-7)
    if not res["success"]:
        return None
    t = res["t"]
    S = N * res["W"][:, (t >= 180) & (t <= 240.0001)][:, ::2]    # Delta = 0.5 grid
    live = S.min(1) > FLOOR
    kap = np.diff(a.indptr)[live] / C
    S = S[live]
    v = mad(np.diff(np.log(S), axis=1))
    Sb = S.mean(1)
    ex = exponents(Sb, (v, v / np.sqrt(kap), np.sqrt(kap)), decline_start(Sb, v))
    return np.concatenate([ex[0], ex[1], -ex[2]]) if ex and ex[0][0] > 0 else None


if __name__ == "__main__":
    t0, rng = time.time(), np.random.default_rng(0)
    rows = []
    for alpha in ALPHAS:
        r = solve_hdmft(alpha)
        op = order_parameters(r)
        hd = hdmft_side(r, op)
        Sev, kev = ev_ensemble(op, np.diff(graph(alpha, 0).indptr) / C, rng)
        e_ev = -exponents(Sev, (np.sqrt(kev),), hd["br"])[0]
        sims = [x for x in Parallel(n_jobs=4)(delayed(economy)(alpha, s) for s in range(SEEDS))
                if x is not None]
        sim = np.median(np.array(sims), 0).reshape(3, QS.size)
        rows.append(np.stack([hd["z"], hd["zA"], hd["e"], e_ev, hd["zA"] - e_ev, *sim]))
        print(f"\nalpha={alpha}  ({len(sims)} economies, {(time.time() - t0) / 60:.1f} min)")
        for name, x in zip(("zeta HD", "zetaA HD", "e_q HD", "e_q EV", "zeta th",
                            "zeta sim", "zetaA sim", "e_q sim"), rows[-1]):
            print(f"  {name:9s} " + " ".join(f"{y:6.3f}" for y in x)
                  + "   ratio " + " ".join(f"{y:5.2f}" for y in x / x[0]), flush=True)
    np.savez(OUT, alphas=np.array(ALPHAS), qs=QS, table=np.array(rows),
             rows=np.array(["zeta_HD", "zetaA_HD", "e_HD", "e_EV", "zeta_th",
                            "zeta_sim", "zetaA_sim", "e_sim"]))

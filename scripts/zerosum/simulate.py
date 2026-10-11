# Zero-sum exchange with turnover: x_i' = sum_j W_ij (x_i x_j)^a / m^(2a-1), W antisymmetric
# (a=1: zero-sum GLV; sum_i x_i' = 0 for any a).
# A firm whose share falls below FLOOR exits; its node is taken by an entrant at X0 with
# freshly drawn couplings on the node's edges (randomness only at birth, none in the dynamics).
#
# Run (from the repo root): RGLV_ALPHA_PL=2.5 uv run python scripts/zerosum/simulate.py \
#     powerlaw 2000 400 1e-4 1e-3 data/zerosum/a25_08.npz 0.8     # kind N T floor x0 out a
# then: uv run python scripts/zerosum/analyse.py data/zerosum/a25_08.npz
import os, sys, numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.getcwd())
from relative_glv.model import coupling

kind, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
FLOOR, X0 = float(sys.argv[4]), float(sys.argv[5])
out = sys.argv[6]
a = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
rng = np.random.default_rng(1)
if kind == "fc":
    A = np.ones((N, N)) - np.eye(N)
else:
    A = (coupling(N, 0.0, 1.0, kind=kind, gamma=-1.0, seed=1).toarray() != 0).astype(float)
C = A.sum(1).mean()
Z = np.triu(rng.standard_normal((N, N)), 1) * A
W = (Z - Z.T) / np.sqrt(C)
deg = A.sum(1)

u = rng.normal(0, 1, N); u -= np.log(np.exp(u).mean())
LNF = np.log(FLOOR)
def rhs(t, u):
    # a<1: x' ~ -x^a hits 0 in finite time, so firms below the floor are frozen (and
    # removed from the exchange) until the exit step at the end of the chunk.
    live = u > LNF
    x = np.where(live, np.exp(np.clip(u, LNF, 30)), 0.0); m = x.mean()
    return np.where(live, x ** (a - 1) * (W @ x ** a) / m ** (2 * a - 1), 0.0)
dt = 0.5
U = np.empty((int(T / dt), N)); firm = np.empty((int(T / dt), N), int)
fid = np.arange(N); nxt = N
for s in range(U.shape[0]):
    u = solve_ivp(rhs, (0, dt), u, method="DOP853", rtol=1e-8, atol=1e-8, max_step=0.05).y[:, -1]
    dead = u < np.log(FLOOR)
    for i in np.flatnonzero(dead):
        nb = np.flatnonzero(A[i])
        z = rng.standard_normal(nb.size) / np.sqrt(C)
        W[i, nb], W[nb, i] = z, -z
        fid[i] = nxt; nxt += 1
    u[dead] = np.log(X0)
    u -= np.log(np.exp(u).mean())
    U[s], firm[s] = u, fid
    if s % 100 == 0: print(s * dt, nxt - N, flush=True)
print(kind, N, "a", a, "C", C, "births per node per time", (nxt - N) / N / T)
np.savez_compressed(out, U=U, firm=firm, deg=deg, dt=dt, floor=FLOOR, x0=X0)

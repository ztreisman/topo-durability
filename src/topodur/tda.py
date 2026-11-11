"""Persistence diagrams, scores, bootstrap significance, circular coordinates, order test,
and representative cycles.

Circular coordinates follow de Silva, Morozov & Vejdemo-Johansson (2011): take a persistent
cohomology class (ripser cocycle over Z/p), lift it to an integer cocycle on the Rips
1-skeleton at a radius inside the bar, and solve min_f ||z - delta0 f||_2; theta = f mod 1.
This is the same algorithm dreimac uses (with all points as landmarks), implemented here
directly because concept families are small (7 to ~300 points) and dreimac's API has
changed across versions. tests/test_tda.py checks it on synthetic circles.
"""

from __future__ import annotations

from dataclasses import dataclass

import gudhi
import numpy as np
from ripser import ripser
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsqr

# ---------------------------------------------------------------- diagrams and scores


def diagrams(D: np.ndarray, maxdim: int = 1, coeff: int = 2, cocycles: bool = False) -> dict:
    return ripser(D, maxdim=maxdim, distance_matrix=True, coeff=coeff, do_cocycles=cocycles)


def finite(dgm: np.ndarray) -> np.ndarray:
    return dgm[np.isfinite(dgm[:, 1])] if len(dgm) else dgm.reshape(0, 2)


def lifetimes(dgm: np.ndarray) -> np.ndarray:
    f = finite(dgm)
    return f[:, 1] - f[:, 0]


def top_class(dgm: np.ndarray) -> int | None:
    lt = lifetimes(dgm)
    return int(np.argmax(lt)) if len(lt) else None


def scores(dgms: list[np.ndarray]) -> dict:
    """h1_max (largest lifetime), h1_ratio (largest death/birth), and the top H1 bar."""
    out: dict = {}
    for dim in (1, 2):
        if dim >= len(dgms):
            continue
        f = finite(dgms[dim])
        lt = f[:, 1] - f[:, 0] if len(f) else np.zeros(0)
        name = f"h{dim}"
        out[f"{name}_max"] = float(lt.max()) if len(lt) else 0.0
        if len(f):
            ratio = f[:, 1] / np.maximum(f[:, 0], 1e-12)
            out[f"{name}_ratio"] = float(ratio.max())
            j = int(np.argmax(lt))
            out[f"{name}_birth"], out[f"{name}_death"] = float(f[j, 0]), float(f[j, 1])
        else:
            out[f"{name}_ratio"] = 1.0
            out[f"{name}_birth"] = out[f"{name}_death"] = np.nan
        out[f"{name}_n"] = int(len(f))
    return out


def bottleneck(d1: np.ndarray, d2: np.ndarray) -> float:
    return float(gudhi.bottleneck_distance(finite(d1), finite(d2)))


# ---------------------------------------------------------------- bootstrap band (Fasy et al. 2014)


@dataclass
class Band:
    c: dict[int, float]           # dim -> 95th percentile bottleneck distance
    boot: dict[int, np.ndarray]   # dim -> all bootstrap distances

    def threshold(self, dim: int) -> float:
        return 2 * self.c[dim]


def bootstrap_band(
    instances: np.ndarray,          # [I, n, D] per-item instance vectors at one layer
    n_inst: np.ndarray,             # [I] valid rows per item
    to_distance,                    # callable: centroids [I, D] -> distance matrix [I, I]
    dgms_ref: list[np.ndarray],
    maxdim: int = 1,
    n_boot: int = 200,
    q: float = 0.95,
    seed: int = 0,
) -> Band:
    """Resample contexts per item, recompute centroids, the full preprocessing, and diagrams;
    c = q-quantile of bottleneck distance to the reference diagram. Significant: lifetime > 2c."""
    rng = np.random.default_rng(seed)
    I = instances.shape[0]
    dists = {d: np.zeros(n_boot) for d in range(1, maxdim + 1)}
    for b in range(n_boot):
        C = np.stack([
            instances[i, rng.integers(0, n_inst[i], n_inst[i])].astype(np.float64).mean(0) for i in range(I)
        ])
        dg = diagrams(to_distance(C), maxdim=maxdim)["dgms"]
        for d in dists:
            dists[d][b] = bottleneck(dg[d], dgms_ref[d])
    return Band({d: float(np.quantile(v, q)) for d, v in dists.items()}, dists)


def n_significant(dgm: np.ndarray, threshold: float) -> int:
    return int((lifetimes(dgm) > threshold).sum())


# ---------------------------------------------------------------- circular coordinates


@dataclass
class CircCoords:
    theta: np.ndarray        # [n] radians in [0, 2pi)
    radius: float            # Rips radius used
    birth: float
    death: float
    connected: bool          # was the 1-skeleton at `radius` connected
    cocycle_ok: bool         # lifted cochain satisfied the cocycle condition on triangles
    edges: np.ndarray        # [E, 2] edges of the 1-skeleton (i < j)
    z: np.ndarray            # [E] integer cocycle on those edges
    zmap: dict               # (i, j) -> cocycle value mod prime, i < j
    prime: int


def _lift(zmap: dict, edges: np.ndarray, prime: int) -> tuple[np.ndarray, np.ndarray]:
    z_mod = np.array([zmap.get((int(a), int(b)), 0) for a, b in edges], dtype=int)
    return z_mod, np.where(z_mod > prime // 2, z_mod - prime, z_mod).astype(float)


def _edges_at(D: np.ndarray, r: float) -> np.ndarray:
    iu, ju = np.triu_indices(len(D), 1)
    m = D[iu, ju] <= r
    return np.stack([iu[m], ju[m]], 1)


def _check_cocycle(D, r, edges, z, p) -> bool:
    idx = {(int(a), int(b)): k for k, (a, b) in enumerate(edges)}
    n = len(D)
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) not in idx:
                continue
            for c in range(b + 1, n):
                if (b, c) in idx and (a, c) in idx:
                    if (z[idx[(a, b)]] + z[idx[(b, c)]] - z[idx[(a, c)]]) % p != 0:
                        return False
    return True


def circular_coordinates(D: np.ndarray, cls: int | None = None, r_frac: float = 0.5,
                         prime: int = 47) -> CircCoords | None:
    """Circle-valued coordinate from the H1 class `cls` (default: most persistent)."""
    res = ripser(D, maxdim=1, distance_matrix=True, coeff=prime, do_cocycles=True)
    dgm1 = res["dgms"][1]
    if len(dgm1) == 0:
        return None
    lt = dgm1[:, 1] - dgm1[:, 0]
    lt[~np.isfinite(lt)] = -1
    cls = int(np.argmax(lt)) if cls is None else cls
    birth, death = map(float, dgm1[cls])
    r = birth + r_frac * (death - birth)
    edges = _edges_at(D, r)
    zmap: dict[tuple[int, int], int] = {}
    for i, j, v in res["cocycles"][1][cls]:
        i, j, v = int(i), int(j), int(v) % prime
        if i > j:
            i, j, v = j, i, (-v) % prime
        zmap[(i, j)] = v
    z_mod, z = _lift(zmap, edges, prime)
    ok = _check_cocycle(D, r, edges, z_mod, prime) if len(D) <= 120 else True

    n, E = len(D), len(edges)
    rows = np.repeat(np.arange(E), 2)
    cols = edges.reshape(-1)
    vals = np.tile([-1.0, 1.0], E)
    delta0 = coo_matrix((vals, (rows, cols)), shape=(E, n)).tocsr()
    f = lsqr(delta0, z, atol=1e-12, btol=1e-12)[0]
    theta = np.mod(2 * np.pi * f, 2 * np.pi)

    adj = csr_matrix((np.ones(E), (edges[:, 0], edges[:, 1])), shape=(n, n))
    from scipy.sparse.csgraph import connected_components
    connected = connected_components(adj, directed=False)[0] == 1
    return CircCoords(theta, r, birth, death, connected, ok, edges, z, zmap, prime)


# ---------------------------------------------------------------- order score


def _uniform_scores(x: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return 2 * np.pi * ranks / len(x)


def circular_rank_corr(theta: np.ndarray, truth: np.ndarray) -> float:
    """Fisher & Lee (1982)-style circular rank correlation, invariant to rotation and
    reflection: max over orientations of |mean exp(i(a -/+ b))|^2 on uniform scores.
    1 = items in the correct cyclic order; ~1/n for a random order."""
    a, b = _uniform_scores(theta), _uniform_scores(truth)
    return float(max(abs(np.exp(1j * (a - b)).mean()) ** 2, abs(np.exp(1j * (a + b)).mean()) ** 2))


def order_test(theta: np.ndarray, truth: np.ndarray, n_perm: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Order score and label-permutation p-value (null 3 in the spec)."""
    rng = np.random.default_rng(seed)
    obs = circular_rank_corr(theta, truth)
    null = np.array([circular_rank_corr(theta, rng.permutation(truth)) for _ in range(n_perm)])
    return obs, float((1 + (null >= obs - 1e-12).sum()) / (1 + n_perm))


def fiedler_order(D: np.ndarray) -> np.ndarray:
    """1-D spectral seriation (Atkins, Boman & Hendrickson 1998): the Fiedler vector of the graph
    Laplacian of a Gaussian affinity on D (bandwidth = median off-diagonal distance). Recovers the
    order of points along a curved arc, where PC1 would fold a horseshoe."""
    off = D[~np.eye(len(D), dtype=bool)]
    sig = float(np.median(off)) or 1.0
    W = np.exp(-(D / sig) ** 2)
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    Dm = 1 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(D)) - Dm[:, None] * W * Dm[None, :]
    _, vecs = np.linalg.eigh(L)
    return vecs[:, 1] * Dm


def _rank_r2(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    r = np.corrcoef(rx, ry)[0, 1]
    return float(r * r) if np.isfinite(r) else 0.0


def line_order_test(D: np.ndarray, truth: np.ndarray, n_perm: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Linear-order score for interval families (line controls): squared Spearman correlation
    between the Fiedler coordinate and the true rank (orientation-free, 1 = perfect order; squared
    so it is on the same r^2-like scale as circular_rank_corr), and label-permutation p-value."""
    rng = np.random.default_rng(seed)
    f = fiedler_order(D)
    obs = _rank_r2(f, truth)
    null = np.array([_rank_r2(f, rng.permutation(truth)) for _ in range(n_perm)])
    return obs, float((1 + (null >= obs - 1e-12).sum()) / (1 + n_perm))


# ---------------------------------------------------------------- representative cycles


def representative_cycle(D: np.ndarray, cc: CircCoords, r_frac: float = 0.02, max_seeds: int = 64) -> list[int]:
    """Shortest closed loop in the Rips 1-skeleton at radius birth + r_frac * lifetime (just after
    the class is born, so the loop uses short edges and its vertex count is meaningful) that pairs
    nontrivially with the cocycle: for each support edge (u, v), the shortest v->u path avoiding
    all support edges, plus (u, v). Returns vertex indices in loop order. A geometric (not
    optimal-homologous) representative; Phase 5 may swap in an optimal-cycle library here."""
    n = len(D)
    edges = _edges_at(D, cc.birth + r_frac * (cc.death - cc.birth))
    _, z = _lift(cc.zmap, edges, cc.prime)
    support = np.flatnonzero(z != 0)
    if len(support) == 0:
        return []
    keep = np.ones(len(edges), bool)
    keep[support] = False
    e = edges[keep]
    w = D[e[:, 0], e[:, 1]] + 1e-12
    G = csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    order = support[np.argsort(D[edges[support, 0], edges[support, 1]])][:max_seeds]
    best, best_len = [], np.inf
    for k in order:
        u, v = map(int, edges[k])
        dist, pred = dijkstra(G, indices=v, return_predecessors=True)
        if not np.isfinite(dist[u]):
            continue
        L = dist[u] + D[u, v]
        if L < best_len:
            path, x = [u], u
            while x != v:
                x = int(pred[x])
                path.append(x)
            best, best_len = path[::-1], L
    return best

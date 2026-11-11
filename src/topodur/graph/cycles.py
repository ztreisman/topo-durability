"""Long-cycle mining in sparse latent graphs (Phase 5).

Flag (clique) complex persistence up to H1 on a sparse distance graph, computed with gudhi
(edge collapse, then expansion to triangles). Representative of an H1 class: its birth edge (u, v)
plus the shortest u-v path in the graph filtered at the birth value without that edge, a standard
cheap representative that is a cycle in the class's birth complex. Score = vertex count x lifetime.

Nulls (spec Phase 5): degree-preserving rewiring (double edge swaps; distances are shuffled across
edges), and the same with every swap that changes the triangle count rejected. Random clique
complexes have nonzero H1 at some densities (Kahle 2009), so a cycle counts only if its lifetime and
its score beat the 95th percentile of the null maxima under both nulls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

log = logging.getLogger(__name__)


@dataclass
class Cycle:
    birth: float
    death: float
    vertices: list[int]

    @property
    def lifetime(self) -> float:
        return self.death - self.birth

    @property
    def score(self) -> float:
        return len(self.vertices) * self.lifetime


def edges_of(D: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U = sp.triu(D, k=1).tocoo()
    return U.row, U.col, U.data


def h1_pairs(D: sp.csr_matrix, collapse: int = 2, max_edge: float | None = None):
    """[(birth, death, (u, v) birth edge)] for H1, death = inf for classes that never die."""
    import gudhi

    r, c, w = edges_of(D)
    if max_edge is not None:
        keep = w <= max_edge
        r, c, w = r[keep], c[keep], w[keep]
    st = gudhi.SimplexTree()
    for i in range(D.shape[0]):
        st.insert([i], 0.0)
    st.insert_batch(np.vstack([r, c]).astype(np.int32), w.astype(np.float64)) if hasattr(st, "insert_batch") else [
        st.insert([int(a), int(b)], float(x)) for a, b, x in zip(r, c, w)]
    if collapse:
        st.collapse_edges(collapse)
    st.expansion(2)
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
    out = []
    for birth_s, death_s in st.persistence_pairs():
        if len(birth_s) != 2:
            continue
        b = st.filtration(birth_s)
        d = st.filtration(death_s) if death_s else np.inf
        if d > b:
            out.append((b, d, (int(birth_s[0]), int(birth_s[1]))))
    return out


def representative(D: sp.csr_matrix, birth: float, edge: tuple[int, int]) -> list[int]:
    u, v = edge
    A = D.copy().tolil()
    A[u, v] = 0
    A[v, u] = 0
    A = A.tocsr()
    A.data[A.data > birth + 1e-12] = 0
    A.eliminate_zeros()
    _, pred = dijkstra(A, directed=False, indices=u, return_predecessors=True, unweighted=True)
    if pred[v] < 0:
        return [u, v]
    path = [v]
    while path[-1] != u:
        path.append(int(pred[path[-1]]))
    return path[::-1]


def mine_cycles(D: sp.csr_matrix, top: int = 200, **kw) -> list[Cycle]:
    pairs = h1_pairs(D, **kw)
    finite_max = max((d for _, d, _ in pairs if np.isfinite(d)), default=1.0)
    cyc = []
    for b, d, e in pairs:
        dd = d if np.isfinite(d) else max(finite_max, b) + 1e-6
        cyc.append(Cycle(b, dd, representative(D, b, e)))
    cyc.sort(key=lambda c: -c.score)
    return cyc[:top]


def is_chordless(D: sp.csr_matrix, verts: list[int], at: float) -> bool:
    k = len(verts)
    if k < 4:
        return True
    pos = {v: i for i, v in enumerate(verts)}
    sub = D[verts][:, verts].tocoo()
    for a, b, w in zip(sub.row, sub.col, sub.data):
        if w <= at and a < b:
            gap = abs(int(a) - int(b))
            if gap not in (1, k - 1):
                return False
    return True


# ------------------------------------------------------------------ nulls

def _adj_sets(r, c, n):
    adj = [set() for _ in range(n)]
    for a, b in zip(r, c):
        adj[a].add(b)
        adj[b].add(a)
    return adj


def rewire(D: sp.csr_matrix, n_swaps_per_edge: int = 10, preserve_triangles: bool = False, seed: int = 0,
           tol: int = 0) -> sp.csr_matrix:
    """Degree-preserving double edge swaps; edge distances are shuffled across the rewired edges.
    With preserve_triangles, a swap is kept only if the global triangle count changes by <= tol."""
    rng = np.random.default_rng(seed)
    r, c, w = edges_of(D)
    n = D.shape[0]
    E = np.vstack([r, c]).T.copy()
    adj = _adj_sets(r, c, n)
    m = len(E)
    tri_delta = 0
    accepted = 0
    for _ in range(n_swaps_per_edge * m):
        i, j = rng.integers(m, size=2)
        if i == j:
            continue
        a, b = E[i]
        x, y = E[j]
        if rng.random() < 0.5:
            x, y = y, x
        if len({a, b, x, y}) < 4 or y in adj[a] or b in adj[x]:
            continue
        if preserve_triangles:
            before = len(adj[a] & adj[b]) + len(adj[x] & adj[y])
            adj[a].discard(b); adj[b].discard(a); adj[x].discard(y); adj[y].discard(x)
            after = len(adj[a] & adj[y]) + len(adj[x] & adj[b])
            adj[a].add(b); adj[b].add(a); adj[x].add(y); adj[y].add(x)
            if abs(tri_delta + after - before) > tol:
                continue
            tri_delta += after - before
        adj[a].discard(b); adj[b].discard(a); adj[x].discard(y); adj[y].discard(x)
        adj[a].add(y); adj[y].add(a); adj[x].add(b); adj[b].add(x)
        E[i] = (a, y)
        E[j] = (x, b)
        accepted += 1
    ws = rng.permutation(w)
    N = sp.coo_matrix((ws, (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    log.info("rewire: %d/%d swaps accepted (triangles %s)", accepted, n_swaps_per_edge * m,
             "preserved" if preserve_triangles else "free")
    return N.maximum(N.T)


def null_maxima(D: sp.csr_matrix, n_null: int = 20, preserve_triangles: bool = False, seed: int = 0,
                **kw) -> np.ndarray:
    """[n_null, 2] max lifetime and max score of H1 cycles in rewired graphs."""
    out = []
    for s in range(n_null):
        N = rewire(D, preserve_triangles=preserve_triangles, seed=seed + s)
        cyc = mine_cycles(N, top=50, **kw)
        out.append([max((c.lifetime for c in cyc), default=0.0), max((c.score for c in cyc), default=0.0)])
    return np.array(out)

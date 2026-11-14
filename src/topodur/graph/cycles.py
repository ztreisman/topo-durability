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
    """Shortest (unweighted) u -> v path in D filtered at `birth`, without the edge (u, v) itself.
    Search directly on the CSR arrays (no copy of the graph per class: the copy made this O(classes x edges)),
    bidirectional: the smaller frontier is expanded one whole level at a time, so it visits two balls of
    radius ~L/2 instead of one of radius L (on large graphs the one-sided search dominated mining time)."""
    u, v = edge
    D = D.tocsr()
    ip, ind, dat = D.indptr, D.indices, D.data
    lim = birth + 1e-12
    pred = ({u: -1}, {v: -1})
    dist = ({u: 0}, {v: 0})
    front = ([u], [v])

    def nbrs(x):
        s, e = ip[x], ip[x + 1]
        return ind[s:e][dat[s:e] <= lim].tolist()

    while front[0] and front[1]:
        s = 0 if len(front[0]) <= len(front[1]) else 1
        o = 1 - s
        nxt, best, best_len = [], None, None
        for x in front[s]:
            for y in nbrs(x):
                if {x, y} == {u, v} or y in pred[s]:
                    continue
                pred[s][y] = x
                dist[s][y] = dist[s][x] + 1
                nxt.append(y)
                if y in pred[o]:
                    L = dist[s][y] + dist[o][y]
                    if best is None or L < best_len:
                        best, best_len = y, L
        if best is not None:      # whole level done: best is a shortest meeting point
            def walk(p, x):
                out = [x]
                while p[out[-1]] != -1:
                    out.append(p[out[-1]])
                return out
            a_, b_ = walk(pred[0], best), walk(pred[1], best)       # best ... u   and   best ... v
            return a_[::-1] + b_[1:]
        front = (nxt, front[1]) if s == 0 else (front[0], nxt)
    return [u, v]


def _representative_slow(D: sp.csr_matrix, birth: float, edge: tuple[int, int]) -> list[int]:
    """Reference implementation (kept for tests)."""
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


def mine_cycles(D: sp.csr_matrix, top: int = 200, max_candidates: int = 5000, **kw) -> list[Cycle]:
    """Top H1 cycles by score. Representatives are built only for the max_candidates longest-lived
    classes: large graphs have tens of thousands of short-lived classes, and building each one's path
    dominated run time (12k nodes: 38k classes, 160 s). Nulls go through the same cap."""
    pairs = h1_pairs(D, **kw)
    finite_max = max((d for _, d, _ in pairs if np.isfinite(d)), default=1.0)
    if len(pairs) > max_candidates:
        life = np.array([(d if np.isfinite(d) else max(finite_max, b) + 1e-6) - b for b, d, _ in pairs])
        pairs = [pairs[i] for i in np.argsort(-life, kind="stable")[:max_candidates]]
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
           tol: int = 0, node_block: np.ndarray | None = None) -> sp.csr_matrix:
    """Degree-preserving double edge swaps; edge distances are shuffled across the rewired edges.
    With preserve_triangles, a swap is kept only if the global triangle count changes by <= tol.
    With node_block (e.g. each neuron's layer), swaps only pair edges joining the same two blocks, oriented
    so each endpoint stays in its block: every node keeps its degree into every block, every block pair keeps
    its edge count, and distances are shuffled only within a block pair (a layered null)."""
    rng = np.random.default_rng(seed)
    r, c, w = edges_of(D)
    n = D.shape[0]
    if node_block is not None:
        nb = np.asarray(node_block)
        flip = nb[r] > nb[c]
        r, c = np.where(flip, c, r), np.where(flip, r, c)
        _, gid = np.unique(nb[r].astype(np.int64) * (int(nb.max()) + 1) + nb[c], return_inverse=True)
        order = np.argsort(gid, kind="stable")
        bounds = np.searchsorted(gid[order], np.arange(gid.max() + 2))
        members = [order[bounds[g]:bounds[g + 1]] for g in range(gid.max() + 1)]
        same = nb[r] == nb[c]
    E = np.vstack([r, c]).T.copy()
    adj = _adj_sets(r, c, n)
    m = len(E)
    tri_delta = 0
    accepted = 0
    for _ in range(n_swaps_per_edge * m):
        if node_block is None:
            i, j = rng.integers(m, size=2)
            can_flip = True
        else:
            i = int(rng.integers(m))
            grp = members[gid[i]]
            j = int(grp[rng.integers(len(grp))])
            can_flip = bool(same[i])
        if i == j:
            continue
        a, b = E[i]
        x, y = E[j]
        if can_flip and rng.random() < 0.5:
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
    if node_block is None:
        ws = rng.permutation(w)
    else:
        ws = w.copy()
        for grp in members:
            ws[grp] = rng.permutation(w[grp])
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

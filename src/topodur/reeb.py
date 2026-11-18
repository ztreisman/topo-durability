"""Fork/rejoin structure of one token's neuron footprint across layers (decided with Zack 2026-11-15).

At each layer the neurons a token activates are grouped into clusters (the layer's H0). Clusters at adjacent
layers are linked when their activity is coupled. The resulting layered graph is the token's Reeb graph: a fork
is a cluster linked to several clusters at the next layer, a rejoin is several clusters linked into one, and a
fork followed by a rejoin is a loop. A layer cluster born by a split and ended by a merge lives on an open layer
interval, which in level-set zigzag terms is a class of H1 of the whole footprint: layer H0 becomes concept H1.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def panel(path, calib: dict | None = None) -> dict[str, list[str]]:
    import yaml

    cfg = yaml.load(open(path), Loader=yaml.BaseLoader)   # all strings: YAML 1.1 reads "on" as True
    out = {k: list(v) for k, v in cfg.items() if k != "families"}
    if calib is not None:
        from .concepts import load_family

        for f in cfg.get("families", []):
            fam = load_family(f)
            topo = "ring_items" if fam.topology != "interval" else "line_items"
            out.setdefault(topo, []).extend(fam.items[i] for i, _ in calib["families"].get(f, []))
    return out


def active_neurons(z: np.ndarray, z_min: float = 3.0, max_n: int = 128) -> np.ndarray:
    """Neurons the token drives: z-score of its response above z_min, at most max_n strongest."""
    idx = np.flatnonzero(z > z_min)
    return idx[np.argsort(-z[idx])][:max_n]


def clusters(C: np.ndarray, tau: float, min_size: int = 2) -> list[np.ndarray]:
    """Connected components of the graph 'correlation > tau' among the active neurons (C = their correlation
    submatrix). Components smaller than min_size are left out (unclustered neurons)."""
    A = sp.csr_matrix((C > tau) & ~np.eye(len(C), dtype=bool))
    n, lab = connected_components(A, directed=False)
    return [np.flatnonzero(lab == k) for k in range(n) if (lab == k).sum() >= min_size]


def reeb_stats(layers: list[int], cl: dict[int, list], links: dict[tuple, float], w_min: float, mass: dict) -> dict:
    """cl[l] = clusters at layer l; links[(l, i, j)] = coupling of cluster i at l with cluster j at l+1;
    mass[l] = activation mass per cluster. Returns per-token Reeb graph statistics."""
    import networkx as nx

    G = nx.Graph()
    for l in layers:
        for i in range(len(cl[l])):
            G.add_node((l, i))
    for (l, i, j), w in links.items():
        if w >= w_min:
            G.add_edge((l, i), (l + 1, j), w=w)
    forks = sum(1 for (l, i) in G.nodes if sum(1 for nb in G[(l, i)] if nb[0] == l + 1) >= 2)
    rejoins = sum(1 for (l, i) in G.nodes if sum(1 for nb in G[(l, i)] if nb[0] == l - 1) >= 2)
    b0 = nx.number_connected_components(G) if len(G) else 0
    b1 = G.number_of_edges() - G.number_of_nodes() + b0
    spans = sorted((max(n[0] for n in c) - min(n[0] for n in c) for c in nx.cycle_basis(G)), reverse=True)
    ent = {}
    for l in layers:
        m = np.asarray(mass.get(l, []), float)
        p = m / m.sum() if m.sum() > 0 else m
        ent[l] = float(-(p[p > 0] * np.log(p[p > 0])).sum()) if len(p) else 0.0
    return {"clusters": {l: len(cl[l]) for l in layers}, "entropy": ent, "forks": forks, "rejoins": rejoins,
            "b0": b0, "b1": b1, "loop_spans": spans[:20]}


# ------------------------------------------------------------------ senses: clusters of a token's contexts

def gap_k(Y: np.ndarray, kmax: int = 6, n_ref: int = 10, min_size: int = 8, seed: int = 0, ref_inits: int = 3):
    """Number of context clusters (senses) in Y [n_ctx, d], by the gap statistic (Tibshirani et al. 2001) against
    a single Gaussian blob with Y's per-axis spread (Y should be in PCA coordinates, so the axes are uncorrelated).
    k > 1 only when Y splits better than one blob of the same shape. Clusters smaller than min_size are dropped.
    Returns (k, list of index arrays)."""
    from sklearn.cluster import KMeans

    n, d = Y.shape
    sdv = Y.std(0)
    if n < 2 * min_size or not np.isfinite(Y).all() or sdv.sum() < 1e-8:
        return 1, [np.arange(n)]

    def fit(X, k, inits=3):
        if k == 1:
            return float(((X - X.mean(0)) ** 2).sum()), np.zeros(len(X), int)
        km = KMeans(k, n_init=inits, random_state=seed).fit(X)
        return float(km.inertia_), km.labels_

    kmax = max(1, min(kmax, n // min_size))
    ks = range(1, kmax + 1)
    real = [fit(Y, k) for k in ks]
    logw = np.log([max(r[0], 1e-12) for r in real])
    rng = np.random.default_rng(seed)
    ref = np.log([[max(fit(rng.normal(size=(n, d)) * sdv, k, ref_inits)[0], 1e-12) for k in ks] for _ in range(n_ref)])
    gap = ref.mean(0) - logw
    s = ref.std(0) * np.sqrt(1 + 1 / n_ref)
    k = next((k for k in range(1, kmax) if gap[k - 1] >= gap[k] - s[k]), kmax) if kmax > 1 else 1
    lab = real[k - 1][1]
    groups = [np.flatnonzero(lab == c) for c in range(k)]
    groups = [g for g in groups if len(g) >= min_size] or [np.arange(n)]
    return len(groups), groups


def context_links(prev: list, nxt: list) -> dict[tuple, float]:
    """Overlap of context clusters at adjacent layers: |A & B| / min(|A|, |B|)."""
    out = {}
    for i, A in enumerate(prev):
        for j, B in enumerate(nxt):
            o = len(np.intersect1d(A, B, assume_unique=True))
            if o:
                out[(i, j)] = o / min(len(A), len(B))
    return out


def pcs(X: np.ndarray, d: int = 10) -> np.ndarray:
    """Top-d principal-component scores of the rows of X (contexts x neurons), centred. Many skewed neurons mix
    into near-Gaussian components, so a unimodal skewed footprint stays one blob for gap_k (tests)."""
    X = np.asarray(X, np.float64)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :d] * S[:d]


def sense_barcode(labels: np.ndarray, jaccard: float = 0.5, trivial: float = 0.9) -> list[tuple[int, int]]:
    """Persistence of sense clusters across layers. labels [n_layers, n_ctx] (-1 = unassigned). A cluster at
    layer i continues at layer i+1 when they share >= `jaccard` of their union (with jaccard >= 0.5 the partner
    is unique), so random re-partitioning gives short bars. Clusters holding >= `trivial` of the contexts are the
    whole token, not a sense, and are left out. Returns bars as (first layer index, length in layers)."""
    nL, n = labels.shape
    sets = [[np.flatnonzero(labels[i] == c) for c in np.unique(labels[i]) if c >= 0] for i in range(nL)]
    sets = [[s for s in S if len(s) < trivial * n] for S in sets]
    bars, alive = [], {}                                  # alive: cluster index at layer i -> (start, length)
    for i in range(nL):
        new = {}
        for j, B in enumerate(sets[i]):
            src = None
            if i > 0:
                for k, A in enumerate(sets[i - 1]):
                    inter = len(np.intersect1d(A, B, assume_unique=True))
                    if inter and inter / (len(A) + len(B) - inter) >= jaccard and k in alive:
                        src = k
                        break
            if src is not None:
                st, ln = alive.pop(src)
                new[j] = (st, ln + 1)
            else:
                new[j] = (i, 1)
        bars += list(alive.values())
        alive = new
    bars += list(alive.values())
    return sorted(bars)

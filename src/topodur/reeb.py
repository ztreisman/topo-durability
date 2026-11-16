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

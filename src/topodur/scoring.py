"""Phase 2 scoring of one family at one layer under one preprocessing setting -> one tidy row."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import nulls, tda
from .concepts import Family
from .extract import Background, FamilyActs
from .geometry import Prep, distance_matrix, participation_ratio, preprocess


@dataclass
class LayerResult:
    row: dict
    Z: np.ndarray                 # preprocessed centroids [I, d]
    dgms: list
    theta: np.ndarray | None
    cycle: list[int]


def cyclic_order_exact(theta: np.ndarray, truth: np.ndarray, tie_tol: float = 1e-6) -> bool:
    """True if sorting items by theta gives the ground-truth cyclic order (any rotation/reflection)."""
    n = len(theta)
    s = np.sort(np.mod(theta, 2 * np.pi))
    gaps = np.diff(np.r_[s, s[0] + 2 * np.pi])
    if (gaps < tie_tol).any():
        return False  # tied coordinates: the order between those items is undetermined
    seq = list(np.argsort(np.argsort(truth))[np.argsort(theta)])  # true ranks in theta order
    start = seq.index(0)
    rot = seq[start:] + seq[:start]
    fwd = list(range(n))
    rev = [0] + list(range(n - 1, 0, -1))
    return rot == fwd or rot == rev


def closure_ratio(D: np.ndarray, truth: np.ndarray) -> float:
    """Distance between the two ends of the ground-truth order (last -> first item) divided by the
    median distance between order-adjacent items. About 1 for a closed loop; large for a line.
    Separates circles from lines, which order_score alone cannot (an ordered line wrapped onto a
    circle also gets a high circular rank correlation)."""
    o = np.argsort(truth, kind="stable")
    steps = D[o[:-1], o[1:]]
    return float(D[o[-1], o[0]] / max(np.median(steps), 1e-12))


def topology_score(row: dict, order_max_items: int) -> tuple[float, str]:
    """Primary topology measure (decided 2026-11-10): order_score for families with at most
    order_max_items items (h1_ratio has little power against the Gaussian null at that size);
    h1_ratio for larger families (cities, and so on)."""
    if row["n_items"] <= order_max_items:
        return row.get("order_score", np.nan), "order_score"
    return row["h1_ratio"], "h1_ratio"


def score_layer(
    fam: Family,
    acts: FamilyActs,
    bg: Background,
    li: int,
    prep: Prep,
    n_boot: int = 200,
    n_perm: int = 2000,
    n_gauss: int = 200,
    seed: int = 0,
    order_max_items: int = 30,
) -> LayerResult:
    layer = acts.layers[li]
    bli = bg.layers.index(layer)
    mean, pcs = bg.mean[bli], bg.pcs[bli]
    maxdim = 2 if fam.topology == "torus" else 1

    def to_Z(C):
        return preprocess(C, mean, pcs, prep)

    def to_D(C):
        return distance_matrix(to_Z(C), prep.metric)

    C = acts.centroids[prep.variant][li]
    Z = to_Z(C)
    D = distance_matrix(Z, prep.metric)
    dgms = tda.diagrams(D, maxdim=maxdim)["dgms"]
    row = {"family": fam.name, "topology": fam.topology, "origin": fam.origin, "layer": layer,
           **prep.asdict(), "n_items": fam.n_items, "n_contexts": int(acts.counts.min())}
    row.update(tda.scores(dgms))
    row["closure_ratio"] = closure_ratio(D, fam.angles())

    if n_boot:
        band = tda.bootstrap_band(acts.instances[prep.variant][li], acts.n_inst, to_D, dgms,
                                  maxdim=maxdim, n_boot=n_boot, seed=seed)
        for d in range(1, maxdim + 1):
            row[f"c_h{d}"] = band.c[d]
            row[f"n_sig_h{d}"] = tda.n_significant(dgms[d], band.threshold(d))

    theta, cycle = None, []
    cc = tda.circular_coordinates(D) if fam.n_items >= 4 else None
    if cc is not None:
        theta = cc.theta
        obs, p = tda.order_test(theta, fam.angles(), n_perm=n_perm, seed=seed)
        row.update(order_score=obs, order_p=p, order_exact=cyclic_order_exact(theta, fam.angles()),
                   circ_connected=cc.connected, cocycle_ok=cc.cocycle_ok)
        cycle = tda.representative_cycle(D, cc)
        row["cycle_len"] = len(cycle)
    else:
        row.update(order_score=np.nan, order_p=np.nan, order_exact=False)

    if n_gauss:
        null = nulls.gaussian_null(Z, prep.metric, maxdim, n_gauss, seed)
        row["gauss_p_h1_ratio"] = nulls.empirical_p(null["h1_ratio"], row["h1_ratio"])
        row["gauss_p_h1_max"] = nulls.empirical_p(null["h1_max"], row["h1_max"])

    Yc = C - mean
    row["act_norm"] = float(np.linalg.norm(Yc, axis=1).mean())
    row["participation_ratio"] = participation_ratio(Yc)
    row["item_n_tokens_mean"] = float(acts.item_n_tokens.mean())
    row["topology_score"], row["topology_measure"] = topology_score(row, order_max_items)
    return LayerResult(row, Z, dgms, theta, cycle)


def gate_pass(row: dict, alpha: float = 0.05) -> bool:
    """Phase 0 replication gate: a significant H1 class whose circular coordinate puts the items
    in the correct cyclic order (exact order, and permutation p < alpha)."""
    return bool(row.get("n_sig_h1", 0) >= 1 and row.get("order_exact") and row.get("order_p", 1) < alpha)

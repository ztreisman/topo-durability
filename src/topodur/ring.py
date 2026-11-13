"""Label-free ring detection: discover a loop on half of the contexts, confirm it on the other half.

Decided with Zack 2026-11-13: the primary topology score must not use the true order, so concept
families, SAE-graph loops and toy-model loops are measured with the same instrument. The true order is
used only to VALIDATE the detector (calibration), never to score.

Per family x layer x dial setting:
  1. Preprocess every stored context vector (background mean, remove k background PCs, unit norm).
  2. Split each item's contexts in two halves, A and B (fixed seed).
  3. Point cloud per half: mini-centroids of `group` contexts of one item (group=0: one centroid per
     item, the old pipeline), at most `cap` points in total.
  4. Project both halves onto the top-`d` PCs of half A's item centroids (the "scope" is set on A).
  5. Rips H1 per half. Null: keep each point's offset from its item centroid, redraw the item
     centroids from a Gaussian with the centroids' covariance (destroys the arrangement of items,
     keeps the within-item scatter); statistic = longest H1 lifetime; p = P(null max >= observed).
  6. Circular coordinate of each half's longest bar -> each item's angle (circular mean of its
     points), item coherence (mean resultant length) and coverage (largest gap between item angles).
  7. Reproducibility (label-free): circular rank correlation of item angles from A vs from B, with a
     permutation p over items.
  8. Validation only: circular rank correlation of A's item angles with the true order.

`detected` = p_A < alpha and p_B < alpha and rep_p < alpha.
"""

from __future__ import annotations

import numpy as np

from . import tda
from .geometry import distance_matrix


def preprocess_points(X: np.ndarray, bg_mean: np.ndarray, bg_pcs: np.ndarray, k_remove: int) -> np.ndarray:
    Y = X.astype(np.float64) - bg_mean
    if k_remove:
        P = bg_pcs[:k_remove].astype(np.float64)
        Y = Y - (Y @ P.T) @ P
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def split_halves(n_inst: np.ndarray, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    for n in n_inst:
        p = rng.permutation(int(n))
        out.append((p[: n // 2], p[n // 2:]))
    return out


def half_cloud(inst: np.ndarray, idx: list[np.ndarray], group: int, cap: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """inst: [I, n, D] preprocessed; idx: per-item row indices of this half. Returns points, item ids."""
    I = len(idx)
    per = max(1, cap // I)
    pts, ids = [], []
    for i, rows in enumerate(idx):
        rows = rng.permutation(rows)
        if group == 0 or len(rows) < group:
            groups = [rows]
        else:
            groups = [rows[j:j + group] for j in range(0, len(rows) - group + 1, group)]
        if len(groups) > per:
            groups = [groups[j] for j in rng.choice(len(groups), per, replace=False)]
        for g in groups:
            pts.append(inst[i, g].mean(0))
            ids.append(i)
    return np.asarray(pts), np.asarray(ids)


def item_means(P: np.ndarray, ids: np.ndarray, I: int) -> np.ndarray:
    return np.stack([P[ids == i].mean(0) for i in range(I)])


def plant(Z: np.ndarray, ids: np.ndarray, I: int, shape: str, amp: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic calibration case in real data: replace the item centroids' first two coordinates by a ring
    (or a half-circle arc) whose RMS radius is `amp` x the RMS of those two coordinates. The other
    coordinates of the centroids (real nuisance structure) and every point's offset from its item
    centroid (real context scatter) are kept. Returns Z', planted angles."""
    mu = item_means(Z, ids, I)
    m2 = mu[:, :2] - mu[:, :2].mean(0)
    r = amp * float(np.sqrt((m2 ** 2).sum(1).mean()))
    ang = 2 * np.pi * np.arange(I) / I if shape == "ring" else np.pi * np.arange(I) / max(I - 1, 1)
    new = mu.copy()
    new[:, 0], new[:, 1] = r * np.cos(ang), r * np.sin(ang)
    return Z - mu[ids] + new[ids], ang


def null_max_lifetime(Z: np.ndarray, ids: np.ndarray, I: int, metric: str, n: int, rng) -> np.ndarray:
    mu = item_means(Z, ids, I)
    R = Z - mu[ids]
    muc = mu - mu.mean(0)
    U, S, Vt = np.linalg.svd(muc, full_matrices=False)
    sd = S / np.sqrt(max(I - 1, 1))
    out = np.zeros(n)
    for b in range(n):
        G = (rng.standard_normal((I, len(sd))) * sd) @ Vt
        out[b] = tda.lifetimes(tda.diagrams(distance_matrix(G[ids] + R, metric))["dgms"][1]).max(initial=0.0)
    return out


def circ_mean(theta: np.ndarray) -> tuple[float, float]:
    z = np.exp(1j * theta).mean()
    return float(np.angle(z) % (2 * np.pi)), float(abs(z))


def analyze_half(Z: np.ndarray, ids: np.ndarray, I: int, metric: str, n_null: int, rng) -> dict:
    D = distance_matrix(Z, metric)
    lt = tda.lifetimes(tda.diagrams(D)["dgms"][1])
    top = float(lt.max(initial=0.0))
    null = null_max_lifetime(Z, ids, I, metric, n_null, rng)
    out = {"top_life": top, "null_q95": float(np.quantile(null, 0.95)), "p": float((1 + (null >= top).sum()) / (1 + n_null)),
           "n_bars_over_null": int((lt > np.quantile(null, 0.95)).sum()), "item_theta": None}
    cc = tda.circular_coordinates(D) if top > 0 else None
    if cc is not None:
        th = np.array([circ_mean(cc.theta[ids == i]) for i in range(I)])
        out["item_theta"], out["coherence"] = th[:, 0], float(th[:, 1].mean())
        s = np.sort(th[:, 0])
        out["max_gap_frac"] = float(np.diff(np.r_[s, s[0] + 2 * np.pi]).max() / (2 * np.pi))
    return out


def ring_task(inst: np.ndarray, n_inst: np.ndarray, bg_mean: np.ndarray, bg_pcs: np.ndarray, truth: np.ndarray, *,
              k_remove: int = 1, d: int = 5, group: int = 0, cap: int = 360, metric: str = "cosine",
              n_null: int = 40, n_perm: int = 1000, planted: str | None = None, amp: float = 1.0, alpha: float = 0.05,
              seed: int = 0) -> dict:
    """inst: [I, n, D] raw stored context vectors of one family at one layer."""
    rng = np.random.default_rng(seed)
    I, n, Dm = inst.shape
    X = np.zeros((I, n, Dm))
    for i in range(I):
        X[i, : n_inst[i]] = preprocess_points(inst[i, : n_inst[i]], bg_mean, bg_pcs, k_remove)
    halves = split_halves(n_inst, seed)
    PA, idA = half_cloud(X, [h[0] for h in halves], group, cap, rng)
    PB, idB = half_cloud(X, [h[1] for h in halves], group, cap, rng)
    muA = item_means(PA, idA, I)
    c = muA.mean(0)
    dd = min(d, I - 1)
    Vt = np.linalg.svd(muA - c, full_matrices=False)[2][:dd]
    ZA, ZB = (PA - c) @ Vt.T, (PB - c) @ Vt.T
    if planted:
        ZA, truth = plant(ZA, idA, I, planted, amp)
        ZB, _ = plant(ZB, idB, I, planted, amp)
    a = analyze_half(ZA, idA, I, metric, n_null, rng)
    b = analyze_half(ZB, idB, I, metric, n_null, rng)
    row = {"n_points_A": len(ZA), "pA": a["p"], "pB": b["p"], "top_life_A": a["top_life"], "top_life_B": b["top_life"],
           "null_q95_A": a["null_q95"], "n_bars_over_null_A": a["n_bars_over_null"],
           "coherence_A": a.get("coherence", np.nan), "max_gap_frac_A": a.get("max_gap_frac", np.nan)}
    rep = rep_p = val = val_p = np.nan
    if a["item_theta"] is not None and b["item_theta"] is not None and I >= 4:
        rep, rep_p = tda.order_test(a["item_theta"], b["item_theta"], n_perm=n_perm, seed=seed)
        val, val_p = tda.order_test(a["item_theta"], truth, n_perm=n_perm, seed=seed)
    row.update(rep=rep, rep_p=rep_p, val_order=val, val_order_p=val_p)
    row["detected"] = bool(a["p"] < alpha and b["p"] < alpha and np.nan_to_num(rep_p, nan=1) < alpha)
    return row

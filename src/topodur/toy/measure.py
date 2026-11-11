"""Measurements on trained toy models, per seed and per substructure (Phase 4).

geometry   H1 of the substructure's feature directions (normalized columns of W, cosine distance),
           order score of their circular coordinate against the true index order, closure ratio
capacity   interference sum_{j != i} (W_i_hat . W_j)^2 and Elhage dimensionality
           D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2, both averaged over the substructure
durability loss increase and geometry retention under weight noise, pruning one hidden dimension,
           and continued training on independent features with the same marginal rates
"""

from __future__ import annotations

import numpy as np
import torch

from .. import tda
from ..geometry import procrustes_distance
from ..scoring import closure_ratio
from .tms import ToyConfig, build_groups, per_feature_loss, sample, train


def _dist(V: np.ndarray) -> np.ndarray:
    U = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return np.clip(1 - U @ U.T, 0, 2)


def geometry(Wn: np.ndarray, feats: list[int], kind: str | None, min_norm: float = 0.3, n_perm: int = 500,
             seed: int = 0) -> dict:
    """Wn: (m, n) one seed's W. Uses the substructure's represented features (column norm >= min_norm)."""
    cols = Wn[:, feats].T                                      # (k, m)
    norms = np.linalg.norm(cols, axis=1)
    rep = norms >= min_norm * max(norms.max(), 1e-12)
    out = {"n_feats": len(feats), "n_represented": int(rep.sum())}
    if rep.sum() < 4:
        return out
    V = cols[rep]
    D = _dist(V)
    out.update(tda.scores(tda.diagrams(D, maxdim=1)["dgms"]))
    if kind in ("cycle", "line"):
        truth = 2 * np.pi * np.arange(len(feats))[rep] / len(feats)
        cc = tda.circular_coordinates(D)
        if cc is not None:
            out["order_score"], out["order_p"] = tda.order_test(cc.theta, truth, n_perm=n_perm, seed=seed)
        out["closure_ratio"] = closure_ratio(D, truth)
    return out


def capacity(Wn: np.ndarray, feats: list[int]) -> dict:
    G = Wn.T @ Wn                                               # (n, n)
    norms = np.sqrt(np.maximum(np.diag(G), 1e-12))
    Ghat = G / norms[:, None]                                   # row i: W_i_hat . W_j
    inter = (Ghat ** 2).sum(1) - np.diag(Ghat) ** 2
    dim = norms ** 2 / np.maximum((Ghat ** 2).sum(1), 1e-12)
    f = np.array(feats)
    return {"interference": float(inter[f].mean()), "dimensionality": float(dim[f].mean()),
            "dims_used": float(dim[f].sum()), "col_norm": float(norms[f].mean())}


def _eval(cfg, groups, W, b, feats, n_eval, gen):
    X = sample(cfg, groups, W.shape[0], n_eval, gen)
    with torch.no_grad():
        return per_feature_loss(W, b, X)[:, feats].sum(1)      # (S,)


def durability(cfg: ToyConfig, W, b, sub: dict, n_eval: int = 20_000, noise=(0.1, 0.3),
               retrain_steps: int = 2000, seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Per substructure name -> metric -> (S,) array."""
    gen = torch.Generator().manual_seed(seed + 11)
    groups = build_groups(cfg)
    out: dict = {k: {} for k in sub}
    base = {k: _eval(cfg, groups, W, b, f, n_eval, torch.Generator().manual_seed(seed)) for k, f in sub.items()}
    for k in sub:
        out[k]["loss"] = base[k].numpy()
    rms = W.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
    for s in noise:
        Wz = W + s * rms * torch.randn(W.shape, generator=gen)
        for k, f in sub.items():
            L = _eval(cfg, groups, Wz, b, f, n_eval, torch.Generator().manual_seed(seed))
            out[k][f"dloss_noise{s:g}"] = ((L - base[k]) / base[k].clamp_min(1e-9)).numpy()
            out[k][f"procrustes_noise{s:g}"] = np.array(
                [procrustes_distance(W[i][:, f].T.numpy(), Wz[i][:, f].T.numpy()) for i in range(W.shape[0])])
    # prune one hidden dimension: average over which row is zeroed
    acc = {k: torch.zeros(W.shape[0]) for k in sub}
    for r in range(cfg.m):
        Wp = W.clone()
        Wp[:, r, :] = 0
        for k, f in sub.items():
            L = _eval(cfg, groups, Wp, b, f, n_eval, torch.Generator().manual_seed(seed))
            acc[k] += (L - base[k]) / base[k].clamp_min(1e-9)
    for k in sub:
        out[k]["dloss_prune1"] = (acc[k] / cfg.m).numpy()
    # continued training on independent features with the same marginal rates
    if retrain_steps:
        ind = ToyConfig(**{**cfg.__dict__, "structure": "independent"})
        W2, b2, _ = train(ind, W, b, steps=retrain_steps, gen=torch.Generator().manual_seed(seed + 13))
        for k, f in sub.items():
            L = _eval(cfg, groups, W2, b2, f, n_eval, torch.Generator().manual_seed(seed))
            out[k]["dloss_retrain"] = ((L - base[k]) / base[k].clamp_min(1e-9)).numpy()
            out[k]["procrustes_retrain"] = np.array(
                [procrustes_distance(W[i][:, f].T.numpy(), W2[i][:, f].T.numpy()) for i in range(W.shape[0])])
        out["_retrained"] = (W2, b2)
    return out

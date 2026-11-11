"""Preprocessing of family centroids (Phase 1), in the spec's order:

  1. center with the background mean at that layer
  2. remove the top-k background PCs, k in {0, 1, 3}
  3. normalize to unit length
  4. project the family's centroids onto their own top-d PCs, d in {3, 5, 8}

The family PCA (step 4) is fit on the centroids being analysed. For bootstraps, the
whole pipeline is re-run on each resampled centroid set (see tda.bootstrap_band).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True)
class Prep:
    k_remove: int = 1
    normalize: bool = True
    d_proj: int = 5
    metric: str = "cosine"     # distance used for Rips after projection: cosine | euclidean
    variant: str = "last"      # token variant: last | mean

    def tag(self) -> str:
        return f"k{self.k_remove}_n{int(self.normalize)}_d{self.d_proj}_{self.metric}_{self.variant}"

    def asdict(self) -> dict:
        return asdict(self)


def prep_grid(ks=(0, 1, 3), ds=(3, 5, 8), metrics=("cosine",), variants=("last",)) -> list[Prep]:
    return [Prep(k, True, d, m, v) for k, d, m, v in itertools.product(ks, ds, metrics, variants)]


def preprocess(X: np.ndarray, bg_mean: np.ndarray, bg_pcs: np.ndarray, p: Prep) -> np.ndarray:
    """X: [n, D] raw centroids (or instances) at one layer. Returns [n, d_proj]."""
    Y = X.astype(np.float64) - bg_mean
    if p.k_remove > 0:
        P = bg_pcs[: p.k_remove].astype(np.float64)        # [k, D], orthonormal rows
        Y = Y - (Y @ P.T) @ P
    if p.normalize:
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return project_pca(Y, p.d_proj)


def project_pca(Y: np.ndarray, d: int) -> np.ndarray:
    Yc = Y - Y.mean(0)
    d = min(d, Yc.shape[0] - 1, Yc.shape[1])
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    return Yc @ Vt[:d].T


def pca_spectrum(Y: np.ndarray) -> np.ndarray:
    Yc = Y - Y.mean(0)
    S = np.linalg.svd(Yc, compute_uv=False)
    return S**2 / max(len(Y) - 1, 1)


def distance_matrix(Z: np.ndarray, metric: str = "cosine") -> np.ndarray:
    D = squareform(pdist(Z, metric=metric))
    return np.nan_to_num(D, nan=0.0)


def participation_ratio(Y: np.ndarray) -> float:
    """Intrinsic-dimension covariate: (sum lambda)^2 / sum lambda^2."""
    lam = pca_spectrum(Y)
    return float(lam.sum() ** 2 / max((lam**2).sum(), 1e-30))


def procrustes_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Orthogonal Procrustes disparity between two [n, d] configurations (scale-normalized)."""
    from scipy.spatial import procrustes

    k = min(A.shape[1], B.shape[1])
    _, _, disp = procrustes(A[:, :k], B[:, :k])
    return float(disp)


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    A = A - A.mean(0)
    B = B - B.mean(0)
    hsic = np.linalg.norm(A.T @ B) ** 2
    return float(hsic / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))

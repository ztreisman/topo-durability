"""Null models.

1. Random item sets matched on token frequency and token count (sample_matched_items).
   Their contexts must come from the same source as the family's (Pile-mined), since family
   templates do not fit arbitrary words. Wired up in Phase 1/2.
2. Gaussian clouds with the family's PCA spectrum (gaussian_null), run through the same
   distance + persistence code.
3. Label permutation for order_score: tda.order_test.

Phase 5 rewiring nulls (degree-preserving, triangle-preserving) go in graph/.
"""

from __future__ import annotations

import numpy as np

from . import tda
from .geometry import distance_matrix, pca_spectrum


def gaussian_null(Z: np.ndarray, metric: str = "cosine", maxdim: int = 1, n: int = 200,
                  seed: int = 0) -> dict[str, np.ndarray]:
    """Z: preprocessed family points [I, d]. Returns arrays of null scores (h1_max, h1_ratio, ...)."""
    rng = np.random.default_rng(seed)
    I, d = Z.shape
    sd = np.sqrt(pca_spectrum(Z)[:d])
    out: dict[str, list] = {}
    for _ in range(n):
        G = rng.standard_normal((I, len(sd))) * sd
        s = tda.scores(tda.diagrams(distance_matrix(G, metric), maxdim=maxdim)["dgms"])
        for k, v in s.items():
            out.setdefault(k, []).append(v)
    return {k: np.asarray(v, float) for k, v in out.items()}


def empirical_p(null: np.ndarray, obs: float) -> float:
    null = null[np.isfinite(null)]
    return float((1 + (null >= obs).sum()) / (1 + len(null)))


def sample_matched_items(
    candidates: list[str],
    cand_logfreq: np.ndarray,
    cand_ntok: np.ndarray,
    target_logfreq: np.ndarray,
    target_ntok: np.ndarray,
    tol: float = 0.25,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> list[str]:
    """One random set, item-by-item matched: same token count, |log10 freq diff| <= tol,
    sampled without replacement. Raises if no match is available."""
    rng = np.random.default_rng(seed)
    used = set(exclude or ())
    out = []
    for lf, nt in zip(target_logfreq, target_ntok):
        ok = [i for i in np.flatnonzero((cand_ntok == nt) & (np.abs(cand_logfreq - lf) <= tol))
              if candidates[i] not in used]
        if not ok:
            raise ValueError(f"no candidate with ntok={nt}, logfreq~{lf:.2f}")
        c = candidates[int(rng.choice(ok))]
        used.add(c)
        out.append(c)
    return out

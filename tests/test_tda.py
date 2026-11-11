"""Synthetic checks for the persistence machinery (Phase 0 step 3)."""

import numpy as np
import pytest

from topodur import nulls, tda
from topodur.geometry import Prep, distance_matrix, preprocess
from topodur.scoring import cyclic_order_exact


def noisy_circle(n=60, noise=0.05, seed=0, dim=2):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    X = np.zeros((n, dim))
    X[:, 0], X[:, 1] = np.cos(t), np.sin(t)
    return X + noise * rng.standard_normal(X.shape), t


def clifford_torus(m=16, noise=0.0, seed=0):
    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, m, endpoint=False), np.linspace(0, 2 * np.pi, m, endpoint=False))
    a, b = a.ravel(), b.ravel()
    X = np.c_[np.cos(a), np.sin(a), np.cos(b), np.sin(b)]
    return X + noise * np.random.default_rng(seed).standard_normal(X.shape)


def n_long(dgm, thresh):
    return int((tda.lifetimes(dgm) > thresh).sum())


def test_circle_one_h1():
    X, _ = noisy_circle()
    dg = tda.diagrams(distance_matrix(X, "euclidean"))["dgms"]
    assert n_long(dg[1], 0.5) == 1
    s = tda.scores(dg)
    assert s["h1_max"] > 1.0 and s["h1_ratio"] > 2.5


def test_torus_two_h1_one_h2():
    X = clifford_torus(m=14)
    dg = tda.diagrams(distance_matrix(X, "euclidean"), maxdim=2)["dgms"]
    assert n_long(dg[1], 0.3) == 2
    assert n_long(dg[2], 0.1) == 1


def test_blob_no_long_h1():
    X = np.random.default_rng(1).standard_normal((60, 3))
    dg = tda.diagrams(distance_matrix(X, "euclidean"))["dgms"]
    assert tda.scores(dg)["h1_max"] < 0.6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_circular_coordinates_recover_order(seed):
    X, t = noisy_circle(n=40, noise=0.08, seed=seed, dim=5)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, t = X[perm], t[perm]
    cc = tda.circular_coordinates(distance_matrix(X, "euclidean"))
    assert cc is not None and cc.connected and cc.cocycle_ok
    score, p = tda.order_test(cc.theta, t, n_perm=500)
    assert score > 0.9 and p < 0.01
    assert cyclic_order_exact(cc.theta, t) or score > 0.95  # tiny noise swaps allowed


def test_circular_coordinates_seven_points():
    """Days-of-week scale: 7 points on a circle in 5D with mild noise."""
    rng = np.random.default_rng(3)
    t = 2 * np.pi * np.arange(7) / 7
    X = np.zeros((7, 5))
    X[:, 0], X[:, 1] = np.cos(t), np.sin(t)
    X += 0.05 * rng.standard_normal(X.shape)
    cc = tda.circular_coordinates(distance_matrix(X, "cosine"))
    assert cc is not None
    assert cyclic_order_exact(cc.theta, t)


def test_order_score_rotation_reflection_invariant():
    t = 2 * np.pi * np.arange(12) / 12
    assert tda.circular_rank_corr(t, t) == pytest.approx(1)
    assert tda.circular_rank_corr((t + 1.0) % (2 * np.pi), t) == pytest.approx(1)
    assert tda.circular_rank_corr((-t) % (2 * np.pi), t) == pytest.approx(1)
    rnd = np.random.default_rng(0).permutation(t)
    assert tda.circular_rank_corr(rnd, t) < 0.6


def test_cyclic_order_exact():
    t = np.arange(6.0)
    assert cyclic_order_exact(np.array([3, 4, 5, 0, 1, 2.0]), t)
    assert cyclic_order_exact(np.array([5, 4, 3, 2, 1, 0.0]), t)
    assert not cyclic_order_exact(np.array([0, 2, 1, 3, 4, 5.0]), t)


def test_representative_cycle_goes_around():
    X, t = noisy_circle(n=30, noise=0.02, seed=4)
    D = distance_matrix(X, "euclidean")
    cyc = tda.representative_cycle(D, tda.circular_coordinates(D))
    assert len(cyc) >= 10
    ang = np.unwrap(np.arctan2(X[cyc + cyc[:1], 1], X[cyc + cyc[:1], 0]))
    assert abs(abs(ang[-1] - ang[0]) - 2 * np.pi) < 0.5  # winding number 1


def test_bootstrap_band_flags_real_loop():
    """Item-level noisy instances around 12 circle centroids in 50D: the loop must be significant."""
    rng = np.random.default_rng(0)
    I, n, Dm = 12, 100, 50
    t = 2 * np.pi * np.arange(I) / I
    C = np.zeros((I, Dm))
    C[:, 0], C[:, 1] = 3 * np.cos(t), 3 * np.sin(t)
    C += 5.0  # offset, removed by centering
    inst = C[:, None, :] + rng.standard_normal((I, n, Dm))
    prep = Prep(k_remove=0, d_proj=5, metric="cosine")
    bg_mean, bg_pcs = np.full(Dm, 5.0), np.eye(Dm)[:3]

    def to_D(Cs):
        return distance_matrix(preprocess(Cs, bg_mean, bg_pcs, prep), "cosine")

    cents = inst.mean(1)
    dg = tda.diagrams(to_D(cents))["dgms"]
    band = tda.bootstrap_band(inst, np.full(I, n), to_D, dg, n_boot=40)
    assert tda.n_significant(dg[1], band.threshold(1)) == 1


def test_gaussian_null_runs():
    X, _ = noisy_circle(n=12, noise=0.01)
    null = nulls.gaussian_null(X, "euclidean", n=20)
    assert len(null["h1_ratio"]) == 20


def test_matched_sampler():
    cands = [f"w{i}" for i in range(100)]
    lf = np.linspace(2, 6, 100)
    nt = np.ones(100, int)
    out = nulls.sample_matched_items(cands, lf, nt, np.array([3.0, 4.0, 5.0]), np.array([1, 1, 1]), seed=1)
    assert len(set(out)) == 3


def test_closure_ratio_circle_vs_line():
    from topodur.scoring import closure_ratio

    t = 2 * np.pi * np.arange(10) / 10
    circ = np.c_[np.cos(t), np.sin(t)]
    line = np.c_[np.arange(10.0), np.zeros(10)]
    assert closure_ratio(distance_matrix(circ, "euclidean"), t) == pytest.approx(1, abs=0.01)
    assert closure_ratio(distance_matrix(line, "euclidean"), np.arange(10.0)) == pytest.approx(9)

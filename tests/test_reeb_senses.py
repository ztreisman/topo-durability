import numpy as np

from topodur.reeb import context_links, gap_k, pcs


def test_gap_one_blob():
    rng = np.random.default_rng(0)
    ks = [gap_k(rng.normal(size=(256, 10)) * np.linspace(3, 0.5, 10), seed=s)[0] for s in range(10)]
    assert sum(k > 1 for k in ks) <= 1                     # ~5% false splits on a Gaussian blob
    # skewed neurons (raw lognormal in 10 dims does split ~40%; through PCA of many neurons it does not)
    ks = [gap_k(pcs(rng.lognormal(size=(256, 200))), seed=s)[0] for s in range(5)]
    assert ks == [1] * 5


def test_gap_planted_skewed():
    rng = np.random.default_rng(3)
    X = rng.lognormal(size=(256, 200))
    X[:90, :20] += 3
    assert gap_k(pcs(np.arcsinh(X)))[0] == 2


def test_gap_two_senses():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(256, 10))
    Y[:100, 0] += 6
    Y -= Y.mean(0)
    k, g = gap_k(Y)
    assert k == 2
    assert abs(min(len(x) for x in g) - 100) <= 3


def test_links():
    L = context_links([np.arange(10), np.arange(10, 20)], [np.arange(15), np.arange(15, 20)])
    assert L[(0, 0)] == 1.0 and L[(1, 0)] == 0.5 and L[(1, 1)] == 1.0 and (0, 1) not in L

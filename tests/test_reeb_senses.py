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


def test_barcode():
    from topodur.reeb import sense_barcode

    n = 100
    lab = np.zeros((5, n), int)
    lab[1:4, :40] = 1                       # a 40-context sense from layer 1 to 3; the rest = 2nd cluster
    lab[4] = 0                              # merged back at layer 4 (whole token: trivial, no bar)
    bars = sense_barcode(lab)
    assert (1, 3) in bars and len([b for b in bars if b[1] == 3]) == 2       # both halves persist 3 layers
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 3, size=(6, n))                                  # random re-partition every layer
    assert max(b[1] for b in sense_barcode(noisy)) == 1

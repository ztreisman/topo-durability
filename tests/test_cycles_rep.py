import numpy as np
import scipy.sparse as sp

from topodur.graph.cycles import _representative_slow, h1_pairs, representative


def _graph(n, p, seed):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=p, random_state=seed, data_rvs=lambda k: rng.uniform(0.01, 1, k)).tocsr()
    A = sp.triu(A, 1)
    return (A + A.T).tocsr()


def test_bidirectional_matches_reference():
    for seed in range(6):
        D = _graph(300, 0.02, seed)
        for b, _, e in h1_pairs(D)[:150]:
            fast, slow = representative(D, b, e), _representative_slow(D, b, e)
            assert len(fast) == len(slow)
            assert fast[0] == e[0] and fast[-1] == e[1]
            lim = b + 1e-12
            for x, y in zip(fast[:-1], fast[1:]):
                assert 0 < D[x, y] <= lim
                assert {x, y} != set(e)
            assert len(set(fast)) == len(fast)


def test_layered_rewire_preserves_block_degrees():
    from topodur.graph.cycles import edges_of, rewire

    rng = np.random.default_rng(0)
    n = 400
    block = rng.integers(0, 5, n)
    D = _graph(n, 0.03, 1)
    N = rewire(D, node_block=block, seed=3)

    def prof(M):
        r, c, w = edges_of(M)
        out = {}
        for a, b, x in zip(r, c, w):
            for p, q in ((a, b), (b, a)):
                out[(p, block[q])] = out.get((p, block[q]), 0) + 1
        pair = {}
        for a, b, x in zip(r, c, w):
            k = tuple(sorted((block[a], block[b])))
            pair.setdefault(k, []).append(round(float(x), 6))
        return out, {k: sorted(v) for k, v in pair.items()}

    d0, w0 = prof(D)
    d1, w1 = prof(N)
    assert d0 == d1 and w0 == w1
    assert (abs(N - D) > 0).nnz > 0

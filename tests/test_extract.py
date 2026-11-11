"""Extraction on pythia-70m on CPU (downloads ~160 MB on first run)."""

import numpy as np
import pytest

from topodur.concepts import load_family, template_contexts
from topodur.extract import FamilyActs, background_stats, extract_family, tokenize_contexts

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def lm():
    from topodur.models import load_model

    return load_model("70m", 143000, dtype="fp32", device="cpu")


def test_numerics_ok(lm):
    from topodur.models import numerics_check

    assert numerics_check(lm) == []


def test_tokenize_truncates_after_item(lm):
    fam = load_family("months")
    ctxs = template_contexts(fam, 5)
    for (ids, span), c in zip(tokenize_contexts(lm.tokenizer, ctxs), ctxs):
        assert span[-1] == len(ids) - 1
        assert lm.tokenizer.decode([ids[i] for i in span]).strip() == fam.items[c.item]


def test_extract_shapes_and_centroids(lm, tmp_path):
    fam = load_family("days")
    ctxs = template_contexts(fam, 12, seed=0)
    acts = extract_family(lm, ctxs, fam.items, batch_size=16, store_instances=12)
    L, I, d = lm.n_layers, fam.n_items, lm.d_model
    assert acts.centroids["last"].shape == (L, I, d)
    assert acts.instances["mean"].shape == (L, I, 12, d)
    assert (acts.counts == 12).all()
    # centroid = mean of instances when every context is stored (fp16 tolerance)
    np.testing.assert_allclose(acts.instances["last"].astype(np.float32).mean(2), acts.centroids["last"],
                               rtol=0, atol=2e-2 * np.abs(acts.centroids["last"]).max())
    acts.save(tmp_path / "a.npz")
    b = FamilyActs.load(tmp_path / "a.npz")
    np.testing.assert_array_equal(b.centroids["last"], acts.centroids["last"])
    assert b.items == fam.items


def test_padding_does_not_change_vectors(lm):
    """Same context alone vs. batched with longer ones must give the same activations."""
    fam = load_family("months")
    ctxs = template_contexts(fam, 6, seed=1)
    one = extract_family(lm, ctxs[:1], fam.items[:1], batch_size=1, store_instances=1)
    many = extract_family(lm, [c for c in ctxs if c.item == 0], fam.items[:1], batch_size=6, store_instances=6)
    j = [c for c in ctxs if c.item == 0].index(ctxs[0])
    np.testing.assert_allclose(many.instances["last"][:, 0, j].astype(np.float32),
                               one.instances["last"][:, 0, 0].astype(np.float32), atol=5e-2)


def test_background_stats(lm):
    texts = ["The committee met on a Tuesday to discuss the budget for the coming year and more. " * 4] * 20
    bg = background_stats(lm, texts, n_texts=20, max_len=64, positions_per_text=8, k=5, batch_size=8)
    assert bg.mean.shape == (lm.n_layers, lm.d_model)
    assert bg.pcs.shape == (lm.n_layers, 5, lm.d_model)
    np.testing.assert_allclose(np.linalg.norm(bg.pcs, axis=2), 1, atol=1e-4)

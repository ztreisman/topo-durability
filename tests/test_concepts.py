import pytest

from topodur.concepts import list_families, load_family, template_contexts


@pytest.mark.parametrize("name", list_families())
def test_family_loads_and_has_contexts(name):
    fam = load_family(name)
    ctxs = template_contexts(fam, 200, seed=0)
    assert len(ctxs) == 200 * fam.n_items
    for c in ctxs[:50]:
        assert c.text[c.span[0]:c.span[1]] == fam.items[c.item]
        assert c.span[0] > 0


def test_same_prefixes_for_every_item():
    fam = load_family("months")
    ctxs = template_contexts(fam, 50, seed=0)
    pre = {}
    for c in ctxs:
        pre.setdefault(c.item, []).append(c.text[: c.span[0]])
    assert all(pre[i] == pre[0] for i in pre)
    assert len(set(pre[0])) == 50  # all distinct


def test_contexts_deterministic():
    fam = load_family("days")
    a = template_contexts(fam, 30, seed=7)
    b = template_contexts(fam, 30, seed=7)
    assert a == b

"""Mine natural contexts for concept-family items from Pile text (Phase 1, second context source).

Matching is case-sensitive and whole-word (an item never matches inside a longer word or hyphenated
compound), with per-family options from the family YAML's `mine:` block:

  pattern:    regex with {item} where the item goes, e.g. "{item} o'clock". Text matched outside the
              item is only a filter; the stored context ends at the item, so the model never sees
              anything after it.
  require:    keywords (case-insensitive substrings), at least one within `window` chars of the item
  capitalize: also match the item with its first letter upper-cased (sentence-initial uses)
  variants:   {substring: [alternatives]} spelling variants (e.g. "-sharp" -> "#")

Each match yields one context: up to `left_chars` characters before the item, cut at a whitespace
boundary, plus the item. At most `per_doc` contexts per item per document, so no single document
dominates an item. Total (uncapped) occurrence counts per item are recorded for the frequency covariate.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .concepts import Context, Family, load_family

log = logging.getLogger(__name__)


@dataclass
class MineOpts:
    max_per_item: int = 500
    per_doc: int = 2
    left_chars: int = 400
    window: int = 300


def item_spellings(fam: Family) -> list[list[str]]:
    """Literal spellings to match for each item."""
    m = fam.mine
    out = []
    for it in fam.items:
        forms = {it}
        for sub, alts in (m.get("variants") or {}).items():
            if sub in it:
                forms |= {it.replace(sub, a) for a in alts}
        if m.get("capitalize"):
            forms |= {f[:1].upper() + f[1:] for f in forms}
        out.append(sorted(forms, key=len, reverse=True))
    return out


def family_regex(fam: Family) -> tuple[re.Pattern, dict[str, int]]:
    spell = item_spellings(fam)
    lookup = {s: i for i, forms in enumerate(spell) for s in forms}
    alt = "|".join(re.escape(s) for s in sorted(lookup, key=len, reverse=True))
    core = r"(?<![\w-])(?P<item>" + alt + r")(?![\w-])"
    pat = fam.mine.get("pattern")
    if pat:
        pre, post = pat.split("{item}")
        core = pre + core + post
    return re.compile(core), lookup


def _left_context(text: str, start: int, left_chars: int) -> int:
    a = max(0, start - left_chars)
    if a > 0:
        ws = text.find(" ", a, start)
        a = ws + 1 if ws != -1 else a
    return a


def mine_texts(fams: list[Family], texts, opts: MineOpts, doc_offset: int = 0):
    """Returns ({family: {item: [ctx dict]}}, {family: Counter(item -> total count)}, n_docs, n_chars)."""
    compiled = [(f, *family_regex(f), [k.lower() for k in (f.mine.get("require") or [])]) for f in fams]
    found = {f.name: defaultdict(list) for f in fams}
    totals = {f.name: Counter() for f in fams}
    n_docs = n_chars = 0
    for d, text in enumerate(texts):
        n_docs += 1
        n_chars += len(text)
        low = None
        for fam, rx, lookup, req in compiled:
            per_doc: Counter = Counter()
            for m in rx.finditer(text):
                s, e = m.span("item")
                i = lookup[m.group("item")]
                if req:
                    if low is None:
                        low = text.lower()
                    w = low[max(0, s - opts.window): e + opts.window]
                    if not any(k in w for k in req):
                        continue
                totals[fam.name][i] += 1
                if per_doc[i] >= opts.per_doc or len(found[fam.name][i]) >= opts.max_per_item:
                    continue
                per_doc[i] += 1
                a = _left_context(text, s, opts.left_chars)
                found[fam.name][i].append({"item": i, "text": text[a:e], "span": [s - a, e - a],
                                           "doc": doc_offset + d})
    return found, totals, n_docs, n_chars


def _read_chunk(path: str, start: int, end: int):
    """Lines that START in [start, end]; each line belongs to exactly one chunk."""
    with open(path, "rb") as f:
        if start:
            f.seek(start - 1)
            f.readline()  # move to the first line starting at or after `start`
        while f.tell() <= end:
            line = f.readline()
            if not line:
                break
            yield json.loads(line)["text"]


def _worker(args):
    path, start, end, names, opts, k = args
    fams = [load_family(n) for n in names]
    return mine_texts(fams, _read_chunk(path, start, end), opts, doc_offset=k * 10**7)


def mine_file(path: str | Path, names: list[str], opts: MineOpts, workers: int | None = None, limit_docs=None):
    """Mine a JSONL file in parallel byte-range chunks; merges and caps per item (chunk order)."""
    path = str(path)
    if limit_docs:
        with open(path, encoding="utf-8") as f:
            texts = [json.loads(l)["text"] for l in itertools.islice(f, limit_docs)]
        return mine_texts([load_family(n) for n in names], texts, opts)
    size = os.path.getsize(path)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    n_chunks = workers * 4
    bounds = [size * i // n_chunks for i in range(n_chunks + 1)]
    jobs = [(path, bounds[i], bounds[i + 1] - 1, names, opts, i) for i in range(n_chunks)]
    found = {n: defaultdict(list) for n in names}
    totals = {n: Counter() for n in names}
    nd = nc = 0
    with ProcessPoolExecutor(workers) as ex:
        for k, (f, t, d, c) in enumerate(ex.map(_worker, jobs)):
            nd += d
            nc += c
            for n in names:
                totals[n].update(t[n])
                for i, lst in f[n].items():
                    room = opts.max_per_item - len(found[n][i])
                    found[n][i].extend(lst[:max(room, 0)])
            if k % 32 == 0:
                log.info("mined chunk %d/%d (%d docs)", k + 1, n_chunks, nd)
    return found, totals, nd, nc


def save_mined(out_dir: Path, fam: Family, found: dict, totals: Counter, n_docs: int, n_chars: int,
               source: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{fam.name}.jsonl", "w", encoding="utf-8") as f:
        for i in range(fam.n_items):
            for c in found.get(i, []):
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    summary = {"family": fam.name, "source": source, "n_docs": n_docs, "n_chars": n_chars,
               "items": fam.items,
               "n_contexts": [len(found.get(i, [])) for i in range(fam.n_items)],
               "total_count": [int(totals.get(i, 0)) for i in range(fam.n_items)]}
    (out_dir / f"{fam.name}.summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def load_mined(ctx_dir: Path, fam: Family, min_per_item: int, max_per_item: int | None = None
               ) -> tuple[Family, list[Context], list[int]]:
    """Mined contexts for the items that have at least min_per_item of them.
    Returns (family restricted to those items, contexts with re-indexed items, kept original indices)."""
    from .concepts import subset

    by: dict[int, list[dict]] = defaultdict(list)
    with open(ctx_dir / f"{fam.name}.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            by[c["item"]].append(c)
    keep = [i for i in range(fam.n_items) if len(by[i]) >= min_per_item]
    new = {old: k for k, old in enumerate(keep)}
    ctxs = [Context(new[i], c["text"], tuple(c["span"]), "pile", -1)
            for i in keep for c in by[i][:max_per_item]]
    return subset(fam, keep), ctxs, keep

"""Build the large concept panel for the durability test (2026-11-17). CPU, run locally (needs the Pile mirror
and WordNet):

  uv run --with nltk python scripts/build_concept_panel.py

A concept here is a dictionary word, encoded by n = 1-4 GPT-NeoX tokens (" " + word). Candidates: lowercase
alphabetic words (>= 3 letters) that are in WordNet, counted as " word" followed by a non-letter in the first
--chars characters of the source (about the stretch the footprint pass streams). Kept if counted >= --min-count,
so each should reach 224 contexts. Sampled per n (--per-n), evenly over log-frequency bins, excluding the
words of configs/token_panel.yaml (those were the exploratory panel; this one is the confirmation).
Covariates stored per concept: n tokens, count, WordNet synset count (polysemy) and majority part of speech.
Output: configs/concept_panel.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.models import load_tokenizer  # noqa: E402
from topodur.pile import resolve_source  # noqa: E402
from topodur.reeb import panel  # noqa: E402

WORD = re.compile(r" ([a-z]{3,})(?![A-Za-z])")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mirror:pile_mine_1000000.jsonl")
    ap.add_argument("--chars", type=float, default=2.2e9, help="text counted (~ 500M tokens)")
    ap.add_argument("--min-count", type=int, default=400)
    ap.add_argument("--per-n", type=int, default=250)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(runs.REPO_ROOT / "configs" / "concept_panel.json"))
    a = ap.parse_args()
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
    cnt, seen = Counter(), 0
    with open(resolve_source(a.source), encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"]
            cnt.update(WORD.findall(t))
            seen += len(t)
            if seen >= a.chars:
                break
    print(f"counted {seen / 1e9:.2f}G chars, {len(cnt)} distinct words")
    tok = load_tokenizer("1.4b")
    old = {w.lower() for ws in panel(str(runs.REPO_ROOT / "configs" / "token_panel.yaml")).values() for w in ws}
    cand = {}
    for w, c in cnt.items():
        if c < a.min_count or w in old:
            continue
        ss = wn.synsets(w)
        if not ss:
            continue
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        if not 1 <= len(ids) <= 4:
            continue
        pos = Counter(s.pos().replace("s", "a") for s in ss).most_common(1)[0][0]
        cand[w] = {"word": w, "ids": ids, "n_tok": len(ids), "count": c, "synsets": len(ss), "pos": pos}
    rng = np.random.default_rng(a.seed)
    out = []
    for n in (1, 2, 3, 4):
        ws = sorted((w for w in cand if cand[w]["n_tok"] == n), key=lambda w: cand[w]["count"])
        if not ws:
            continue
        lf = np.log([cand[w]["count"] for w in ws])
        edges = np.quantile(lf, np.linspace(0, 1, a.bins + 1))
        b = np.clip(np.searchsorted(edges, lf, side="right") - 1, 0, a.bins - 1)
        take = []
        for k in range(a.bins):                      # even over bins; spill leftover quota to the remaining pool
            idx = np.flatnonzero(b == k)
            take += rng.choice(idx, min(len(idx), a.per_n // a.bins), replace=False).tolist()
        rest = [i for i in range(len(ws)) if i not in set(take)]
        take += rng.choice(rest, min(len(rest), a.per_n - len(take)), replace=False).tolist() if rest else []
        out += [{**cand[ws[i]], "freq_bin": int(b[i])} for i in sorted(take)]
        print(f"n = {n}: {len(ws)} candidates, {len(take)} taken")
    meta = {"source": a.source, "chars_counted": seen, "min_count": a.min_count, "seed": a.seed, "concepts": out}
    Path(a.out).write_text(json.dumps(meta, indent=0))
    print(f"{len(out)} concepts -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

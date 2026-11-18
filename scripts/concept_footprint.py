"""Concept footprints for the large panel (configs/concept_panel.json): MLP activations of every layer at the LAST
token of each occurrence of a concept (1-4 tokens), plus the contexts themselves. One GPU.

  uv run python scripts/concept_footprint.py --size 1.4b
  ... --limit 0.3          # 0.3M tokens, 16 contexts per concept, data/neurons_dryrun/

An occurrence = the concept's token ids (" " + word) at positions p-n+1..p (p - n + 1 >= 4) followed by a token
that does not continue the word (no letter or digit right after), so " pier" does not match " pierced".
Only sequences holding a new occurrence are run through the model, truncated after their last one (causal).
Output: data/neurons[_dryrun]/<unit>/cfoot/T<i>.npy [n_ctx, n_layers, N] fp16 (i = index in the panel),
contexts.npz (T<i> sequences [n, 256] int32, P<i> end positions, as token_contexts.py writes), footprint.json
(same layout as token_footprint.py; class = "n<k>" for a k-token concept); reports .../concept_contexts.md.
Size at 1.4B: 224 x 24 x 8192 x 2 B = 88 MB per concept (~85 GB for 969; not published).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.finetune import token_stream  # noqa: E402
from topodur.models import load_model  # noqa: E402

log = logging.getLogger("concept_footprint")


def ngram_keys(S: np.ndarray, n: int) -> np.ndarray:
    """uint64 key of the n tokens ending at each position (positions < n-1 get garbage; mask them)."""
    S = S.astype(np.uint64)
    K = np.zeros_like(S)
    for k in range(n):
        K[:, n - 1:] = (K[:, n - 1:] << np.uint64(16)) | S[:, k:S.shape[1] - n + 1 + k]
    return K


def key_of(ids: list[int]) -> np.uint64:
    k = np.uint64(0)
    for t in ids:
        k = (k << np.uint64(16)) | np.uint64(t)
    return k


def find(S: np.ndarray, keys: dict, cont: np.ndarray, open_: np.ndarray) -> list[tuple[int, int, int]]:
    """Occurrences (row, end position, concept) of open concepts in the batch S [B, T], row-major order."""
    hits = []
    T = S.shape[1]
    for n, (karr, cidx) in keys.items():
        K = ngram_keys(S, n)
        m = np.isin(K, karr)
        m[:, :n - 1 + 4] = False                       # >= 4 tokens of context before the concept
        m[:, T - 1] = False                            # need the next token for the word-boundary check
        r, p = np.nonzero(m)
        if not len(r):
            continue
        ok = ~cont[S[r, p + 1]]
        r, p = r[ok], p[ok]
        c = cidx[np.searchsorted(karr, K[r, p])]
        keep = open_[c]
        hits += list(zip(r[keep].tolist(), p[keep].tolist(), c[keep].tolist()))
    hits.sort()
    return hits


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--panel", default=str(runs.REPO_ROOT / "configs" / "concept_panel.json"))
    ap.add_argument("--source", default="mirror:pile_mine_1000000.jsonl")
    ap.add_argument("--skip-seqs", type=int, default=100_000)
    ap.add_argument("--n-ctx", type=int, default=224)
    ap.add_argument("--max-tokens", type=float, default=600e6)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--batch-seqs", type=int, default=64)
    ap.add_argument("--fwd-batch", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None, help="test pass: tokens in millions")
    a = ap.parse_args()
    unit = f"{a.size}_s0_step{a.step}"
    tag = "neurons_dryrun" if a.limit else "neurons"
    out = runs.DATA_DIR / tag / unit / "cfoot"
    out.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(out / "footprint.log")
    if runs.is_done(out / "footprint.json"):
        log.info("already done")
        return 0
    if a.limit:
        a.max_tokens, a.n_ctx, a.skip_seqs = a.limit * 1e6, 16, 0
    concepts = json.loads(Path(a.panel).read_text())["concepts"]
    lm = load_model(a.size, a.step, dtype=a.dtype)
    tok = lm.tokenizer
    for c in concepts:                                 # the panel was built with the 1.4b tokenizer (shared)
        assert tok(" " + c["word"], add_special_tokens=False)["input_ids"] == c["ids"], c["word"]
    V = len(tok)
    cont = np.zeros(max(V, lm.model.config.vocab_size), bool)
    for t in range(V):                                 # decoded text, not the BPE form: 'Ġ' (space) counts as a letter
        s = tok.decode([t])
        cont[t] = bool(s) and s[0].isalnum()
    keys = {}
    for n in (1, 2, 3, 4):
        cs = [(key_of(c["ids"]), i) for i, c in enumerate(concepts) if c["n_tok"] == n]
        if cs:
            cs.sort()
            keys[n] = (np.array([k for k, _ in cs], np.uint64), np.array([i for _, i in cs]))
    L, N = lm.model.config.num_hidden_layers, lm.model.config.intermediate_size
    net = lm.model.gpt_neox
    caps = {}
    hooks = [net.layers[l].mlp.act.register_forward_hook(lambda m, i, o, l=l: caps.__setitem__(l, o)) for l in range(L)]
    C = len(concepts)
    have = np.zeros(C, int)
    open_ = np.ones(C, bool)
    A = [np.lib.format.open_memmap(out / f"T{i}.npy", mode="w+", dtype=np.float16, shape=(a.n_ctx, L, N))
         for i in range(C)]
    seqs = [np.zeros((a.n_ctx, 256), np.int32) for _ in range(C)]
    pos = [np.zeros(a.n_ctx, np.int16) for _ in range(C)]
    stream = token_stream(tok, a.source, 256, skip_seqs=a.skip_seqs)
    seen, step = 0, 0
    try:
        while seen < a.max_tokens and open_.any():
            S = np.array([next(stream) for _ in range(a.batch_seqs)], dtype=np.int64)
            seen += S.size
            step += 1
            acc = []
            for r, p, c in find(S, keys, cont, open_):
                if have[c] < a.n_ctx:
                    acc.append((r, p, c, have[c]))
                    have[c] += 1
                    if have[c] == a.n_ctx:
                        open_[c] = False
            if acc:
                rows = sorted({r for r, *_ in acc})
                for s in range(0, len(rows), a.fwd_batch):
                    rr = rows[s:s + a.fwd_batch]
                    sub = [x for x in acc if x[0] in set(rr)]
                    T = max(p for _, p, _, _ in sub) + 1
                    x = torch.from_numpy(S[rr, :T]).to(lm.device)
                    net(input_ids=x, use_cache=False)
                    ri = torch.tensor([rr.index(r) for r, *_ in sub], device=lm.device)
                    pi = torch.tensor([p for _, p, _, _ in sub], device=lm.device)
                    V_ = torch.stack([caps[l][ri, pi] for l in range(L)], 1).half().cpu().numpy()
                    for k, (r, p, c, j) in enumerate(sub):
                        A[c][j] = V_[k]
                        seqs[c][j] = S[r]
                        pos[c][j] = p
            if step % 200 == 0:
                log.info("%.1fM tokens: %d / %d concepts full, median contexts %d, min %d", seen / 1e6,
                         int((~open_).sum()), C, int(np.median(have)), int(have.min()))
    finally:
        for h in hooks:
            h.remove()
    for i in range(C):
        A[i].flush()
        if have[i] < a.n_ctx:                           # rewrite the partial ones at their true length
            part = np.array(A[i][:have[i]])
            del A[i]
            A.insert(i, None)
            np.save(out / f"T{i}.npy", part)
    np.savez_compressed(out / "contexts.npz", **{f"T{i}": seqs[i][:have[i]] for i in range(C) if have[i]},
                        **{f"P{i}": pos[i][:have[i]] for i in range(C) if have[i]})
    meta = {"layers": list(range(L)), "tokens_streamed": seen, "n_ctx": a.n_ctx, "panel": a.panel,
            "tokens": {str(i): {"class": f"n{c['n_tok']}", "word": c["word"], "n": int(have[i]), "n_tok": c["n_tok"],
                                "count": c["count"], "synsets": c["synsets"], "pos": c["pos"], "freq_bin": c["freq_bin"]}
                       for i, c in enumerate(concepts)}}
    runs.atomic_write_text(out / "footprint.json", json.dumps(meta, indent=1))
    rep = runs.REPO_ROOT / "reports" / tag / unit
    rep.mkdir(parents=True, exist_ok=True)
    ex = ["# Example contexts per panel concept (first 2; the concept is in [brackets])", ""]
    for i, c in enumerate(concepts):
        ex.append(f"## {c['word']} (n{c['n_tok']}, {have[i]} contexts)")
        for j in range(min(2, have[i])):
            s, p, n = seqs[i][j], int(pos[i][j]), c["n_tok"]
            ex.append(f"- ...{tok.decode(s[max(0, p - n - 23):p - n + 1])}[{tok.decode(s[p - n + 1:p + 1])}]"
                      f"{tok.decode(s[p + 1:p + 12])}...".replace("\n", " "))
        ex.append("")
    runs.atomic_write_text(rep / "concept_contexts.md", "\n".join(ex))
    runs.mark_done(out / "footprint.json", {"tokens": seen})
    log.info("done: %.1fM tokens streamed; contexts per concept min %d median %d; %d full", seen / 1e6,
             int(have.min()), int(np.median(have)), int((~open_).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

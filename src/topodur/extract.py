"""Residual-stream activations after each block, per-item centroids, and background stats.

Layer index l in 0..n_layers-1 means the output of block l (the residual stream after it).
Two token variants per context:
  last : the last token of the item span (primary)
  mean : mean over the item's tokens (variant, recorded for multi-token items)
Contexts are truncated right after the item (the model is causal, so later text is irrelevant)
and left-truncated to max_len tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .concepts import Context
from .models import LoadedModel
from .runs import atomic_path

log = logging.getLogger(__name__)
VARIANTS = ("last", "mean")


@dataclass
class FamilyActs:
    """Activations for one family under one (model, step)."""
    items: list[str]
    layers: list[int]
    centroids: dict[str, np.ndarray]       # variant -> [L, I, d] float32
    instances: dict[str, np.ndarray]       # variant -> [L, I, n_inst, d] float16
    counts: np.ndarray                     # [I] contexts used per item
    n_inst: np.ndarray                     # [I] instances stored per item (<= n_inst dim)
    item_n_tokens: np.ndarray              # [I] median token count of the item span
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        arrays = {"items": np.array(self.items), "layers": np.array(self.layers),
                  "counts": self.counts, "n_inst": self.n_inst, "item_n_tokens": self.item_n_tokens}
        for v in self.centroids:
            arrays[f"centroids_{v}"] = self.centroids[v]
            arrays[f"instances_{v}"] = self.instances[v]
        arrays["meta"] = np.array(repr(self.meta))
        with atomic_path(path) as tmp:
            with open(tmp, "wb") as f:
                np.savez(f, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "FamilyActs":
        z = np.load(path, allow_pickle=False)
        vs = [k.removeprefix("centroids_") for k in z.files if k.startswith("centroids_")]
        return cls(
            items=[str(s) for s in z["items"]], layers=[int(x) for x in z["layers"]],
            centroids={v: z[f"centroids_{v}"] for v in vs},
            instances={v: z[f"instances_{v}"] for v in vs},
            counts=z["counts"], n_inst=z["n_inst"], item_n_tokens=z["item_n_tokens"],
            meta={"raw": str(z["meta"])},
        )


def _block_out(out):
    return out[0] if isinstance(out, tuple) else out


class _Capture:
    """Forward hooks that keep each block's output for the current batch."""

    def __init__(self, lm: LoadedModel, layers: list[int]):
        self.layers = layers
        self.h: dict[int, torch.Tensor] = {}
        self.handles = [lm.blocks[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def fn(_m, _i, out):
            self.h[l] = _block_out(out)
        return fn

    def close(self):
        for h in self.handles:
            h.remove()


def tokenize_contexts(tok, contexts: list[Context], max_len: int = 128):
    """Return per-context (input_ids, span token indices), truncated at the item end."""
    enc = tok([c.text for c in contexts], return_offsets_mapping=True, add_special_tokens=False)
    out = []
    for c, ids, offs in zip(contexts, enc["input_ids"], enc["offset_mapping"]):
        s, e = c.span
        span = [i for i, (a, b) in enumerate(offs) if b > s and a < e]
        if not span:
            raise ValueError(f"item span not found in tokens: {c.text!r}")
        end = span[-1] + 1
        start = max(0, end - max_len)
        span = [i - start for i in span if i >= start]
        out.append((ids[start:end], span))
    return out


@torch.no_grad()
def extract_family(
    lm: LoadedModel,
    contexts: list[Context],
    items: list[str],
    layers: list[int] | None = None,
    batch_size: int = 32,
    max_len: int = 128,
    store_instances: int = 200,
    seed: int = 0,
    limit: int | None = None,
    max_batch_tokens: int = 2048,
) -> FamilyActs:
    layers = list(range(lm.n_layers)) if layers is None else layers
    if limit:
        # keep the first `limit` contexts of every item
        per = {}
        contexts = [c for c in contexts if per.setdefault(c.item, []).append(c) or len(per[c.item]) <= limit]
    I, L, d = len(items), len(layers), lm.d_model
    toks = tokenize_contexts(lm.tokenizer, contexts, max_len)

    # pre-choose which contexts are stored as instances (seeded, per item)
    rng = np.random.default_rng(seed)
    by_item: dict[int, list[int]] = {}
    for j, c in enumerate(contexts):
        by_item.setdefault(c.item, []).append(j)
    keep_slot: dict[int, int] = {}
    n_inst = np.zeros(I, dtype=np.int32)
    for i, js in by_item.items():
        chosen = rng.permutation(js)[:store_instances]
        for slot, j in enumerate(sorted(chosen)):
            keep_slot[j] = slot
        n_inst[i] = len(chosen)

    cap = int(n_inst.max()) if len(n_inst) else 0
    sums = {v: torch.zeros(L, I, d, dtype=torch.float64) for v in VARIANTS}
    inst = {v: np.zeros((L, I, cap, d), dtype=np.float16) for v in VARIANTS}
    counts = np.zeros(I, dtype=np.int64)
    ntok: dict[int, list[int]] = {}

    order = sorted(range(len(contexts)), key=lambda j: len(toks[j][0]))
    cap_hooks = _Capture(lm, layers)
    pad = lm.tokenizer.pad_token_id
    try:
        # batches of at most batch_size contexts AND max_batch_tokens padded tokens (order is by length,
        # so the longest context in a batch is its last one)
        batches, cur = [], []
        for j in order:
            if cur and (len(cur) >= batch_size or (len(cur) + 1) * len(toks[j][0]) > max_batch_tokens):
                batches.append(cur)
                cur = []
            cur.append(j)
        if cur:
            batches.append(cur)
        done = 0
        for bi, idx in enumerate(batches):
            T = max(len(toks[j][0]) for j in idx)
            ids = torch.full((len(idx), T), pad, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            for r, j in enumerate(idx):
                n = len(toks[j][0])
                ids[r, :n] = torch.tensor(toks[j][0])
                att[r, :n] = 1
            lm.model.gpt_neox(input_ids=ids.to(lm.device), attention_mask=att.to(lm.device))
            # span weights: last-token one-hot and span-mean weights, [B, T]
            w_last = torch.zeros((len(idx), T))
            w_mean = torch.zeros((len(idx), T))
            for r, j in enumerate(idx):
                span = toks[j][1]
                w_last[r, span[-1]] = 1.0
                w_mean[r, span] = 1.0 / len(span)
            W = torch.stack([w_last, w_mean]).to(lm.device)                 # [2, B, T]
            H = torch.stack([cap_hooks.h[l] for l in layers]).float()       # [L, B, T, d]
            if not torch.isfinite(H).all():
                bad = [l for l, h in zip(layers, H) if not torch.isfinite(h).all()]
                raise FloatingPointError(f"non-finite activations at layers {bad}")
            V = torch.einsum("vbt,lbtd->vlbd", W, H).double().cpu()           # [2, L, B, d]
            for r, j in enumerate(idx):
                it = contexts[j].item
                counts[it] += 1
                ntok.setdefault(it, []).append(len(toks[j][1]))
                for vi, v in enumerate(VARIANTS):
                    sums[v][:, it] += V[vi, :, r]
                    if j in keep_slot:
                        inst[v][:, it, keep_slot[j]] = V[vi, :, r].numpy().astype(np.float16)
            done += len(idx)
            if bi % 50 == 0:
                log.info("extract: %d/%d contexts", done, len(order))
    finally:
        cap_hooks.close()
    cents = {v: (sums[v] / torch.from_numpy(np.maximum(counts, 1))[None, :, None]).float().numpy()
             for v in VARIANTS}
    return FamilyActs(
        items=items, layers=layers, centroids=cents, instances=inst, counts=counts, n_inst=n_inst,
        item_n_tokens=np.array([int(np.median(ntok.get(i, [0]))) for i in range(I)]),
        meta={"model": lm.size, "step": lm.step, "dtype": lm.dtype, "max_len": max_len},
    )


# ---------------------------------------------------------------- background statistics

@dataclass
class Background:
    layers: list[int]
    mean: np.ndarray        # [L, d]
    pcs: np.ndarray         # [L, k, d] top-k principal directions (unit rows)
    evals: np.ndarray       # [L, k]
    n_vectors: int

    def save(self, path):
        with atomic_path(path) as tmp:
            with open(tmp, "wb") as f:
                np.savez(f, layers=np.array(self.layers), mean=self.mean, pcs=self.pcs,
                         evals=self.evals, n_vectors=self.n_vectors)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls([int(x) for x in z["layers"]], z["mean"], z["pcs"], z["evals"], int(z["n_vectors"]))


@torch.no_grad()
def background_stats(
    lm: LoadedModel,
    texts,
    layers: list[int] | None = None,
    n_texts: int = 2000,
    max_len: int = 256,
    positions_per_text: int = 32,
    skip_first: int = 8,
    k: int = 10,
    batch_size: int = 16,
    seed: int = 0,
) -> Background:
    """Mean and top-k PCs of the residual stream at random token positions.

    The first `skip_first` positions are excluded: Pythia's early positions carry
    very large-norm "attention sink" activations that would dominate the mean.
    """
    layers = list(range(lm.n_layers)) if layers is None else layers
    d = lm.d_model
    dev = lm.device
    s1 = torch.zeros(len(layers), d, dtype=torch.float64, device=dev)
    s2 = torch.zeros(len(layers), d, d, dtype=torch.float32, device=dev)  # fp32 XtX on GPU
    n = 0
    rng = np.random.default_rng(seed)
    cap = _Capture(lm, layers)
    batch: list[list[int]] = []

    def flush():
        nonlocal n
        T = max(map(len, batch))
        ids = torch.full((len(batch), T), lm.tokenizer.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for r, b in enumerate(batch):
            ids[r, :len(b)] = torch.tensor(b)
            att[r, :len(b)] = 1
        lm.model.gpt_neox(input_ids=ids.to(dev), attention_mask=att.to(dev))
        rows, cols = [], []
        for r, b in enumerate(batch):
            pos = rng.choice(np.arange(skip_first, len(b)), size=min(positions_per_text, len(b) - skip_first),
                             replace=False)
            rows += [r] * len(pos)
            cols += pos.tolist()
        rows_t, cols_t = torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)
        for li, l in enumerate(layers):
            x = cap.h[l][rows_t, cols_t].float()
            s1[li] += x.double().sum(0)
            s2[li] += x.T @ x
        n += len(rows)
        batch.clear()

    try:
        used = 0
        for text in texts:
            ids = lm.tokenizer(text, add_special_tokens=False)["input_ids"][:max_len]
            if len(ids) < skip_first + 8:
                continue
            batch.append(ids)
            used += 1
            if len(batch) == batch_size:
                flush()
            if used >= n_texts:
                break
        if batch:
            flush()
    finally:
        cap.close()
    if n == 0:
        raise ValueError("no background vectors collected")
    mean = (s1 / n).float()
    means, pcs, evals = [], [], []
    for li in range(len(layers)):
        cov = s2[li].double() / n - torch.outer(mean[li].double(), mean[li].double())
        w, V = (t.cpu().numpy() for t in torch.linalg.eigh(cov))  # on the GPU: CPU BLAS can be oversubscribed
        top = np.argsort(w)[::-1][:k]
        pcs.append(V[:, top].T.astype(np.float32))
        evals.append(w[top].astype(np.float32))
    log.info("background: %d vectors from %d texts", n, used)
    return Background(layers, mean.cpu().numpy(), np.stack(pcs), np.stack(evals), n)

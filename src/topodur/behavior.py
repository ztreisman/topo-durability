"""Behavioral offset queries with a wraparound split (Phase 3, measure 5).

A family's YAML may carry a `behavior:` block:
  prompt:  "{k} months after {item} is"       ({k} = number word, {item} = start item)
  answer:  " {item}"                          (continuation format of each candidate; default " {item}")
  offsets: [2, 3, 4, 5]                       (steps forward in the ground-truth order)
  fewshot: ["Two months after March is May.", ...]   (prepended, one per line)
  step_unit: optional multiplier shown in {k} instead of the step count (e.g. 45 for compass degrees)

Every (start item, offset) pair is a query; the model's answer is the candidate item with the highest
total log-probability of its continuation. For circle families a query is "wraparound" when the
forward path crosses the end of the list (December -> February). Interval families only get
non-wrapping queries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .concepts import Family

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten", 11: "eleven", 12: "twelve"}


@dataclass
class Query:
    start: int
    offset: int
    target: int
    wrap: bool
    prompt: str


def queries(fam: Family, spec: dict) -> list[Query]:
    n = fam.n_items
    order = list(np.argsort(fam.angles(), kind="stable"))  # item indices in ground-truth order
    rank = {it: r for r, it in enumerate(order)}
    head = "".join(s + "\n" for s in spec.get("fewshot", []))
    unit = spec.get("step_unit")
    out = []
    for it in range(n):
        for k in spec["offsets"]:
            r = rank[it] + k
            wrap = r >= n
            if wrap and not fam.is_cyclic:
                continue
            tgt = order[r % n]
            kk = k * unit if unit else k
            kw = f"{kk:g}" if unit else NUM_WORDS.get(kk, str(kk))
            p = spec["prompt"].format(k=kw, K=kw.capitalize(), item=fam.items[it])
            out.append(Query(it, k, tgt, wrap, head + p[:1].upper() + p[1:]))
    return out


@torch.no_grad()
def candidate_logprobs(lm, prompts: list[str], conts: list[str], batch_size: int = 32) -> np.ndarray:
    """[P, C] total log-probability of each continuation after each prompt."""
    tok = lm.tokenizer
    P = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    C = [tok(c, add_special_tokens=False)["input_ids"] for c in conts]
    pairs = [(i, j) for i in range(len(P)) for j in range(len(C))]
    out = np.zeros((len(P), len(C)))
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b:b + batch_size]
        seqs = [P[i] + C[j] for i, j in chunk]
        T = max(map(len, seqs))
        ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            att[r, :len(s)] = 1
        logits = lm.model(input_ids=ids.to(lm.device), attention_mask=att.to(lm.device)).logits.float()
        lp = torch.log_softmax(logits, -1).cpu()
        for r, (i, j) in enumerate(chunk):
            a = len(P[i])
            tgt = torch.tensor(C[j])
            out[i, j] = float(lp[r, a - 1:a - 1 + len(tgt)].gather(-1, tgt[:, None]).sum())
    return out


def evaluate(lm, fam: Family, spec: dict, batch_size: int = 32) -> dict:
    qs = queries(fam, spec)
    ans = spec.get("answer", " {item}")
    conts = [ans.format(item=x) for x in fam.items]
    lp = candidate_logprobs(lm, [q.prompt for q in qs], conts, batch_size)
    pred = lp.argmax(1)
    correct = np.array([p == q.target for p, q in zip(pred, qs)])
    wrap = np.array([q.wrap for q in qs])
    # margin: target log-prob minus the best other candidate (graded, less noisy than accuracy)
    tgt = np.array([q.target for q in qs])
    other = lp.copy()
    other[np.arange(len(qs)), tgt] = -np.inf
    margin = lp[np.arange(len(qs)), tgt] - other.max(1)
    res = {"family": fam.name, "n_queries": len(qs), "acc": float(correct.mean()),
           "margin": float(margin.mean()), "chance": 1 / fam.n_items}
    for name, m in (("wrap", wrap), ("nowrap", ~wrap)):
        res[f"acc_{name}"] = float(correct[m].mean()) if m.any() else np.nan
        res[f"margin_{name}"] = float(margin[m].mean()) if m.any() else np.nan
        res[f"n_{name}"] = int(m.sum())
    return res

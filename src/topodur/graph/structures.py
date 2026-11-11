"""Matched cycles, induced paths, and cliques from a latent graph, and their durability (Phase 5).

Sampling (all on the graph thresholded at the cycle's birth distance, so structures share a scale):
  cycles  chordless H1 representatives that beat both rewiring nulls
  paths   induced (chordless) paths of the same vertex count, grown by random extension, whose mean
          log firing rate is within `rate_tol` of the cycle's
  cliques maximal cliques of the thresholded graph, truncated to the cycle's size when larger;
          cliques are usually smaller than long cycles, so size matching is reported, not forced

Durability of a structure:
  act_corr   mean over its latents of the Pearson correlation between the latent's activations on
             the base model and on a modified model (fine-tuned / compressed), same tokens, same SAE
  ablation   zero one latent's decoder contribution in the residual stream at the SAE layer (on the
             tokens where it fires) and measure (a) the relative change of the OTHER latents' total
             activation in the structure and (b) the change in next-token loss; averaged over latents
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def thresholded(D: sp.csr_matrix, at: float) -> sp.csr_matrix:
    A = D.copy()
    A.data[A.data > at + 1e-12] = 0
    A.eliminate_zeros()
    return A


def induced_path(A: sp.csr_matrix, k: int, rng, tries: int = 200) -> list[int] | None:
    n = A.shape[0]
    deg = np.diff(A.indptr)
    starts = np.where(deg > 0)[0]
    if len(starts) == 0:
        return None
    for _ in range(tries):
        path = [int(rng.choice(starts))]
        inpath = set(path)
        while len(path) < k:
            end = path[-1]
            nbrs = [int(x) for x in A.indices[A.indptr[end]:A.indptr[end + 1]] if x not in inpath]
            rng.shuffle(nbrs)
            ok = None
            for c in nbrs:  # chordless: c may touch only the current end of the path
                cn = set(A.indices[A.indptr[c]:A.indptr[c + 1]].tolist())
                if not (cn & (inpath - {end})):
                    ok = c
                    break
            if ok is None:
                break
            path.append(ok)
            inpath.add(ok)
        if len(path) == k:
            return path
    return None


def matched_paths(A, k, target_rate, rates_nodes, n, rng, rate_tol=0.5, max_tries=2000):
    out = []
    for _ in range(max_tries):
        p = induced_path(A, k, rng, tries=5)
        if p is None:
            continue
        if abs(np.log10(rates_nodes[p]).mean() - np.log10(target_rate)) <= rate_tol:
            out.append(p)
            if len(out) >= n:
                break
    return out


def cliques(A: sp.csr_matrix, k: int, n: int, rng) -> list[list[int]]:
    import networkx as nx

    G = nx.from_scipy_sparse_array(A)
    cl = [c for c in nx.find_cliques(G) if len(c) >= 3]
    rng.shuffle(cl)
    return [sorted(c)[:k] for c in cl[:n]]


@torch.no_grad()
def latent_acts(lm, sae, layer: int, token_batches: list[torch.Tensor], latents: np.ndarray) -> np.ndarray:
    """[n_tokens, len(latents)] post-TopK activations of the given latents (0 where not in the top k)."""
    lat = torch.as_tensor(latents, device=sae.W_dec.device)
    cap = {}
    hk = lm.model.gpt_neox.layers[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    out = []
    try:
        for ids in token_batches:
            lm.model.gpt_neox(input_ids=ids.to(lm.device), use_cache=False)
            h = cap["h"].float().reshape(-1, cap["h"].shape[-1])
            v, i = sae.encode(h)
            dense = torch.zeros(len(h), sae.cfg.n_latents, device=h.device)
            dense.scatter_(1, i, v)
            out.append(dense[:, lat].cpu())
    finally:
        hk.remove()
    return torch.cat(out).numpy()


def act_corr(base: np.ndarray, mod: np.ndarray) -> np.ndarray:
    """Per-latent Pearson correlation between two [T, L] activation matrices."""
    b = base - base.mean(0)
    m = mod - mod.mean(0)
    den = np.sqrt((b ** 2).sum(0) * (m ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (b * m).sum(0) / den


@torch.no_grad()
def ablation_effects(lm_full, sae, layer: int, ids: torch.Tensor, struct: list[int]) -> dict:
    """Ablate each latent of a structure in turn: subtract its decoder contribution from the residual
    stream at `layer` (full model, so the next-token loss is measured), re-encode the ablated residual,
    and record (a) the relative change in the total activation of the structure's OTHER latents on the
    tokens where the ablated latent fired, and (b) the change in mean next-token loss."""
    st = torch.as_tensor(struct, device=sae.W_dec.device)
    state = {"ablate": None}
    rec = {}

    def dense_acts(x):
        v, idx = sae.encode(x)
        d = torch.zeros(v.shape[0], sae.cfg.n_latents, device=x.device)
        d.scatter_(1, idx, v)
        return d

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        hf = h.float().reshape(-1, h.shape[-1])
        dense = dense_acts(hf)
        rec["before"] = dense[:, st]
        if state["ablate"] is None:
            return None
        l = state["ablate"]
        hf2 = hf - dense[:, l:l + 1] * sae.W_dec[l][None, :] / sae.scale
        rec["after"] = dense_acts(hf2)[:, st]
        hf2 = hf2.reshape(h.shape).to(h.dtype)
        return (hf2,) + tuple(o[1:]) if isinstance(o, tuple) else hf2

    hk = lm_full.model.gpt_neox.layers[layer].register_forward_hook(hook)
    try:
        ids = ids.to(lm_full.device)

        def loss():
            logits = lm_full.model(input_ids=ids, use_cache=False).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
            return float(-lp.mean())

        state["ablate"] = None
        base_loss = loss()
        rel, dl = [], []
        for j, l in enumerate(struct):
            state["ablate"] = int(l)
            dl.append(loss() - base_loss)
            fired = rec["before"][:, j] > 0
            if fired.any():
                others = [k for k in range(len(struct)) if k != j]
                b = rec["before"][fired][:, others].sum()
                a = rec["after"][fired][:, others].sum()
                rel.append(float((a - b).abs() / b.clamp_min(1e-9)))
        state["ablate"] = None
    finally:
        hk.remove()
    return {"dloss_mean": float(np.mean(dl)), "others_rel_change": float(np.mean(rel)) if rel else np.nan,
            "n_fired": len(rel)}

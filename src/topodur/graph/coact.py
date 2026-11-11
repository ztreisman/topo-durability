"""Latent co-activation statistics and sparse latent graphs (Phase 5).

Counts, over `n_tokens` streamed tokens: per-latent firing (per token), and per-pair co-occurrence within
non-overlapping windows of `window` tokens (a latent is "in" a window if it fires on any token of it).
NPMI(i, j) = log(p_ij / (p_i p_j)) / -log(p_ij) over windows.

Nodes: latents with per-token firing rate in [min_rate, max_rate]. Edges: each node's top `k_nn`
neighbours with NPMI > min_npmi (union, symmetric); distance = 1 - NPMI. A second graph uses decoder
cosine similarity with the same sparsification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .sae import ActStream, TopKSAE

log = logging.getLogger(__name__)


@dataclass
class CoactStats:
    fire_tokens: np.ndarray      # [n_latents] tokens on which each latent fired
    fire_windows: np.ndarray     # [n_latents] windows containing each latent
    pair_windows: np.ndarray     # [n_nodes, n_nodes] window co-occurrence among nodes
    nodes: np.ndarray            # latent ids of the nodes
    n_tokens: int
    n_windows: int

    def save(self, path: Path):
        np.savez_compressed(path, fire_tokens=self.fire_tokens, fire_windows=self.fire_windows,
                            pair_windows=self.pair_windows, nodes=self.nodes, n_tokens=self.n_tokens,
                            n_windows=self.n_windows)

    @classmethod
    def load(cls, path: Path) -> "CoactStats":
        z = np.load(path)
        return cls(z["fire_tokens"], z["fire_windows"], z["pair_windows"], z["nodes"], int(z["n_tokens"]),
                   int(z["n_windows"]))


@torch.no_grad()
def firing_rates(sae: TopKSAE, stream: ActStream, n_tokens: int) -> np.ndarray:
    counts = torch.zeros(sae.cfg.n_latents, device=sae.W_dec.device)
    seen = 0
    for X in stream:
        v, i = sae.encode(X)
        counts.index_add_(0, i[v > 0], torch.ones(int((v > 0).sum()), device=counts.device))
        seen += len(X)
        if seen >= n_tokens:
            break
    return (counts / seen).cpu().numpy()


@torch.no_grad()
def coactivation(sae: TopKSAE, lm, layer: int, source: str, n_tokens: int, nodes: np.ndarray,
                 window: int = 64, seq_len: int = 256, batch_seqs: int = 32, skip_first: int = 4) -> CoactStats:
    """Streams sequences in order (no shuffling: windows must be contiguous tokens)."""
    from ..finetune import token_stream

    dev = sae.W_dec.device
    n_lat = sae.cfg.n_latents
    node_of = torch.full((n_lat,), -1, dtype=torch.long, device=dev)
    node_of[torch.as_tensor(nodes, device=dev)] = torch.arange(len(nodes), device=dev)
    N = len(nodes)
    nodes_t = torch.as_tensor(nodes, device=dev)
    pair = torch.zeros((N, N), dtype=torch.int32, device=dev)
    fire_tok = torch.zeros(n_lat, dtype=torch.long, device=dev)
    fire_win = torch.zeros(n_lat, dtype=torch.long, device=dev)
    captured = {}
    blk = lm.model.gpt_neox.layers[layer]
    hook = blk.register_forward_hook(lambda m, i, o: captured.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    tokens = token_stream(lm.tokenizer, source, seq_len)
    n_seen = n_win = 0
    usable = ((seq_len - skip_first) // window) * window
    try:
        while n_seen < n_tokens:
            seqs = [next(tokens) for _ in range(batch_seqs)]
            lm.model.gpt_neox(input_ids=torch.tensor(seqs, device=lm.device), use_cache=False)
            h = captured["h"][:, skip_first:skip_first + usable].float()          # [B, T, d]
            B, T, d = h.shape
            v, idx = sae.encode(h.reshape(-1, d))                                   # [B*T, k]
            on = v > 0
            fire_tok.index_add_(0, idx[on], torch.ones(int(on.sum()), dtype=torch.long, device=dev))
            # window membership: [W, n_lat] boolean via scatter
            w_of = (torch.arange(B * T, device=dev) // window)[:, None].expand_as(idx)
            Wn = B * T // window
            M = torch.zeros((Wn, n_lat), dtype=torch.bool, device=dev)
            M[w_of[on], idx[on]] = True
            fire_win += M.sum(0)
            Mh = M[:, nodes_t].half()
            pair += (Mh.T @ Mh).round().int()
            n_seen += B * T
            n_win += Wn
            if (n_seen // (B * T)) % 200 == 0:
                log.info("coactivation: %.1fM / %.1fM tokens", n_seen / 1e6, n_tokens / 1e6)
    finally:
        hook.remove()
    return CoactStats(fire_tok.cpu().numpy(), fire_win.cpu().numpy(), pair.cpu().numpy(), np.asarray(nodes),
                      n_seen, n_win)


def npmi_matrix(st: CoactStats) -> np.ndarray:
    W = st.n_windows
    p_i = st.fire_windows[st.nodes] / W
    p_ij = st.pair_windows / W
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p_ij) - np.log(p_i)[:, None] - np.log(p_i)[None, :]
        npmi = pmi / -np.log(p_ij)
    npmi[~np.isfinite(npmi)] = -1.0
    np.fill_diagonal(npmi, 1.0)
    return npmi


def knn_graph(S: np.ndarray, k: int, min_sim: float) -> sp.csr_matrix:
    """Symmetric sparse distance matrix (1 - similarity) keeping each node's top-k neighbours above min_sim."""
    n = S.shape[0]
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    nb = np.argpartition(-S, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), nb.shape[1])
    cols = nb.ravel()
    sims = S[rows, cols]
    keep = sims > min_sim
    rows, cols, sims = rows[keep], cols[keep], sims[keep]
    D = sp.coo_matrix((1 - sims, (rows, cols)), shape=(n, n)).tocsr()
    D = D.maximum(D.T)  # union; equal values both ways
    return D


def decoder_cosine(sae: TopKSAE, nodes: np.ndarray, chunk: int = 4096) -> np.ndarray:
    W = sae.W_dec.detach()[torch.as_tensor(nodes, device=sae.W_dec.device)]
    W = W / W.norm(dim=1, keepdim=True)
    return torch.cat([W[i:i + chunk] @ W.T for i in range(0, len(W), chunk)]).cpu().numpy()

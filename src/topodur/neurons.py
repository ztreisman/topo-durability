"""Neuron-space statistics (primary Phase 5 nodes, decided with Zack 2026-11-13): MLP neurons, i.e. the
post-activation units of each block's MLP (a privileged basis: the nonlinearity acts per neuron; residual
dimensions are not neurons).

For a set of layers, one streaming pass over Pile text accumulates per neuron: sum, the full second-moment
matrix, and per-token-id sums. By the law of total covariance the neuron covariance splits exactly into
  between-token  Cov_t(E[a_i | token], E[a_j | token])   -> explained by the current token's identity
                                                              (the "inherited from the input" part)
  within-token   E_t[Cov(a_i, a_j | token)]             -> what context, i.e. the network's processing,
                                                              adds on top of the token (the emergent candidate)
Saved per layer (fp16 where large): corr_total, corr_tok (correlation of the between-token parts),
corr_ctx (correlation of the within-token parts), r2_tok (share of each neuron's variance explained by
token identity), top tokens per neuron, token-mean tuning to calibration tokens (single-token items of
token-level families), and each neuron's input and output weight vectors (W_in rows, W_out columns).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


def calibration_tokens(tok, families: list[str]) -> dict[str, list[tuple[int, int]]]:
    """family -> [(item index, token id)] for items that are a single token with a leading space."""
    from .concepts import load_family

    out = {}
    for f in families:
        fam = load_family(f)
        pairs = []
        for i, it in enumerate(fam.items):
            ids = tok(" " + it, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                pairs.append((i, ids[0]))
        out[f] = pairs
    return out


@torch.no_grad()
def neuron_stats(lm, layers: list[int], source: str, n_tokens: int, calib_ids: list[int], seq_len: int = 256,
                 batch_seqs: int = 16, skip_first: int = 4, skip_seqs: int = 0, top_k_tokens: int = 12,
                 min_token_count: int = 50) -> dict[int, dict]:
    from .finetune import token_stream

    dev = lm.device
    cfg = lm.model.config
    V, N = lm.model.gpt_neox.embed_in.weight.shape[0], cfg.intermediate_size
    caps: dict[int, torch.Tensor] = {}
    hooks = [lm.model.gpt_neox.layers[l].mlp.act.register_forward_hook(
        lambda m, i, o, l=l: caps.__setitem__(l, o)) for l in layers]
    acc = {l: {"s": torch.zeros(N, dtype=torch.float64, device=dev),
               "ss": torch.zeros(N, N, dtype=torch.float64, device=dev),
               "tok": torch.zeros(V, N, dtype=torch.float32, device=dev)} for l in layers}
    cnt = torch.zeros(V, dtype=torch.float64, device=dev)
    toks = token_stream(lm.tokenizer, source, seq_len, skip_seqs=skip_seqs)
    seen, step = 0, 0
    try:
        while seen < n_tokens:
            seqs = torch.tensor([next(toks) for _ in range(batch_seqs)], device=dev)
            lm.model.gpt_neox(input_ids=seqs, use_cache=False)
            ids = seqs[:, skip_first:].reshape(-1)
            cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
            for l in layers:
                a = caps[l][:, skip_first:].reshape(-1, N).float()
                acc[l]["s"] += a.sum(0, dtype=torch.float64)
                acc[l]["ss"] += (a.T @ a).double()
                acc[l]["tok"].index_add_(0, ids, a)
            seen += ids.numel()
            step += 1
            if step % 200 == 0:
                log.info("neuron stats: %.1fM / %.1fM tokens", seen / 1e6, n_tokens / 1e6)
    finally:
        for h in hooks:
            h.remove()

    out = {}
    keep = cnt > 0
    c = cnt[keep]
    frequent = (cnt >= min_token_count)[keep]
    tok_ids = torch.nonzero(keep).squeeze(1)
    calib = torch.as_tensor(calib_ids, dtype=torch.long, device=dev)
    for l in layers:
        n = float(seen)
        mu = acc[l]["s"] / n
        cov = acc[l]["ss"] / n - torch.outer(mu, mu)
        T = acc[l].pop("tok")[keep]                                     # [V', N] per-token sums
        M = T / c[:, None].float()                                      # per-token means
        cov_b = (T.T @ M).double() / n - torch.outer(mu, mu)            # sum_t n_t m_t m_t^T / n - mu mu^T
        del T
        # ANOVA bias correction: per-token means of rare tokens carry within-token noise into the
        # between part. E[SSB] = (K-1) s_w^2 + n B, with s_w^2 = SSW / (n - K) unbiased.
        K = float(keep.sum())
        cov_w = (cov - cov_b) * (n / max(n - K, 1.0))
        cov_b = cov_b - (K - 1) / n * cov_w
        var, var_b, var_w = cov.diagonal().clamp_min(1e-12), cov_b.diagonal().clamp_min(1e-12), cov_w.diagonal().clamp_min(1e-12)
        corr = lambda C, v: (C / torch.sqrt(torch.outer(v, v))).clamp(-1, 1).half().cpu().numpy()
        Mf = M[frequent]
        tv, ti = Mf.topk(min(top_k_tokens, len(Mf)), dim=0)              # [k, N]
        calib_mu = torch.zeros(len(calib_ids), N, device=dev)
        pos = {int(t): j for j, t in enumerate(tok_ids.tolist())}
        for j, t in enumerate(calib.tolist()):
            if t in pos:
                calib_mu[j] = M[pos[t]]
        mlp = lm.model.gpt_neox.layers[l].mlp
        out[l] = {
            "corr_total": corr(cov, var), "corr_tok": corr(cov_b, var_b), "corr_ctx": corr(cov_w, var_w),
            "mean": mu.float().cpu().numpy(), "sd": torch.sqrt(var).float().cpu().numpy(),
            "r2_tok": (var_b / var).clamp(0, 1).float().cpu().numpy(),
            "top_tok_ids": tok_ids[frequent][ti].T.cpu().numpy(), "top_tok_vals": tv.T.float().cpu().numpy(),
            "calib_ids": np.asarray(calib_ids), "calib_n": cnt[calib].cpu().numpy(), "calib_mu": calib_mu.T.float().cpu().numpy(),   # [N, n_calib]
            "w_in": mlp.dense_h_to_4h.weight.detach().half().cpu().numpy(),                 # [N, d]
            "w_out": mlp.dense_4h_to_h.weight.detach().T.contiguous().half().cpu().numpy(), # [N, d]
            "n_tokens": seen,
        }
        del M, cov, cov_b, cov_w
        acc[l].clear()
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    return out

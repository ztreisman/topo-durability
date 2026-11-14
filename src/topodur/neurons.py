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
        def corr(C, v, eps):
            # a part with ~no variance (ratio to the neuron's total sd < eps) has no defined correlation: 0, not +-1
            bad = torch.sqrt(v / var) < eps
            R = (C / torch.sqrt(torch.outer(v, v))).clamp(-1, 1)
            R[bad, :] = 0
            R[:, bad] = 0
            return R.half().cpu().numpy()
        Mf = M[frequent]
        tv, ti = Mf.topk(min(top_k_tokens, len(Mf)), dim=0)              # [k, N]
        calib_mu = torch.zeros(len(calib_ids), N, device=dev)
        pos = {int(t): j for j, t in enumerate(tok_ids.tolist())}
        for j, t in enumerate(calib.tolist()):
            if t in pos:
                calib_mu[j] = M[pos[t]]
        blk = lm.model.gpt_neox.layers[l]
        mlp = blk.mlp
        out[l] = {
            "corr_total": corr(cov, var, 1e-3), "corr_tok": corr(cov_b, var_b, 1e-3), "corr_ctx": corr(cov_w, var_w, 1e-3),
            "mean": mu.float().cpu().numpy(), "sd": torch.sqrt(var).float().cpu().numpy(),
            "r2_tok": (var_b / var).clamp(0, 1).float().cpu().numpy(),
            "sd_tok": torch.sqrt(var_b).float().cpu().numpy(), "sd_ctx": torch.sqrt(var_w).float().cpu().numpy(),
            "top_tok_ids": tok_ids[frequent][ti].T.cpu().numpy(), "top_tok_vals": tv.T.float().cpu().numpy(),
            "calib_ids": np.asarray(calib_ids), "calib_n": cnt[calib].cpu().numpy(), "calib_mu": calib_mu.T.float().cpu().numpy(),   # [N, n_calib]
            "w_in": mlp.dense_h_to_4h.weight.detach().half().cpu().numpy(),                 # [N, d]
            "w_out": mlp.dense_4h_to_h.weight.detach().T.contiguous().half().cpu().numpy(), # [N, d]
            "ln_w": blk.post_attention_layernorm.weight.detach().float().cpu().numpy(),   # MLP input LayerNorm
            "n_tokens": seen,
            # per-token means (all tokens seen), for the cross-layer pass; saved to a separate file
            "_tokmeans": M.half().cpu().numpy(), "_tok_ids": tok_ids.cpu().numpy(), "_tok_cnt": c.cpu().numpy(),
        }
        del M, cov, cov_b, cov_w
        acc[l].clear()
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    return out


@torch.no_grad()
def neuron_cross_stats(lm, pairs: list[tuple[int, int]], source: str, n_tokens: int, mu: dict, sd: dict,
                       seq_len: int = 256, batch_seqs: int = 16, skip_first: int = 4, skip_seqs: int = 0):
    """Cross-layer correlation of MLP neurons for layer pairs (l, m), on exactly the token stream of
    neuron_stats (same source, n_tokens, batching), so the per-token means saved there split it into
    between-token and within-token parts (cross_split).

    Activations are standardized with the within-pass mean and sd and scaled by 1/sqrt(batch tokens), so each
    batch product Z_l^T Z_m is a batch-mean correlation of O(1): safe and fast in fp16 on tensor cores
    (fp32 accumulation inside the matmul; per-batch rounding averages out over thousands of batches).
    Returns ({pair: E[z_l z_m^T] as fp32 [N, N]}, {layer: E[z]}, tokens seen)."""
    from .finetune import token_stream

    dev = lm.device
    N = lm.model.config.intermediate_size
    layers = sorted({l for p in pairs for l in p})
    half = dev.type == "cuda"
    caps: dict[int, torch.Tensor] = {}
    hooks = [lm.model.gpt_neox.layers[l].mlp.act.register_forward_hook(
        lambda m, i, o, l=l: caps.__setitem__(l, o)) for l in layers]
    mu_t = {l: torch.as_tensor(mu[l], dtype=torch.float32, device=dev) for l in layers}
    inv = {l: 1.0 / torch.as_tensor(sd[l], dtype=torch.float32, device=dev).clamp_min(1e-6) for l in layers}
    acc = {p: torch.zeros(N, N, dtype=torch.float32, device=dev) for p in pairs}
    zs = {l: torch.zeros(N, dtype=torch.float64, device=dev) for l in layers}
    toks = token_stream(lm.tokenizer, source, seq_len, skip_seqs=skip_seqs)
    seen, nb = 0, 0
    try:
        while seen < n_tokens:
            seqs = torch.tensor([next(toks) for _ in range(batch_seqs)], device=dev)
            lm.model.gpt_neox(input_ids=seqs, use_cache=False)
            B = seqs[:, skip_first:].numel()
            Z = {}
            for l in layers:
                z = (caps.pop(l)[:, skip_first:].reshape(-1, N).float() - mu_t[l]) * inv[l]
                zs[l] += z.sum(0, dtype=torch.float64)
                Z[l] = (z / B ** 0.5).half() if half else z / B ** 0.5
            for (l, m) in pairs:
                acc[(l, m)] += (Z[l].T @ Z[m]).float()
            del Z
            seen += B
            nb += 1
            if nb % 200 == 0:
                log.info("cross stats: %.1fM / %.1fM tokens", seen / 1e6, n_tokens / 1e6)
    finally:
        for h in hooks:
            h.remove()
    for p in pairs:
        acc[p] /= nb
    return acc, {l: (zs[l] / seen).float() for l in layers}, seen


@torch.no_grad()
def cross_split(Ezz: torch.Tensor, zbar_l, zbar_m, tm_l: np.ndarray, tm_m: np.ndarray, cnt: np.ndarray,
                st_l: dict, st_m: dict, n: int, chunk: int = 4096) -> dict[str, torch.Tensor]:
    """Total / between-token / within-token correlation [N, N] between layers l and m (law of total covariance,
    same ANOVA correction as neuron_stats). Ezz = E[z_l z_m^T] in standardized units; tm_* per-token means
    (raw units, fp16); st_* = within-pass mean, sd, sd_tok, sd_ctx of each layer."""
    dev = Ezz.device
    f = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)
    mu_l, sd_l, mu_m, sd_m = f(st_l["mean"]), f(st_l["sd"]).clamp_min(1e-6), f(st_m["mean"]), f(st_m["sd"]).clamp_min(1e-6)
    cov = Ezz - torch.outer(zbar_l, zbar_m)                          # standardized total covariance
    cb = torch.zeros_like(Ezz)
    w = torch.as_tensor(cnt, dtype=torch.float32, device=dev) / n
    for s in range(0, len(cnt), chunk):
        a = (f(tm_l[s:s + chunk]) - mu_l) / sd_l
        b = (f(tm_m[s:s + chunk]) - mu_m) / sd_m
        cb += (a * w[s:s + chunk, None]).T @ b
    cb -= torch.outer(zbar_l, zbar_m)
    K = float((cnt > 0).sum())
    cw = (cov - cb) * (n / max(n - K, 1.0))
    cb = cb - (K - 1) / n * cw
    rb_l, rb_m = f(st_l["sd_tok"]) / sd_l, f(st_m["sd_tok"]) / sd_m   # sd of each part, standardized units
    rw_l, rw_m = f(st_l["sd_ctx"]) / sd_l, f(st_m["sd_ctx"]) / sd_m
    bad = lambda x, y, eps: (x < eps)[:, None] | (y < eps)[None, :]  # a part with ~no variance: undefined, set 0
    out = {"total": cov.masked_fill(bad(f(st_l["sd"]), f(st_m["sd"]), 1e-6), 0.0),
           "tok": (cb / torch.outer(rb_l, rb_m).clamp_min(1e-6)).masked_fill(bad(rb_l, rb_m, 1e-3), 0.0),
           "ctx": (cw / torch.outer(rw_l, rw_m).clamp_min(1e-6)).masked_fill(bad(rw_l, rw_m, 1e-3), 0.0)}
    return {g: M.clamp(-1, 1) for g, M in out.items()}


@torch.no_grad()
def sparse_topk(C: dict[str, torch.Tensor], k: int) -> dict[str, np.ndarray]:
    """Union of each row's and each column's top-k positive entries of every graph in C, with all graphs'
    values at those positions (fp16) - the cross-layer graph's candidate edges."""
    keys = []
    for M in C.values():
        N0, N1 = M.shape
        vr, ir = M.topk(k, dim=1)
        keys.append((torch.arange(N0, device=M.device)[:, None] * N1 + ir).ravel())
        vc, ic = M.topk(k, dim=0)
        keys.append((ic * N1 + torch.arange(N1, device=M.device)[None, :]).ravel())
    key = torch.unique(torch.cat(keys))
    N1 = next(iter(C.values())).shape[1]
    i, j = key // N1, key % N1
    out = {"i": i.to(torch.int16).cpu().numpy(), "j": j.to(torch.int16).cpu().numpy()}
    for g, M in C.items():
        out[g] = M[i, j].half().cpu().numpy()
    return out

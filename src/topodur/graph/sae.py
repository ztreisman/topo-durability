"""TopK sparse autoencoder on the residual stream, trained from streamed activations (Phase 5).

Architecture (Gao et al. 2024 TopK SAE, as in EleutherAI's sparsify): z = TopK(W_enc (x - b_dec) + b_enc),
x_hat = W_dec z + b_dec, unit-norm decoder rows, MSE loss plus the AuxK loss on the top k_aux dead latents
(dead = not fired in the last `dead_tokens` tokens). Inputs are scaled by a fixed constant estimated
on the first batches so the mean squared norm is d (fp16 residual norms in Pythia reach ~1000).

Activations come from the model truncated after the target block (blocks past it are never run).
Training is resumable from `state.pt`.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class SAEConfig:
    d_in: int = 2048
    n_latents: int = 32768
    k: int = 32
    k_aux: int = 512
    aux_coef: float = 1 / 32
    dead_tokens: int = 10_000_000
    lr: float | None = None          # default: 2e-4 / sqrt(n_latents / 16384) (Gao et al.)


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        W = torch.randn(cfg.n_latents, cfg.d_in)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W.clone())                    # [n, d]
        self.W_enc = nn.Parameter(W.clone())                    # tied init
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_latents))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("last_fired", torch.zeros(cfg.n_latents, dtype=torch.long))
        self.register_buffer("tokens_seen", torch.tensor(0, dtype=torch.long))

    def pre(self, x):
        return (x * self.scale - self.b_dec) @ self.W_enc.T + self.b_enc

    def encode(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k values and indices, [B, k]."""
        v, i = self.pre(x).topk(self.cfg.k, dim=-1)
        return F.relu(v), i

    def _sparse_decode(self, v, i):
        """sum_j v[:, j] * W_dec[i[:, j]] without materializing [B, k, d] (at k_aux=512 and B=4096
        that tensor is 16 GB; the L14 run OOMed on 10.5 GB cards once dead latents appeared)."""
        shp = v.shape
        out = F.embedding_bag(i.reshape(-1, shp[-1]), self.W_dec, per_sample_weights=v.reshape(-1, shp[-1]),
                              mode="sum")
        return out.reshape(*shp[:-1], self.W_dec.shape[1])

    def decode(self, v, i):
        return self._sparse_decode(v, i) + self.b_dec

    def forward(self, x):
        xs = x * self.scale
        pre = (xs - self.b_dec) @ self.W_enc.T + self.b_enc
        v, i = pre.topk(self.cfg.k, dim=-1)
        v = F.relu(v)
        xh = self.decode(v, i)
        err = xs - xh
        mse = err.pow(2).sum(-1).mean()
        # bookkeeping for dead latents
        n = x.shape[0]
        self.tokens_seen += n
        fired = torch.zeros(self.cfg.n_latents, dtype=torch.bool, device=x.device)
        fired[i[v > 0]] = True
        self.last_fired[fired] = self.tokens_seen
        dead = (self.tokens_seen - self.last_fired) > self.cfg.dead_tokens
        aux = torch.zeros((), device=x.device)
        if dead.any() and self.tokens_seen > self.cfg.dead_tokens:
            k_aux = min(self.cfg.k_aux, int(dead.sum()))
            pre_dead = pre.masked_fill(~dead, -torch.inf)
            va, ia = pre_dead.topk(k_aux, dim=-1)
            va = F.relu(va)
            err_hat = self._sparse_decode(va, ia)
            aux = (err.detach() - err_hat).pow(2).sum(-1).mean() / err.detach().pow(2).sum(-1).mean().clamp_min(1e-9)
        total_var = (xs - xs.mean(0)).pow(2).sum(-1).mean()
        return {"loss": mse + self.cfg.aux_coef * aux, "mse": mse, "aux": aux,
                "fvu": mse / total_var.clamp_min(1e-9), "dead_frac": dead.float().mean()}

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self):
        """Project out the gradient component parallel to each decoder row (keeps unit norm)."""
        if self.W_dec.grad is not None:
            g = self.W_dec.grad
            g -= (g * self.W_dec.data).sum(1, keepdim=True) * self.W_dec.data

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), d / "sae.pt")
        (d / "cfg.json").write_text(json.dumps(asdict(self.cfg)))

    @classmethod
    def load(cls, d: Path, device="cpu") -> "TopKSAE":
        m = cls(SAEConfig(**json.loads((d / "cfg.json").read_text())))
        m.load_state_dict(torch.load(d / "sae.pt", map_location=device))
        return m.to(device)


class ActStream:
    """Residual-stream vectors after block `layer`, streamed from text in fixed-length sequences.
    The model is truncated after the block (later blocks are dropped). Positions < skip_first are
    dropped (attention-sink tokens have outlier norms). Yields [n, d] fp32 batches, shuffled within a
    buffer of `buffer_seqs` sequences."""

    def __init__(self, lm, layer: int, source: str, seq_len: int = 256, batch_seqs: int = 32,
                 buffer_seqs: int = 512, skip_first: int = 4, skip_seqs: int = 0, seed: int = 0):
        from ..finetune import token_stream

        self.lm, self.layer, self.seq_len, self.batch_seqs = lm, layer, seq_len, batch_seqs
        self.buffer_seqs, self.skip_first = buffer_seqs, skip_first
        self.tokens = token_stream(lm.tokenizer, source, seq_len, skip_seqs=skip_seqs)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        lm.model.gpt_neox.layers = lm.model.gpt_neox.layers[: layer + 1]   # truncate
        self.seqs_read = skip_seqs

    @torch.no_grad()
    def __iter__(self):
        captured = {}
        blk = self.lm.model.gpt_neox.layers[self.layer]
        hook = blk.register_forward_hook(lambda m, i, o: captured.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        try:
            while True:
                buf = []
                for _ in range(self.buffer_seqs // self.batch_seqs):
                    seqs = [next(self.tokens) for _ in range(self.batch_seqs)]
                    self.seqs_read += len(seqs)
                    self.lm.model.gpt_neox(input_ids=torch.tensor(seqs, device=self.lm.device), use_cache=False)
                    buf.append(captured["h"][:, self.skip_first:].reshape(-1, captured["h"].shape[-1]).float())
                X = torch.cat(buf)
                X = X[torch.randperm(len(X), generator=self.gen).to(X.device)]
                yield X
        finally:
            hook.remove()


def train_sae(lm, layer: int, source: str, out: Path, cfg: SAEConfig, total_tokens: int, batch_tokens: int = 4096,
              ckpt_every: int = 500, limit_steps: int | None = None, seed: int = 0) -> TopKSAE:
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dev = lm.device
    sae = TopKSAE(cfg).to(dev)
    lr = cfg.lr or 2e-4 / math.sqrt(cfg.n_latents / 2**14)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=6.25e-10)
    step, seqs_read = 0, 0
    state = out / "state.pt"
    if state.exists():
        st = torch.load(state, map_location=dev, weights_only=False)
        sae.load_state_dict(st["sae"])
        opt.load_state_dict(st["opt"])
        step, seqs_read = st["step"], st["seqs_read"]
        log.info("resumed SAE at step %d (%d tokens)", step, step * batch_tokens)
    stream = ActStream(lm, layer, source, skip_seqs=seqs_read, seed=seed + step)
    total_steps = total_tokens // batch_tokens
    logf = open(out / "train_log.jsonl", "a")
    t0, toks = time.time(), 0
    for X in stream:
        if step == 0 and float(sae.scale) == 1.0:
            sae.scale.fill_(math.sqrt(cfg.d_in) / X.norm(dim=1).mean().item())  # mean squared norm ~ d
            with torch.no_grad():
                sae.b_dec.data = (X * sae.scale).median(0).values
            log.info("input scale %.4g", float(sae.scale))
        for b in range(0, len(X) - batch_tokens + 1, batch_tokens):
            if step >= total_steps or (limit_steps and step >= limit_steps):
                break
            r = sae(X[b:b + batch_tokens])
            opt.zero_grad(set_to_none=True)
            r["loss"].backward()
            sae.remove_parallel_grad()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sae.normalize_decoder()
            step += 1
            toks += batch_tokens
            if step % 100 == 0:
                rec = {"step": step, "tokens": step * batch_tokens, **{k: float(v) for k, v in r.items()},
                       "tok_per_s": toks / (time.time() - t0)}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if step % 1000 == 0:
                    log.info("step %d/%d fvu %.3f dead %.3f (%.0f tok/s)", step, total_steps, rec["fvu"],
                             rec["dead_frac"], rec["tok_per_s"])
            if step % ckpt_every == 0:
                tmp = state.with_suffix(".tmp")
                torch.save({"sae": sae.state_dict(), "opt": opt.state_dict(), "step": step,
                            "seqs_read": stream.seqs_read}, tmp)
                tmp.replace(state)
        if step >= total_steps or (limit_steps and step >= limit_steps):
            break
    sae.save(out / "final")
    logf.close()
    return sae

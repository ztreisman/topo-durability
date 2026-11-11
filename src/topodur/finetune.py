"""LoRA fine-tuning on an unrelated corpus (Phase 3, durability measure 3 and the primary metric).

Base weights stay fp16 and frozen; LoRA weights are fp32, trained under fp16 autocast with a
GradScaler (Turing has no bf16). Documents are tokenized, joined with EOS, and cut into fixed-length
sequences. Saves the adapter at fixed token counts (milestones) and a resumable training checkpoint
(adapter + optimizer + scaler + data position) every `ckpt_every` steps.

Output dir layout:
  adapters/<N>M/            LoRA adapter after N million training tokens (0M = untrained adapter)
  train_state.pt            resume point
  train_log.jsonl           step, tokens, loss, lr
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .models import load_model
from .pile import stream_texts

log = logging.getLogger(__name__)


@dataclass
class FTConfig:
    size: str = "1.4b"
    step: int = 143000
    corpus: str = "mirror:corpora/code_py_30000.jsonl"
    total_tokens: int = 20_000_000
    milestones: list[int] = field(default_factory=lambda: [0, 1_000_000, 5_000_000, 10_000_000, 20_000_000])
    seq_len: int = 512
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.0
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["query_key_value", "dense", "dense_h_to_4h",
                                                                "dense_4h_to_h"])
    ckpt_every: int = 200
    seed: int = 0
    grad_checkpointing: bool = True


def token_stream(tok, source: str, seq_len: int, skip_seqs: int = 0):
    """Yields lists of seq_len token ids from concatenated documents (EOS between documents),
    skipping the first skip_seqs sequences (resume). Loops over the corpus if it runs out."""
    buf: list[int] = []
    n = 0
    epoch = 0
    while True:
        for text in stream_texts(source, min_chars=1):
            buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
            while len(buf) >= seq_len:
                seq, buf = buf[:seq_len], buf[seq_len:]
                if n >= skip_seqs:
                    yield seq
                n += 1
        epoch += 1
        log.warning("corpus exhausted; starting epoch %d", epoch + 1)


def milestone_name(tokens: int) -> str:
    return f"{tokens / 1e6:g}M"


def train(cfg: FTConfig, out: Path, limit_steps: int | None = None) -> None:
    from peft import LoraConfig, get_peft_model

    out.mkdir(parents=True, exist_ok=True)
    (out / "ft_config.json").write_text(json.dumps(asdict(cfg), indent=1))
    torch.manual_seed(cfg.seed)
    lm = load_model(cfg.size, cfg.step, dtype="fp16")
    model = lm.model
    model.config.use_cache = False
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    pcfg = LoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, target_modules=cfg.target_modules,
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, pcfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights in fp32; the frozen base stays fp16
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    tokens_per_step = cfg.seq_len * cfg.batch_size * cfg.grad_accum
    total_steps = math.ceil(cfg.total_tokens / tokens_per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda")

    step = 0
    state = out / "train_state.pt"
    if state.exists():
        st = torch.load(state, map_location="cpu", weights_only=False)
        model.load_state_dict(st["lora"], strict=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step = st["step"]
        log.info("resumed at step %d (%d tokens)", step, step * tokens_per_step)

    def save_milestones(tokens_done: int):
        for m in cfg.milestones:
            d = out / "adapters" / milestone_name(m)
            if m <= tokens_done and not (d / ".complete").exists():
                model.save_pretrained(d)
                (d / ".complete").write_text(json.dumps({"tokens": tokens_done, "step": step}))
                log.info("saved adapter %s at %d tokens", d.name, tokens_done)

    save_milestones(step * tokens_per_step)
    stream = token_stream(lm.tokenizer, cfg.corpus, cfg.seq_len, skip_seqs=step * cfg.batch_size * cfg.grad_accum)
    model.train()
    logf = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    while step < total_steps and (limit_steps is None or step < limit_steps):
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(cfg.grad_accum):
            ids = torch.tensor([next(stream) for _ in range(cfg.batch_size)], device=lm.device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, labels=ids).loss / cfg.grad_accum
            scaler.scale(loss).backward()
            loss_sum += float(loss)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        tokens = step * tokens_per_step
        if step % 10 == 0:
            rec = {"step": step, "tokens": tokens, "loss": loss_sum, "lr": sched.get_last_lr()[0],
                   "tok_per_s": tokens_per_step * 10 / (time.time() - t0)}
            t0 = time.time()
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if step % 100 == 0:
                log.info("step %d/%d tokens %.2fM loss %.3f (%.0f tok/s)", step, total_steps, tokens / 1e6,
                         loss_sum, rec["tok_per_s"])
        save_milestones(tokens)
        if step % cfg.ckpt_every == 0 or step == total_steps:
            lora = {k: v for k, v in model.state_dict().items() if "lora_" in k}
            tmp = state.with_suffix(".tmp")
            torch.save({"lora": lora, "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step}, tmp)
            tmp.replace(state)
    logf.close()
    if step >= total_steps:
        save_milestones(cfg.total_tokens)
        (out / ".complete").write_text(json.dumps({"steps": step}))


@torch.no_grad()
def perplexity(lm, source: str, n_seqs: int = 200, seq_len: int = 512, batch_size: int = 8) -> float:
    """Held-out perplexity over the first n_seqs fixed-length sequences of `source`."""
    stream = token_stream(lm.tokenizer, source, seq_len)
    seqs = [next(stream) for _ in range(n_seqs)]
    nll, n = 0.0, 0
    lm.model.eval()
    for i in range(0, n_seqs, batch_size):
        ids = torch.tensor(seqs[i:i + batch_size], device=lm.device)
        logits = lm.model(input_ids=ids).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
        nll -= float(lp.sum())
        n += lp.numel()
    return math.exp(nll / n)

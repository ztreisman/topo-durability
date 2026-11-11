"""Load Pythia at a given training step and dtype, with a NaN/overflow check.

Variant: STANDARD Pythia (trained on the standard Pile). PolyPythias
(van der Wal et al. 2025, arXiv:2503.09543) also uses the standard Pile, so the
seed comparisons stay on the same data. Do not mix in the -deduped models.

Pythia itself was trained in fp16, so fp16 inference is native for it; the check
below still runs for every new (model, step) because early checkpoints and
fine-tuned weights can behave differently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM, PreTrainedTokenizerBase

from . import mirror

log = logging.getLogger(__name__)

PYTHIA_SIZES = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
POLYPYTHIA_SIZES = ["14m", "31m", "70m", "160m", "410m"]  # seeds 1..9 (verify on the hub)
FINAL_STEP = 143000
DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
FP16_MAX = 65504.0


def all_checkpoint_steps() -> list[int]:
    return [0] + [2**i for i in range(10)] + list(range(1000, FINAL_STEP + 1, 1000))


def log_spaced_steps() -> list[int]:
    """~30 checkpoints: 0, 1, 2, 4, ..., 512, then 1000, 2000, 4000, ..., 128000, 143000."""
    steps = [0] + [2**i for i in range(10)]
    s = 1000
    while s < FINAL_STEP:
        steps.append(s)
        s *= 2
    # densify the doubling tail so the late phase is not just 3 points
    extra = [3000, 6000, 12000, 24000, 48000, 96000, 112000, 128000]
    return sorted(set(steps + extra + [FINAL_STEP]))


def repo_id(size: str, seed: int | None = None) -> str:
    if size not in PYTHIA_SIZES:
        raise ValueError(f"unknown Pythia size {size!r}")
    if seed is None or seed == 0:
        return f"EleutherAI/pythia-{size}"
    if size not in POLYPYTHIA_SIZES:
        raise ValueError(f"PolyPythia seeds exist only for {POLYPYTHIA_SIZES}")
    return f"EleutherAI/pythia-{size}-seed{seed}"


@dataclass
class LoadedModel:
    model: GPTNeoXForCausalLM
    tokenizer: PreTrainedTokenizerBase
    size: str
    step: int
    seed: int | None
    dtype: str
    device: torch.device

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    @property
    def blocks(self):
        return self.model.gpt_neox.layers


def load_tokenizer(size: str) -> PreTrainedTokenizerBase:
    # The tokenizer is identical across sizes and steps; load from the final revision.
    repo = repo_id(size)
    local = mirror.resolve(mirror.tokenizer_dir(repo), f"tokenizer for {repo}")
    tok = AutoTokenizer.from_pretrained(local) if local else AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(
    size: str,
    step: int = FINAL_STEP,
    dtype: str = "fp16",
    device: str | None = None,
    seed: int | None = None,
    multi_gpu: bool = False,
    check: bool = True,
    fallback_fp32: bool = True,
) -> LoadedModel:
    if step not in all_checkpoint_steps():
        raise ValueError(f"step {step} is not a published Pythia checkpoint")
    if dtype == "bf16" and torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (8, 0):
        raise ValueError("bf16 is not native on Turing (sm_75); use fp16 or fp32")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and dtype == "fp16":
        log.info("fp16 on CPU is slow and partly unsupported; using fp32")
        dtype = "fp32"

    repo = repo_id(size, seed)
    local = mirror.resolve(mirror.model_dir(repo, step), f"{repo}@step{step}")
    kwargs = dict(torch_dtype=DTYPES[dtype])
    if multi_gpu:
        kwargs["device_map"] = "auto"
    log.info("loading %s @ step%d (%s) from %s", repo, step, dtype, local or "the hub")
    if local:
        model = GPTNeoXForCausalLM.from_pretrained(local, **kwargs)
    else:
        model = GPTNeoXForCausalLM.from_pretrained(repo, revision=f"step{step}", **kwargs)
    if not multi_gpu:
        model.to(device)
    model.eval()
    lm = LoadedModel(model, load_tokenizer(size), size, step, seed, dtype,
                     next(model.parameters()).device)

    if check:
        bad = numerics_check(lm)
        if bad:
            msg = f"non-finite or near-overflow activations in layers {bad} ({size} step{step} {dtype})"
            if dtype == "fp16" and fallback_fp32:
                log.warning(msg + "; reloading in fp32")
                del model, lm
                torch.cuda.empty_cache()
                return load_model(size, step, "fp32", device, seed, multi_gpu, check, False)
            raise RuntimeError(msg)
    return lm


@torch.no_grad()
def numerics_check(lm: LoadedModel, texts: list[str] | None = None, frac: float = 0.5) -> list[int]:
    """Run a small batch; return block indices whose output is non-finite or exceeds
    `frac` of the fp16 max (a warning sign before an outright overflow)."""
    texts = texts or [
        "The quick brown fox jumps over the lazy dog. " * 8,
        "In January the committee met on a Wednesday to review the budget.",
        "def f(x):\n    return x ** 2 + 1\n" * 4,
        "第一章 天下大势，分久必合，合久必分。",
    ]
    enc = lm.tokenizer(texts, return_tensors="pt", padding=True).to(lm.device)
    worst: dict[int, float] = {}
    bad: list[int] = []

    def hook(i):
        def fn(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(h).all():
                bad.append(i)
            worst[i] = float(h.float().abs().max())
        return fn

    hs = [b.register_forward_hook(hook(i)) for i, b in enumerate(lm.blocks)]
    try:
        out = lm.model(**enc)
    finally:
        for h in hs:
            h.remove()
    if not torch.isfinite(out.logits).all():
        bad.append(-1)
    if lm.dtype == "fp16":
        bad += [i for i, v in worst.items() if v > frac * FP16_MAX]
    log.info("numerics check: max |h| per block = %s", {i: round(v, 1) for i, v in worst.items()})
    return sorted(set(bad))

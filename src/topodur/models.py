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
from pathlib import Path
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


def parse_variant(spec: str | None) -> dict:
    """Model variants for the durability measures (Phase 3). Spec strings:
      lora:<adapter dir>      LoRA adapter merged into the base weights (finetune.py output)
      int8 | nf4              bitsandbytes 8-bit / 4-bit NF4 weight quantization
      prune:<frac>            unstructured magnitude pruning of every block Linear, per matrix
      noise:<sigma>[:<seed>]  W += sigma * std(W) * N(0, 1) for every block Linear
    """
    if not spec:
        return {"kind": None}
    kind, _, rest = spec.partition(":")
    if kind == "lora":
        return {"kind": "lora", "path": rest}
    if kind in ("int8", "nf4"):
        return {"kind": kind}
    if kind == "prune":
        return {"kind": "prune", "frac": float(rest)}
    if kind == "noise":
        sig, _, sd = rest.partition(":")
        return {"kind": "noise", "sigma": float(sig), "seed": int(sd or 0)}
    raise ValueError(f"unknown model variant {spec!r}")


def variant_tag(spec: str | None) -> str:
    """Short filesystem-safe tag for a variant, appended to the unit id after '+'."""
    v = parse_variant(spec)
    k = v["kind"]
    if k is None:
        return ""
    if k == "lora":
        # .../finetune/<size>_<corpus>/adapters/<tokens> -> lora-<corpus>-<tokens>; the corpus must be in the
        # tag, or the code and wiki_de adapters at the same token count collide on one unit id
        p = Path(v["path"])
        run = p.parent.parent.name if p.parent.name == "adapters" else p.parent.name
        corpus = run.split("_", 1)[1] if "_" in run else run
        return f"lora-{corpus}-{p.name}"
    if k == "prune":
        return f"prune{v['frac']:g}"
    if k == "noise":
        return f"noise{v['sigma']:g}s{v['seed']}"
    return k


def _block_linears(model):
    for blk in model.gpt_neox.layers:
        for m in blk.modules():
            if isinstance(m, torch.nn.Linear):
                yield m


@torch.no_grad()
def apply_weight_variant(model, v: dict) -> None:
    if v["kind"] == "prune":
        for m in _block_linears(model):
            w = m.weight
            k = int(v["frac"] * w.numel())
            if k:
                thr = w.abs().float().flatten().kthvalue(k).values
                w.masked_fill_(w.abs() <= thr, 0)
    elif v["kind"] == "noise":
        g = torch.Generator(device="cpu").manual_seed(v["seed"])
        for m in _block_linears(model):
            w = m.weight
            noise = torch.randn(w.shape, generator=g, dtype=torch.float32)
            w.add_((v["sigma"] * w.float().std() * noise.to(w.device)).to(w.dtype))


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
    variant: str | None = None,
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
    v = parse_variant(variant)
    kwargs = dict(torch_dtype=DTYPES[dtype])
    if multi_gpu:
        kwargs["device_map"] = "auto"
    if v["kind"] in ("int8", "nf4"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if v["kind"] == "int8" else
                                         BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=DTYPES[dtype]))
        kwargs["device_map"] = {"": 0}
    log.info("loading %s @ step%d (%s, variant %s) from %s", repo, step, dtype, variant, local or "the hub")
    if local:
        model = GPTNeoXForCausalLM.from_pretrained(local, **kwargs)
    else:
        model = GPTNeoXForCausalLM.from_pretrained(repo, revision=f"step{step}", **kwargs)
    if v["kind"] == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, v["path"]).merge_and_unload()
    if v["kind"] in ("prune", "noise"):
        apply_weight_variant(model, v)
    if not multi_gpu and v["kind"] not in ("int8", "nf4"):
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
                return load_model(size, step, "fp32", device, seed, multi_gpu, check, False, variant)
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


def unit_id(size: str, step: int, seed: int = 0, variant: str | None = None) -> str:
    """Results-directory id of one (model, checkpoint, seed, variant): e.g. 1.4b_s0_step143000+prune0.25."""
    tag = variant_tag(variant)
    return f"{size}_s{seed}_step{step}" + (f"+{tag}" if tag else "")

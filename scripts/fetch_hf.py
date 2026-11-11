"""Laptop side of the offline workflow: lambda01 cannot reach Hugging Face.

Downloads model checkpoints and a Pile sample into a plain-directory mirror, which
scripts/push_data.sh then copies to lambda01. models.py and pile.py read from the mirror whenever
TOPODUR_HF_MIRROR is set (setup_lambda.sh sets it, along with HF_HUB_OFFLINE=1).

Layout:
  <mirror>/models/EleutherAI/pythia-1.4b/step143000/{config.json, model.safetensors | pytorch_model.bin}
  <mirror>/models/EleutherAI/pythia-1.4b/tokenizer/...
  <mirror>/pile/pile_sample_<N>.jsonl
Each finished directory/file gets a .complete marker, so reruns and pushes skip it.

  uv run python scripts/fetch_hf.py models --sizes 70m 1.4b 2.8b --steps 143000
  uv run python scripts/fetch_hf.py models --sizes 1.4b --steps logspaced     # Phase 3 trajectory (~80 GB)
  uv run python scripts/fetch_hf.py pile --n-docs 200000
  uv run python scripts/fetch_hf.py text --source codeparrot/codeparrot-clean --field content       --n-docs 60000 --name corpora/code.jsonl                                 # fine-tuning corpora
  uv run python scripts/fetch_hf.py list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodur.mirror import mirror_root, model_dir, tokenizer_dir  # noqa: E402

WEIGHT_PREF = ["model.safetensors", "pytorch_model.bin"]


def fetch_model(repo: str, step: int) -> None:
    from huggingface_hub import HfApi, snapshot_download

    out = model_dir(repo, step)
    if (out / ".complete").exists():
        print(f"have {repo}@step{step}")
        return
    files = HfApi().list_repo_files(repo, revision=f"step{step}")
    weights = next((w for w in WEIGHT_PREF if w in files), None)
    if weights is None:  # sharded checkpoints
        weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
        extra = [f for f in files if f.endswith(".index.json")]
    else:
        extra = []
    print(f"fetching {repo}@step{step} ({weights})")
    snapshot_download(repo, revision=f"step{step}", local_dir=out,
                      allow_patterns=["config.json", "generation_config.json", weights, *extra])
    (out / ".complete").write_text(json.dumps({"repo": repo, "step": step, "weights": weights}))


def fetch_tokenizer(repo: str) -> None:
    from huggingface_hub import snapshot_download

    out = tokenizer_dir(repo)
    if (out / ".complete").exists():
        return
    snapshot_download(repo, revision="step143000", local_dir=out,
                      allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
    (out / ".complete").write_text("{}")


def fetch_pile(n_docs: int, source: str) -> None:
    fetch_text(source, None, "train", n_docs, f"pile/pile_sample_{n_docs}.jsonl", "text", 0)


def fetch_text(source: str, config: str | None, split: str, n_docs: int, name: str, field: str,
               skip: int, data_files: str | None = None, min_chars: int = 0) -> None:
    """Stream n_docs documents of `source` into <mirror>/<name> as JSONL {"text", "meta"}."""
    from datasets import load_dataset

    out = mirror_root() / name
    if Path(str(out) + ".complete").exists():
        print(f"have {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".part")
    kw = {"data_files": data_files} if data_files else {}
    ds = load_dataset(source, config, split=split, streaming=True, **kw)
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i < skip:
                continue
            if n >= n_docs:
                break
            text = ex[field]
            if len(text) < min_chars:
                continue
            f.write(json.dumps({"text": text, "meta": ex.get("meta")}) + "\n")
            n += 1
            if n % 20000 == 0:
                print(f"  {name}: {n}/{n_docs} docs", flush=True)
    os.replace(tmp, out)
    Path(str(out) + ".complete").write_text(json.dumps({"source": source, "config": config, "split": split,
                                                        "n_docs": n, "skip": skip}))
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


def main() -> int:
    from topodur.models import log_spaced_steps, repo_id

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models")
    m.add_argument("--sizes", nargs="+", required=True)
    m.add_argument("--steps", default="143000", help="comma list or 'logspaced'")
    m.add_argument("--seeds", nargs="*", type=int, default=[0], help="PolyPythia seeds (0 = original)")
    p = sub.add_parser("pile")
    p.add_argument("--n-docs", type=int, default=200000)
    p.add_argument("--source", default="monology/pile-uncopyrighted")
    t = sub.add_parser("text", help="any HF text dataset -> <mirror>/<name> (JSONL)")
    t.add_argument("--source", required=True)
    t.add_argument("--config", default=None)
    t.add_argument("--split", default="train")
    t.add_argument("--data-files", default=None)
    t.add_argument("--n-docs", type=int, required=True)
    t.add_argument("--skip", type=int, default=0)
    t.add_argument("--field", default="text")
    t.add_argument("--min-chars", type=int, default=0)
    t.add_argument("--name", required=True, help="path under the mirror, e.g. corpora/code.jsonl")
    sub.add_parser("list")
    a = ap.parse_args()

    if a.cmd == "models":
        steps = log_spaced_steps() if a.steps == "logspaced" else [int(s) for s in a.steps.split(",")]
        for size in a.sizes:
            fetch_tokenizer(repo_id(size))
            for seed in a.seeds:
                for step in steps:
                    fetch_model(repo_id(size, seed or None), step)
    elif a.cmd == "pile":
        fetch_pile(a.n_docs, a.source)
    elif a.cmd == "text":
        fetch_text(a.source, a.config, a.split, a.n_docs, a.name, a.field, a.skip, a.data_files, a.min_chars)
    else:
        for c in sorted(mirror_root().rglob(".complete")):
            print(c.parent.relative_to(mirror_root()))
        for c in sorted(mirror_root().rglob("*.jsonl.complete")):
            print(c.relative_to(mirror_root()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

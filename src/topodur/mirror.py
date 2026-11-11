"""Local mirror of Hugging Face models and Pile text (lambda01 cannot reach the hub).

TOPODUR_HF_MIRROR points at the mirror root; default <repo>/data/hf_mirror. When the env var is set,
models.py loads ONLY from the mirror and fails with a clear message if a checkpoint is missing,
instead of silently trying the network.
"""

from __future__ import annotations

import os
from pathlib import Path

from .runs import DATA_DIR


def mirror_root() -> Path:
    return Path(os.environ.get("TOPODUR_HF_MIRROR", DATA_DIR / "hf_mirror"))


def mirror_required() -> bool:
    return "TOPODUR_HF_MIRROR" in os.environ


def model_dir(repo: str, step: int) -> Path:
    return mirror_root() / "models" / repo / f"step{step}"


def tokenizer_dir(repo: str) -> Path:
    return mirror_root() / "models" / repo / "tokenizer"


def resolve(path: Path, what: str) -> str | None:
    """Mirror path if complete; None to use the hub; error if the mirror is required but missing."""
    if (path / ".complete").exists():
        return str(path)
    if mirror_required():
        raise FileNotFoundError(
            f"{what} not in the mirror at {path}. Fetch it on the laptop with scripts/fetch_hf.py "
            f"and push with scripts/push_data.sh (lambda01 cannot reach Hugging Face)."
        )
    return None

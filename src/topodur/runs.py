"""Run directories, provenance, logging, atomic writes, and the cache disk budget.

Every run writes to data/results/<experiment>/<run_id>/ with:
  config.yaml      resolved config
  provenance.json  git hash, seed, package versions, GPU info, host, time
  run.log          Python logging output
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("TOPODUR_DATA", REPO_ROOT / "data"))
RESULTS_DIR = DATA_DIR / "results"
CACHE_DIR = DATA_DIR / "cache"
CACHE_BUDGET_BYTES = int(float(os.environ.get("TOPODUR_CACHE_BUDGET_TB", "1.5")) * 1e12)

log = logging.getLogger("topodur")


# ---------------------------------------------------------------- config

def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in (overrides or {}).items():
        if v is not None:
            _set_dotted(cfg, k, v)
    return cfg


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


# ---------------------------------------------------------------- provenance

def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        h = out.stdout.strip() or "nogit"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    names = ["torch", "transformers", "datasets", "numpy", "scipy", "ripser", "persim", "gudhi", "pandas"]
    out = {"python": sys.version.split()[0]}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = "missing"
    return out


def gpu_info() -> list[dict]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception as e:  # pragma: no cover
        return [{"error": repr(e)}]


def provenance(seed: int | None) -> dict:
    return {
        "git": git_hash(),
        "seed": seed,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "argv": sys.argv,
        "packages": package_versions(),
        "gpus": gpu_info(),
    }


# ---------------------------------------------------------------- run dirs

def run_dir(experiment: str, run_id: str) -> Path:
    return RESULTS_DIR / experiment / run_id


def start_run(experiment: str, run_id: str, cfg: dict, seed: int | None = None) -> Path:
    """Create the run dir, save config + provenance, attach a file log handler."""
    d = run_dir(experiment, run_id)
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(d / "config.yaml", yaml.safe_dump(cfg, sort_keys=False))
    atomic_write_text(d / "provenance.json", json.dumps(provenance(seed), indent=2))
    setup_logging(d / "run.log")
    log.info("run %s/%s started", experiment, run_id)
    return d


def setup_logging(path: Path | None = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve()
                   for h in root.handlers):
            fh = logging.FileHandler(path)
            fh.setFormatter(fmt)
            root.addHandler(fh)


# ---------------------------------------------------------------- atomic writes

@contextmanager
def atomic_path(path: str | Path) -> Iterator[Path]:
    """Yield a temp path in the same dir; rename onto `path` only on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix or ".tmp")
    os.close(fd)
    tmp = Path(tmp)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: str | Path, text: str) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")


def mark_done(path: Path, info: dict | None = None) -> None:
    atomic_write_text(Path(str(path) + ".done"), json.dumps(info or {}))


def is_done(path: Path) -> bool:
    return Path(str(path) + ".done").exists()


# ---------------------------------------------------------------- disk budget

def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def check_cache_budget(planned_bytes: int = 0) -> None:
    """Raise before writing if the activation cache would exceed the budget."""
    used = dir_size(CACHE_DIR) if CACHE_DIR.exists() else 0
    free = shutil.disk_usage(DATA_DIR if DATA_DIR.exists() else REPO_ROOT).free
    if used + planned_bytes > CACHE_BUDGET_BYTES:
        raise RuntimeError(
            f"cache budget exceeded: used {used/1e12:.3f} TB + planned {planned_bytes/1e12:.3f} TB "
            f"> {CACHE_BUDGET_BYTES/1e12:.2f} TB. Delete consumed caches (ask first)."
        )
    if planned_bytes > free * 0.9:
        raise RuntimeError(f"not enough free disk: planned {planned_bytes/1e9:.1f} GB, free {free/1e9:.1f} GB")

"""Streaming text from the Pile.

Pythia (standard) was trained on the standard Pile. The original Pile mirror is gone from the
hub; `monology/pile-uncopyrighted` is the standard Pile minus the copyrighted subsets
(Books3, BookCorpus2, OpenSubtitles, YTSubtitles, OWT2), which is the closest public
streaming source. Its jsonl shards are pre-shuffled, so the first N docs are a fair sample.
A local .jsonl/.txt file can be used instead (offline machines, tests). A source of the form
"mirror:<file>" resolves to <TOPODUR_HF_MIRROR>/pile/<file> (written by scripts/fetch_hf.py pile);
lambda01 cannot reach Hugging Face, so it always uses that form.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

DEFAULT_SOURCE = "monology/pile-uncopyrighted"


def stream_texts(source: str = DEFAULT_SOURCE, split: str = "train", skip: int = 0,
                 min_chars: int = 200) -> Iterator[str]:
    from . import mirror

    if source.startswith("mirror:"):
        p = mirror.mirror_root() / "pile" / source.removeprefix("mirror:")
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run scripts/fetch_hf.py pile on the laptop, then push_data.sh")
    else:
        p = Path(source)
    if p.exists():
        it = _local(p)
    else:
        if mirror.mirror_required():
            raise RuntimeError(f"cannot stream {source!r} from the hub on an offline machine; use mirror:<file>")
        from datasets import load_dataset

        ds = load_dataset(source, split=split, streaming=True)
        if skip:
            ds = ds.skip(skip)
        it = (ex["text"] for ex in ds)
    for t in it:
        if len(t) >= min_chars:
            yield t


def _local(p: Path) -> Iterator[str]:
    with open(p, encoding="utf-8") as f:
        if p.suffix == ".jsonl":
            for line in f:
                yield json.loads(line)["text"]
        else:
            for para in f.read().split("\n\n"):
                yield para

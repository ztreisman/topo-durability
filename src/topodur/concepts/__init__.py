"""Concept families: items, ground-truth order, templates, and context generation.

A family YAML (families/<name>.yaml) has:
  name, topology (circle|torus|interval|unknown), origin (explicit|implicit|emergent|control),
  items: ordered list (ground-truth cyclic or linear order),
  positions: optional per-item angle in turns [0,1) when items are not evenly spaced
             (e.g. city longitude); torus families give a list of [a, b] pairs,
  line_control_for: optional name of the circle family it controls,
  templates: list of strings with exactly one {item} and optional {slot}s,
  fillers: optional per-family slot lists (merged over families/_fillers.yaml).

Every item gets the SAME list of filled prefixes, so centroids differ only by the item.
Text after {item} is kept for readability but never reaches the model (causal truncation).
"""

from __future__ import annotations

import itertools
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

FAMILY_DIR = Path(__file__).parent / "families"


@dataclass
class Family:
    name: str
    topology: str
    origin: str
    items: list[str]
    templates: list[str]
    fillers: dict[str, list[str]] = field(default_factory=dict)
    positions: list | None = None
    line_control_for: str | None = None
    notes: str = ""

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def is_cyclic(self) -> bool:
        return self.topology in ("circle", "torus")

    def angles(self) -> np.ndarray:
        """Ground-truth angle in radians per item (first circle factor for tori)."""
        if self.positions is None:
            return 2 * np.pi * np.arange(self.n_items) / self.n_items
        p = np.asarray(self.positions, dtype=float)
        return 2 * np.pi * (p[:, 0] if p.ndim == 2 else p)


@dataclass(frozen=True)
class Context:
    item: int          # index into family.items
    text: str
    span: tuple[int, int]  # char span of the item in text
    source: str        # "template" | "pile"
    template: int      # template index, -1 for mined contexts


def list_families() -> list[str]:
    return sorted(p.stem for p in FAMILY_DIR.glob("*.yaml") if not p.stem.startswith("_"))


def load_family(name: str) -> Family:
    with open(FAMILY_DIR / f"{name}.yaml", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    with open(FAMILY_DIR / "_fillers.yaml", encoding="utf-8") as f:
        shared = yaml.safe_load(f)
    fillers = {**shared, **(d.get("fillers") or {})}
    fam = Family(
        name=d["name"], topology=d["topology"], origin=d["origin"], items=list(d["items"]),
        templates=list(d.get("templates") or []), fillers=fillers, positions=d.get("positions"),
        line_control_for=d.get("line_control_for"), notes=d.get("notes", ""),
    )
    _validate(fam)
    return fam


def _slots(template: str) -> list[str]:
    return [f for _, f, _, _ in string.Formatter().parse(template) if f]


def _validate(fam: Family) -> None:
    if len(set(fam.items)) != len(fam.items):
        raise ValueError(f"{fam.name}: duplicate items")
    if fam.positions is not None and len(fam.positions) != len(fam.items):
        raise ValueError(f"{fam.name}: positions length mismatch")
    for t in fam.templates:
        s = _slots(t)
        if s.count("item") != 1:
            raise ValueError(f"{fam.name}: template needs exactly one {{item}}: {t!r}")
        if t.index("{item}") == 0:
            raise ValueError(f"{fam.name}: template must have a prefix before {{item}}: {t!r}")
        missing = [x for x in s if x != "item" and x not in fam.fillers]
        if missing:
            raise ValueError(f"{fam.name}: unknown slots {missing} in {t!r}")


def _fill_prefixes(fam: Family, template: str, k: int, rng: random.Random) -> list[dict]:
    """Up to k slot assignments for one template whose PREFIXES (text before the item, the only
    part the model sees) are distinct. Slots appearing only after the item are filled randomly."""
    pre, post = template.split("{item}")
    slots = list(dict.fromkeys(_slots(pre)))
    post_only = [s for s in dict.fromkeys(_slots(post)) if s not in slots]
    return [{**a, **{s: rng.choice(fam.fillers[s]) for s in post_only}}
            for a in _distinct_assignments(fam, slots, k, rng)]


def _distinct_assignments(fam: Family, slots: list[str], k: int, rng: random.Random) -> list[dict]:
    if not slots:
        return [{}]
    pools = [fam.fillers[s] for s in slots]
    total = int(np.prod([len(p) for p in pools]))
    if total <= 5 * k:
        combos = list(itertools.product(*pools))
        rng.shuffle(combos)
        combos = combos[:k]
    else:
        seen: set = set()
        combos = []
        while len(combos) < k:
            c = tuple(rng.choice(p) for p in pools)
            if c not in seen:
                seen.add(c)
                combos.append(c)
    return [dict(zip(slots, c)) for c in combos]


def template_contexts(fam: Family, n_per_item: int, seed: int = 0) -> list[Context]:
    """n_per_item contexts per item, spread evenly over templates, identical prefixes per item."""
    rng = random.Random(seed)
    # Generate a generous pool per template, then round-robin so templates are balanced
    # and slot-free templates (one prefix only) do not leave the item short.
    per_t = -(-n_per_item // len(fam.templates))  # ceil
    pools = [[(ti, a) for a in _fill_prefixes(fam, t, 4 * per_t, rng)] for ti, t in enumerate(fam.templates)]
    assignments: list = []
    depth = 0
    while len(assignments) < n_per_item and any(depth < len(p) for p in pools):
        assignments += [p[depth] for p in pools if depth < len(p)]
        depth += 1
    if len(assignments) < n_per_item:
        raise ValueError(f"{fam.name}: only {len(assignments)} distinct prefixes, need {n_per_item}")
    assignments = assignments[:n_per_item]
    out = []
    for item_idx, item in enumerate(fam.items):
        for ti, a in assignments:
            t = fam.templates[ti]
            pre, post = t.split("{item}")
            pre_s = pre.format(**a)
            pre_s = pre_s[:1].upper() + pre_s[1:]
            text = pre_s + item + post.format(**a)
            out.append(Context(item_idx, text, (len(pre_s), len(pre_s) + len(item)), "template", ti))
    return out


def item_regex(item: str) -> re.Pattern:
    """Case-sensitive whole-word match for Pile mining."""
    return re.compile(r"(?<![\w-])" + re.escape(item) + r"(?![\w-])")

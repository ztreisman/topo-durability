"""What are we measuring? Reliability of the per-concept durability and sense-topology measures (2026-11-18).
CPU. Run after the durability passes (scripts/token_durability.py) and the footprint pass.

  uv run python scripts/measure_reliability.py --fp-dir cfoot --workers 48
  ... --skip-topology        # durability part only (fast)
  ... --limit                # dry-run data

Durability (change in log p of the concept in its own contexts, variant minus base):
  split-half      contexts split in two fixed random halves; Spearman across concepts of the mean change on half A
                  vs half B, and the Spearman-Brown reliability of the full 224 (context-sampling noise only)
  across variants the same correlation between two variants, computed on DISJOINT halves (variant 1 on half A,
                  variant 2 on half B, averaged with the swap), so shared noise in the base log p cannot inflate it;
                  replicate pairs (two seeds of a fine-tune, three noise seeds) give the training-noise reliability
  composite       z-scored mean over variant sets, with its split-half reliability
Topology (base model):
  split-half      every measure (senses, bars, persistence, b1 with strict links) computed on each half
                  separately (gap statistic, as token_senses.py), correlated across concepts
  stable          per layer, k = the largest k <= 6 whose clustering reappears across the two quarters of the half
                  (prediction strength >= --ps-min, Tibshirani & Walther 2005; else 1), labels = k-means with that k;
                  the same measures on these stable clusters, again per half
Ceiling: the largest observable correlation between a topology measure and durability is
sqrt(reliability_topology x reliability_durability).
Outputs: data/neurons/<unit>/reliability_<fp>.parquet (per concept), reports .../reliability_<fp>.md / .png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.reeb import gap_k, pcs  # noqa: E402
from token_durability_analysis import topology  # noqa: E402

log = logging.getLogger("measure_reliability")
MEAS = ["senses", "bars", "mean_len", "total_pers", "b1"]
_G = {}


def sb(r: float) -> float:
    """Spearman-Brown: reliability of the full measure from the split-half correlation."""
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


def spear(x, y) -> float:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 3 else np.nan


# ------------------------------------------------------------------ stable clusters

def prediction_strength(train: np.ndarray, test: np.ndarray, k: int, seed: int = 0) -> float:
    """Tibshirani & Walther (2005): cluster train and test separately with k-means; for each test cluster, the share
    of its pairs that the train centroids also put together; PS = the minimum over test clusters."""
    from sklearn.cluster import KMeans
    if k == 1:
        return 1.0
    if len(train) < 2 * k or len(test) < 2 * k:
        return 0.0
    ctr = KMeans(k, n_init=3, random_state=seed).fit(train).cluster_centers_
    lt = KMeans(k, n_init=3, random_state=seed).fit_predict(test)
    via = np.argmin(((test[:, None, :] - ctr[None]) ** 2).sum(-1), 1)
    ps = []
    for c in range(k):
        idx = np.flatnonzero(lt == c)
        if len(idx) < 2:
            continue
        same = (via[idx][:, None] == via[idx][None, :])
        ps.append((same.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
    return float(min(ps)) if ps else 0.0


def stable_labels(Y: np.ndarray, kmax: int, ps_min: float, min_size: int, seed: int = 0):
    """Largest k whose clustering reappears across two random halves of Y (both directions); labels on all of Y."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    a, b = Y[perm[:len(Y) // 2]], Y[perm[len(Y) // 2:]]
    k_st = 1
    for k in range(2, kmax + 1):
        if min(prediction_strength(a, b, k, seed), prediction_strength(b, a, k, seed)) >= ps_min:
            k_st = k
    lab = np.zeros(len(Y), int) if k_st == 1 else KMeans(k_st, n_init=3, random_state=seed).fit_predict(Y)
    for c in np.unique(lab):
        if (lab == c).sum() < min_size:
            lab[lab == c] = -1
    return k_st, lab


# ------------------------------------------------------------------ per concept

def _init(mu, sd):
    _G["mu"], _G["sd"] = mu, sd


def one_concept(path: str, layers: list[int], lay_index: list[int], mid: list[int], o: dict):
    X = np.load(path, mmap_mode="r")[:o["n_ctx"]]
    n = len(X)
    perm = np.random.default_rng(o["seed"]).permutation(n)
    halves = {"A": np.sort(perm[:n // 2]), "B": np.sort(perm[n // 2:])}
    lab = {(h, kind): np.full((len(layers), len(halves[h])), -1, int) for h in halves for kind in ("gap", "stable")}
    kst = {h: [] for h in halves}
    for i, l in enumerate(layers):
        Z = (X[:, lay_index.index(l)].astype(np.float32) - _G["mu"][l]) / _G["sd"][l]
        P = pcs(Z, o["dims"])                                   # one projection for both halves
        for h, idx in halves.items():
            Y = P[idx]
            _, g = gap_k(Y, kmax=o["kmax"], n_ref=o["n_ref"], min_size=o["min_size"], ref_inits=o["ref_inits"])
            for c, gi in enumerate(g):
                lab[(h, "gap")][i, gi] = c
            k, sl = stable_labels(Y, o["kmax"], o["ps_min"], o["min_size"], o["seed"] + i)
            lab[(h, "stable")][i] = sl
            if l in mid:
                kst[h].append(k)
    row = {}
    for (h, kind), L in lab.items():
        t = topology(L, layers, mid, o["w_strict"])
        for m in MEAS:
            row[f"{kind}_{m}_{h}"] = t[m]
    for h in halves:
        row[f"stable_k_mid_{h}"] = float(np.mean(kst[h]))
    return row


# ------------------------------------------------------------------ main

def durability_part(ddir: Path, toks: list[int], n_ctx: int, seed: int):
    vd = sorted(p for p in ddir.iterdir() if (p / "logp.npz.done").exists() and not p.name.endswith("_dry"))
    lp = {p.name: np.load(p / "logp.npz") for p in vd}
    variants = [v for v in lp if v != "base"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ctx)
    A, B = np.sort(perm[:n_ctx // 2]), np.sort(perm[n_ctx // 2:])
    rows = []
    for t in toks:
        k = f"T{t}"
        if k not in lp["base"].files or len(lp["base"][k]) < n_ctx:
            continue
        base = lp["base"][k][:n_ctx]
        r = {"token_id": t, "base_logp_A": float(base[A].mean()), "base_logp_B": float(base[B].mean())}
        for v in variants:
            d = lp[v][k][:n_ctx] - base
            r[f"d_{v}_A"], r[f"d_{v}_B"], r[f"d_{v}"] = float(d[A].mean()), float(d[B].mean()), float(d.mean())
        rows.append(r)
    return pd.DataFrame(rows), variants


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="1.4b_s0_step143000")
    ap.add_argument("--fp-dir", default="cfoot")
    ap.add_argument("--n-ctx", type=int, default=224)
    ap.add_argument("--dims", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--ref-inits", type=int, default=3)
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--w-strict", type=float, default=0.7)
    ap.add_argument("--ps-min", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-topology", action="store_true")
    ap.add_argument("--max-concepts", type=int, default=None)
    ap.add_argument("--limit", action="store_true")
    a = ap.parse_args()
    tag = "neurons_dryrun" if a.limit else "neurons"
    root = runs.DATA_DIR / tag / a.unit
    rep = runs.REPO_ROOT / "reports" / tag / a.unit
    rep.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(root / f"reliability_{a.fp_dir}.log")
    if a.limit:
        a.min_size = 2
    fp = root / a.fp_dir
    meta = json.loads((fp / "footprint.json").read_text())
    toks = [int(t) for t, v in meta["tokens"].items() if v["n"] >= a.n_ctx and (fp / f"T{t}.npy").exists()]
    if a.max_concepts:
        toks = toks[:a.max_concepts]
    ddir = root / ("durab" if a.fp_dir == "footprint" else f"durab_{a.fp_dir}")
    D, variants = durability_part(ddir, toks, a.n_ctx, a.seed)
    log.info("%d concepts with %d contexts; variants %s", len(D), a.n_ctx, variants)
    T = pd.DataFrame({"token_id": D.token_id})
    tpq = root / f"reliability_{a.fp_dir}_topology.parquet"
    if not a.skip_topology:
        lay_index = meta["layers"]
        layers = [l for l in lay_index if (root / f"L{l:02d}.npz").exists()]
        mid = [l for l in layers if len(layers) // 3 <= layers.index(l) < 2 * len(layers) // 3]
        mu, sd = {}, {}
        for l in layers:
            z = np.load(root / f"L{l:02d}.npz")
            s = z["sd"].astype(np.float32)
            mu[l], sd[l] = z["mean"].astype(np.float32), np.maximum(s, np.quantile(s, 0.1))
        o = {k: getattr(a, k) for k in ("n_ctx", "dims", "kmax", "n_ref", "ref_inits", "min_size", "w_strict", "ps_min", "seed")}
        rows = []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(mu, sd)) as ex:
            futs = {ex.submit(one_concept, str(fp / f"T{t}.npy"), layers, lay_index, mid, o): t for t in D.token_id}
            for i, (f, t) in enumerate(futs.items()):
                rows.append({"token_id": t, **f.result()})
                if (i + 1) % 50 == 0:
                    log.info("topology: %d / %d concepts", i + 1, len(futs))
        T = pd.DataFrame(rows)
        T.to_parquet(tpq, index=False)
    elif tpq.exists():
        T = pd.read_parquet(tpq)
    df = D.merge(T, on="token_id", how="left")
    cls = {int(t): v["class"] for t, v in meta["tokens"].items()}
    df["class"] = df.token_id.map(cls)
    df.to_parquet(root / f"reliability_{a.fp_dir}.parquet", index=False)
    report(df, variants, rep, a)
    return 0


def report(df, variants, rep, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = [f"# Measurement reliability ({a.fp_dir}): {a.unit}", "",
         f"{len(df)} concepts, {a.n_ctx} contexts each, split into two fixed random halves of {a.n_ctx // 2}. "
         "Reliability = Spearman-Brown corrected split-half correlation across concepts (1 = the measure ranks concepts "
         "the same way on independent contexts; 0 = pure noise).", ""]
    # durability: split-half per variant
    L += ["## Durability: context-sampling reliability (split-half)", "",
          "| variant | mean d log p | split-half r | reliability |", "|---|---|---|---|"]
    rel_d = {}
    for v in variants:
        r = spear(df[f"d_{v}_A"], df[f"d_{v}_B"])
        rel_d[v] = sb(r)
        L.append(f"| {v} | {df[f'd_{v}'].mean():+.3f} | {r:+.3f} | {rel_d[v]:+.3f} |")
    L.append(f"| base log p (sanity) | | {spear(df.base_logp_A, df.base_logp_B):+.3f} | {sb(spear(df.base_logp_A, df.base_logp_B)):+.3f} |")
    # cross-variant on disjoint halves
    L += ["", "## Durability: is it the same trait across perturbations? (disjoint halves; replicate pairs first)", "",
          "Correlation of variant 1 on half A with variant 2 on half B (and the swap, averaged). Shared base noise "
          "cannot inflate it. Divided by sqrt(rel1 x rel2) it is the correlation of the TRUE per-concept damage.", "",
          "| variant 1 | variant 2 | r (disjoint) | disattenuated |", "|---|---|---|---|"]

    def rep_key(p):
        a_, b_ = p
        same = a_.split("-")[1].split("_s")[0] == b_.split("-")[1].split("_s")[0] if a_.startswith("lora") and b_.startswith("lora") else False
        return (0 if (same or (a_.startswith("noise") and b_.startswith("noise"))) else 1, p)

    cross = []
    for v1, v2 in sorted(combinations(variants, 2), key=rep_key):
        r = 0.5 * (spear(df[f"d_{v1}_A"], df[f"d_{v2}_B"]) + spear(df[f"d_{v1}_B"], df[f"d_{v2}_A"]))
        den = np.sqrt(rel_d[v1] * rel_d[v2]) if rel_d[v1] > 0 and rel_d[v2] > 0 else np.nan
        cross.append((v1, v2, r, r / den if den else np.nan))
        L.append(f"| {v1} | {v2} | {r:+.3f} | {r / den if den else float('nan'):+.3f} |")
    # composite
    sets = {"LoRA 20M (all seeds)": [v for v in variants if v.startswith("lora") and v.endswith("20M")],
            "all LoRA milestones": [v for v in variants if v.startswith("lora")],
            "all variants": variants}
    L += ["", "## Composite durability (mean of z-scored changes)", "", "| set | variants | split-half r | reliability |", "|---|---|---|---|"]
    comp_rel = {}
    for name, vs in sets.items():
        if len(vs) < 2:
            continue
        z = lambda h: np.mean([(df[f"d_{v}_{h}"] - df[f"d_{v}_{h}"].mean()) / df[f"d_{v}_{h}"].std() for v in vs], 0)
        df[f"comp_{name}_A"], df[f"comp_{name}_B"] = z("A"), z("B")
        r = spear(df[f"comp_{name}_A"], df[f"comp_{name}_B"])
        comp_rel[name] = sb(r)
        L.append(f"| {name} | {len(vs)} | {r:+.3f} | {comp_rel[name]:+.3f} |")
    # topology
    rel_t = {}
    if "gap_senses_A" in df:
        L += ["", "## Topology: context-sampling reliability (split-half; base model)", "",
              "gap = clusters as in token_senses.py (gap statistic); stable = only splits that reappear across the "
              f"quarters of a half (prediction strength >= {a.ps_min}).", "",
              "| measure | gap: mean (A) | gap: reliability | stable: mean (A) | stable: reliability |", "|---|---|---|---|---|"]
        for m in MEAS:
            rg, rs = sb(spear(df[f"gap_{m}_A"], df[f"gap_{m}_B"])), sb(spear(df[f"stable_{m}_A"], df[f"stable_{m}_B"]))
            rel_t[("gap", m)], rel_t[("stable", m)] = rg, rs
            L.append(f"| {m} | {df[f'gap_{m}_A'].mean():.2f} | {rg:+.3f} | {df[f'stable_{m}_A'].mean():.2f} | {rs:+.3f} |")
        rk = sb(spear(df.stable_k_mid_A, df.stable_k_mid_B))
        L.append(f"| stable k (middle layers) | | | {df.stable_k_mid_A.mean():.2f} | {rk:+.3f} |")
        # ceiling
        best_d = max(comp_rel.items(), key=lambda x: x[1]) if comp_rel else max(rel_d.items(), key=lambda x: x[1])
        L += ["## Ceiling: largest observable topology-durability correlation = sqrt(rel_topology x rel_durability)", "",
              f"Durability: {best_d[0]} (reliability {best_d[1]:+.3f}).", "",
              "| measure | gap | stable |", "|---|---|---|"]
        for m in MEAS:
            c = lambda kind: np.sqrt(max(rel_t[(kind, m)], 0) * max(best_d[1], 0))
            L.append(f"| {m} | {c('gap'):.3f} | {c('stable'):.3f} |")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    vs = list(rel_d)
    ax[0].barh(vs, [rel_d[v] for v in vs], color="#2a6f97")
    ax[0].set_xlabel("durability reliability (split-half, SB)"); ax[0].set_xlim(-0.2, 1)
    if rel_t:
        ms = MEAS
        y = np.arange(len(ms))
        ax[1].barh(y - 0.2, [rel_t[("gap", m)] for m in ms], 0.4, color="#c9772f", label="gap")
        ax[1].barh(y + 0.2, [rel_t[("stable", m)] for m in ms], 0.4, color="#2b9348", label="stable")
        ax[1].set_yticks(y, ms); ax[1].set_xlabel("topology reliability (split-half, SB)"); ax[1].set_xlim(-0.2, 1)
        ax[1].legend(frameon=False)
    for s in ax:
        s.axvline(0, color="#999", lw=0.8); s.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(rep / f"reliability_{a.fp_dir}.png", dpi=130)
    L += ["", f"![reliability](reliability_{a.fp_dir}.png)", ""]
    runs.atomic_write_text(rep / f"reliability_{a.fp_dir}.md", "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())

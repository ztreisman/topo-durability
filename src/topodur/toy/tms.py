"""Toy Model of Superposition (Elhage et al. 2022) with correlated sparse features (Phase 4).

Model, batched over S independent seeds:  h = W x,  x_hat = ReLU(W^T h + b),  W: (S, m, n), b: (S, n).
Loss: mean over samples of sum_i (x_i - x_hat_i)^2 (uniform importance, so structure is the only
difference between features).

Data: features fire through GROUPS. Each group fires independently per sample with its own rate; a
feature is active if any of its groups fires; active features take independent U[0,1] values.
Structures (n features, base rate q; every feature's marginal firing rate is about 2q in all of them):
  independent  singletons at rate 2q
  ring         edges (i, i+1 mod n) at rate q
  path         ring minus the closing edge (n-1, 0); endpoints get a singleton at rate q
  clique       disjoint blocks of `clique_size` features firing together at rate 2q
  ring_rare    ring whose closing edge fires at rate rho*q (endpoints topped up by singletons)
  periodic     place cells: an event (rate 2q * n / bump_width) draws a position on the circle and
               activates the features whose preferred angle is within the bump (cosine tuning)
  mixed        one model holding a ring, a path, and cliques side by side (refined-LLC comparisons)
Port note: the localized-SGLD LLC estimator follows ztreisman/arc-llc (tms.sgld_llc), here with a
minibatch loss and an optional parameter mask (weight-refined LLC, Wang et al. 2024).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

STRUCTURES = ["independent", "ring", "path", "clique", "ring_rare", "periodic", "mixed"]


@dataclass
class ToyConfig:
    structure: str
    n: int = 20
    m: int = 5
    q: float = 0.02
    seeds: int = 20
    steps: int = 10_000
    batch: int = 1024
    lr: float = 3e-3
    clique_size: int = 4
    rho: float = 0.1          # ring_rare: closing-edge rate relative to the others
    bump_width: int = 3       # periodic: features active per event
    seed0: int = 0


@dataclass
class Groups:
    """Group membership (G, n) and per-group firing rates (G,), plus per-feature metadata."""
    member: torch.Tensor
    rate: torch.Tensor
    sub: dict = field(default_factory=dict)        # substructure name -> feature indices
    order: dict = field(default_factory=dict)      # substructure name -> "cycle" | "line" | None
    periodic: bool = False


def _edges(idx, closed: bool):
    k = len(idx)
    e = [(idx[i], idx[i + 1]) for i in range(k - 1)]
    return e + [(idx[-1], idx[0])] if closed else e


def build_groups(cfg: ToyConfig) -> Groups:
    n, q = cfg.n, cfg.q
    rows, rates = [], []

    def add(feats, r):
        v = np.zeros(n)
        v[list(feats)] = 1
        rows.append(v)
        rates.append(r)

    sub, order = {}, {}
    idx = list(range(n))
    s = cfg.structure
    if s == "independent":
        for i in idx:
            add([i], 2 * q)
        sub["all"], order["all"] = idx, None
    elif s in ("ring", "path", "ring_rare"):
        for e in _edges(idx, closed=True):
            closing = e == (idx[-1], idx[0])
            if closing and s == "path":
                continue
            add(e, cfg.rho * q if (closing and s == "ring_rare") else q)
        if s in ("path", "ring_rare"):  # top up the endpoints' marginal rate
            top = q if s == "path" else (1 - cfg.rho) * q
            add([idx[0]], top)
            add([idx[-1]], top)
        sub["all"], order["all"] = idx, ("line" if s == "path" else "cycle")
    elif s == "clique":
        k = cfg.clique_size
        for b in range(0, n, k):
            add(idx[b:b + k], 2 * q)
        sub["all"], order["all"] = idx, None
    elif s == "periodic":
        sub["all"], order["all"] = idx, "cycle"
        return Groups(torch.zeros(0, n), torch.zeros(0), sub, order, periodic=True)
    elif s == "mixed":
        k = int(round(0.3 * n))
        ring, path, cl = idx[:k], idx[k:2 * k], idx[2 * k:]
        for e in _edges(ring, True):
            add(e, q)
        for e in _edges(path, False):
            add(e, q)
        add([path[0]], q)
        add([path[-1]], q)
        for b in range(0, len(cl), cfg.clique_size):
            add(cl[b:b + cfg.clique_size], 2 * q)
        sub.update(ring=ring, path=path, clique=cl)
        order.update(ring="cycle", path="line", clique=None)
    else:
        raise ValueError(s)
    return Groups(torch.tensor(np.array(rows), dtype=torch.float32), torch.tensor(rates, dtype=torch.float32),
                  sub, order)


def sample(cfg: ToyConfig, groups: Groups, S: int, B: int, gen: torch.Generator) -> torch.Tensor:
    """(S, B, n) batch of inputs."""
    n = cfg.n
    if groups.periodic:
        w = cfg.bump_width
        p_event = min(1.0, 2 * cfg.q * n / w)
        ev = torch.rand(S, B, 1, generator=gen) < p_event
        theta = torch.rand(S, B, 1, generator=gen) * 2 * math.pi
        pref = torch.arange(n) * 2 * math.pi / n
        d = torch.remainder(theta - pref + math.pi, 2 * math.pi) - math.pi          # signed angular distance
        half = w * math.pi / n                                                      # bump half-width
        x = torch.clamp(torch.cos(d * math.pi / (2 * half)), min=0) * (d.abs() < half)
        return x * ev
    fire = (torch.rand(S, B, len(groups.rate), generator=gen) < groups.rate).float()
    active = (fire @ groups.member) > 0
    return active * torch.rand(S, B, n, generator=gen)


def forward(W: torch.Tensor, b: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """W (S, m, n), b (S, n), X (S, B, n) -> x_hat (S, B, n)."""
    return torch.relu((X @ W.transpose(1, 2)) @ W + b[:, None, :])


def per_feature_loss(W, b, X) -> torch.Tensor:
    """(S, n): mean squared reconstruction error per feature."""
    return ((X - forward(W, b, X)) ** 2).mean(1)


def init_params(cfg: ToyConfig, gen: torch.Generator):
    W = torch.randn(cfg.seeds, cfg.m, cfg.n, generator=gen) * (1 / math.sqrt(cfg.m))
    b = torch.zeros(cfg.seeds, cfg.n)
    return W, b


def train(cfg: ToyConfig, W=None, b=None, groups: Groups | None = None, steps: int | None = None,
          gen: torch.Generator | None = None, record_every: int = 0):
    """Adam on fresh batches (infinite data). Returns W, b, and optional snapshots [(step, W, b)]."""
    gen = gen or torch.Generator().manual_seed(cfg.seed0)
    groups = groups or build_groups(cfg)
    if W is None:
        W, b = init_params(cfg, gen)
    W = W.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    steps = steps or cfg.steps
    opt = torch.optim.Adam([W, b], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    snaps = []
    for t in range(steps):
        X = sample(cfg, groups, cfg.seeds, cfg.batch, gen)
        loss = per_feature_loss(W, b, X).sum(1)          # (S,)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        sched.step()
        if record_every and (t % record_every == 0 or t == steps - 1):
            snaps.append((t, W.detach().clone(), b.detach().clone()))
    return W.detach(), b.detach(), snaps


def sgld_llc(W, b, cfg: ToyConfig, groups: Groups, mask_feats=None, n_data: int = 20_000, steps: int = 1500,
             eps: float = 1e-4, gamma: float = 100.0, chains: int = 2, batch: int = 1024, burn: int = 300,
             seed: int = 0) -> np.ndarray:
    """Localized SGLD LLC estimate per seed, lambda = n*beta*(E[L_n(w)] - L_n(w*)), beta = 1/log n.
    mask_feats: if given, only W[:, :, feats] and b[:, feats] move (weight-refined LLC)."""
    gen = torch.Generator().manual_seed(seed + 7)
    S = W.shape[0]
    data = sample(cfg, groups, S, n_data, gen)                     # fixed dataset per seed
    nb = n_data / math.log(n_data)
    W0 = W.repeat_interleave(chains, 0)
    b0 = b.repeat_interleave(chains, 0)
    data = data.repeat_interleave(chains, 0)
    mW = torch.ones_like(W0)
    mb = torch.ones_like(b0)
    if mask_feats is not None:
        mW.zero_()
        mb.zero_()
        mW[:, :, mask_feats] = 1
        mb[:, mask_feats] = 1

    def full_loss(Wc, bc):
        with torch.no_grad():
            return torch.stack([per_feature_loss(Wc, bc, data[:, i:i + 4096]).sum(1)
                                for i in range(0, n_data, 4096)]).mean(0)

    burn = min(burn, steps // 3)
    L0 = full_loss(W0, b0)
    Wc, bc = W0.clone(), b0.clone()
    acc = torch.zeros(S * chains)
    cnt = 0
    for t in range(steps):
        idx = torch.randint(0, n_data, (batch,), generator=gen)
        Wc.requires_grad_(True)
        bc.requires_grad_(True)
        L = per_feature_loss(Wc, bc, data[:, idx]).sum(1)
        gW, gb = torch.autograd.grad(L.sum(), [Wc, bc])
        with torch.no_grad():
            Wc = Wc - 0.5 * eps * (nb * gW + gamma * (Wc - W0)) * mW + math.sqrt(eps) * torch.randn(
                Wc.shape, generator=gen) * mW
            bc = bc - 0.5 * eps * (nb * gb + gamma * (bc - b0)) * mb + math.sqrt(eps) * torch.randn(
                bc.shape, generator=gen) * mb
            if t >= burn and t % 10 == 0:
                acc += full_loss(Wc, bc)
                cnt += 1
    lam = nb * (acc / cnt - L0)
    return lam.reshape(S, chains).mean(1).numpy()

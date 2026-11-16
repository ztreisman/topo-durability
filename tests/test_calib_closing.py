import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("neuron_mine", ROOT / "scripts" / "neuron_mine.py")
nm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nm)


def _tuning(kind, n_items=12, n_neurons=128, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n_items)
    pref = rng.uniform(0, n_items, n_neurons)
    if kind == "random":
        return rng.standard_normal((n_neurons, n_items))
    d = np.abs(x[None, :] - pref[:, None])
    if kind == "ring":
        d = np.minimum(d, n_items - d)
    return np.exp(-(d / 2.0) ** 2) + noise * rng.standard_normal((n_neurons, n_items))


class _Fam:
    def __init__(self, n, topo):
        self.items = [f"i{k}" for k in range(n)]
        self.topology = topo


def _call(kind, n, seed=0, noise=0.3):
    T = _tuning(kind, n_items=n, seed=seed, noise=noise)
    z = {"calib_mu": T, "sd": np.ones(len(T)), "calib_n": np.full(n, 100)}
    calib = {"ids": list(range(n)), "families": {"fam": [(k, k) for k in range(n)]}}
    nm.load_family = lambda name: _Fam(n, "circle" if kind == "ring" else "interval")
    return nm.calibrate(z, calib, 128, 300, seed).call.iloc[0]


def test_calibration_calls():
    calls = {(k, n, s): _call(k, n, seed=s) for k in ("ring", "line") for n in (7, 12, 24) for s in range(3)}
    assert all(c in (k, "-") for (k, _, _), c in calls.items())          # never the wrong shape
    assert sum(c == k for (k, _, _), c in calls.items()) >= 14           # mostly decided at moderate noise
    assert all(_call("random", 12, seed=s) == "-" for s in range(6))   # no shape -> no call

"""Phase 0 step 1-2: verify the environment.

On lambda01: expects 8 GPUs, compute capability 7.5, a working fp16 matmul on every GPU,
sm_75 kernels in the torch build, CUDA runtime <= 12.8, and the TDA stack importable.
Exit code 0 only if every check passes. Run:  uv run python scripts/check_env.py [--expect-gpus 8]
"""

from __future__ import annotations

import argparse
import importlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-gpus", type=int, default=None, help="fail unless exactly this many GPUs")
    ap.add_argument("--cpu-ok", action="store_true", help="laptop mode: skip GPU checks")
    a = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    print("python", sys.version.split()[0])
    check("python 3.12", sys.version_info[:2] == (3, 12))

    import torch

    print("torch", torch.__version__, "cuda runtime", torch.version.cuda)
    for mod in ["transformers", "datasets", "ripser", "persim", "gudhi", "pandas", "pyarrow", "sklearn"]:
        try:
            m = importlib.import_module(mod)
            check(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:
            check(f"import {mod}", False, repr(e))
    try:
        importlib.import_module("dreimac")
        print("[info] dreimac available (optional cross-check)")
    except Exception:
        print("[info] dreimac not installed (optional; tda.py has its own circular coordinates)")

    # tiny persistence sanity check
    import numpy as np
    from ripser import ripser

    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    dg = ripser(np.c_[np.cos(t), np.sin(t)], maxdim=1)["dgms"][1]
    check("ripser finds one H1 class on a circle", len(dg) >= 1 and (dg[:, 1] - dg[:, 0]).max() > 0.5)

    if a.cpu_ok:
        print("CPU mode: skipping GPU checks")
        return 0 if ok else 1

    check("cuda available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return 1
    if torch.version.cuda:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        check("cuda runtime <= 12.8", (major, minor) <= (12, 8), torch.version.cuda)
    archs = torch.cuda.get_arch_list()
    check("torch build has sm_75 kernels", "sm_75" in archs, str(archs))
    n = torch.cuda.device_count()
    check("gpu count", a.expect_gpus is None or n == a.expect_gpus, f"{n} (expected {a.expect_gpus})")
    for i in range(n):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        try:
            x = torch.randn(2048, 2048, device=f"cuda:{i}", dtype=torch.float16)
            y = (x @ x).float()
            fin = bool(torch.isfinite(y).all())
            ref = (x.float() @ x.float())
            err = float((y - ref).abs().max() / ref.abs().max())
            check(f"gpu{i} fp16 matmul", fin and err < 1e-2, f"{name} cc{cap[0]}.{cap[1]} relerr={err:.1e}")
        except Exception as e:
            check(f"gpu{i} fp16 matmul", False, repr(e))
        if cap != (7, 5):
            print(f"[warn] gpu{i} capability {cap} (expected 7.5 on lambda01)")
        free, total = torch.cuda.mem_get_info(i)
        print(f"       gpu{i} memory free {free/1e9:.1f} / {total/1e9:.1f} GB")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

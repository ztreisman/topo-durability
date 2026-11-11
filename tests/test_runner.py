"""Job runner end-to-end on CPU-only fake jobs. POSIX only (runs on lambda01, skipped on Windows)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="runner is POSIX-only")
RUNNER = Path(__file__).resolve().parents[1] / "jobs" / "runner.py"


def run(tmp_path, *args):
    env = dict(os.environ, TOPODUR_JOBS_DIR=str(tmp_path))
    return subprocess.run([sys.executable, str(RUNNER), *args], env=env, capture_output=True, text=True, timeout=25)


def test_runner_dependencies_retries_and_failures(tmp_path):
    q = tmp_path / "queue.jsonl"
    marker = tmp_path / "flaky_once"
    jobs = [
        {"id": "a", "gpus": 0, "cmd": f"echo a > {tmp_path}/a.out"},
        {"id": "b", "gpus": 0, "after": ["a"], "cmd": f"test -f {tmp_path}/a.out && echo b > {tmp_path}/b.out"},
        # fails the first time, succeeds on retry
        {"id": "flaky", "gpus": 0, "retries": 1, "cmd": f"if [ -f {marker} ]; then exit 0; else touch {marker}; exit 3; fi"},
        {"id": "bad", "gpus": 0, "retries": 0, "cmd": "exit 7"},
        {"id": "blocked", "gpus": 0, "after": ["bad"], "cmd": "echo never"},
    ]
    q.write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    # 'blocked' never becomes runnable, so bound the run with a timeout instead of --exit-when-empty
    try:
        run(tmp_path, "run", "--queue", str(q), "--gpus", "", "--poll", "1", "--cpu-slots", "4")
    except subprocess.TimeoutExpired:
        pass
    st = tmp_path / "state"
    assert (st / "a.done").exists() and (st / "b.done").exists()
    assert (tmp_path / "b.out").read_text().strip() == "b"
    assert (st / "flaky.done").exists() and len(list((tmp_path / "logs").glob("flaky.*.log"))) == 2
    assert (st / "bad.failed").exists()
    assert not (st / "blocked.done").exists()
    out = run(tmp_path, "status", "--queue", str(q)).stdout
    assert "FAILED" in out and "pending" in out

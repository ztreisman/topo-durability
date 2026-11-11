#!/usr/bin/env python3
"""Minimal GPU job queue for lambda01. Stdlib only, so it runs under any python3.

Queue file (JSONL, append-only; re-read every poll, so jobs can be added while it runs):
  {"id": "extract-1.4b-step143000", "cmd": "uv run python scripts/extract.py ...",
   "gpus": 1, "after": ["other-id"], "retries": 2, "cpu": false}

  gpus     GPUs the job needs (CUDA_VISIBLE_DEVICES is set to that many free GPUs); 0 for CPU jobs
  after    ids that must be done first
  retries  extra attempts after a non-zero exit (default 2)
  cpu      CPU-only job; at most --cpu-slots run at once

State lives in jobs/state/<id>.{running,exit,done,failed} and logs in jobs/logs/<id>.<attempt>.log.
Jobs are started in their own session and write their exit code to a file, so they keep running
if the runner dies; a restarted runner re-adopts them. Commands must themselves be resumable.

  python3 jobs/runner.py run   [--queue jobs/queue.jsonl] [--gpus 0,1,2,3,4,5,6,7]
  python3 jobs/runner.py status
  python3 jobs/runner.py retry <id>       # clear failed state so it runs again
  touch jobs/STOP                          # stop launching new jobs (running ones continue)
"""

from __future__ import annotations

import argparse
import json
import os

if os.name != "posix":
    raise SystemExit("runner.py is POSIX-only (lambda01). On Windows os.kill(pid, 0) terminates the process.")
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
JOBS_DIR = Path(os.environ.get("TOPODUR_JOBS_DIR", HERE))  # overridable for tests
STATE = JOBS_DIR / "state"
LOGS = JOBS_DIR / "logs"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def read_queue(path: Path) -> list[dict]:
    jobs, seen = [], set()
    if not path.exists():
        return jobs
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"queue line {n} is not valid JSON, skipped: {e}")
            continue
        if j["id"] in seen:
            continue  # first definition wins
        seen.add(j["id"])
        j.setdefault("gpus", 1)
        j.setdefault("after", [])
        j.setdefault("retries", 2)
        j.setdefault("cpu", j["gpus"] == 0)
        jobs.append(j)
    return jobs


def st(job_id: str, kind: str) -> Path:
    return STATE / f"{job_id}.{kind}"


CHILDREN: dict[int, subprocess.Popen] = {}  # jobs launched by THIS runner process


def pid_alive(pid: int) -> bool:
    # Our own children must be polled: an exited, unwaited child is a zombie, and os.kill(pid, 0)
    # still succeeds on zombies. Jobs adopted from a previous runner are not our children.
    if pid in CHILDREN:
        return CHILDREN[pid].poll() is None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def gpu_busy_external(threshold_mb: int) -> set[int]:
    """GPUs with more than threshold_mb used (e.g. by other users' processes)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(i) for i, m in (l.split(",") for l in out.strip().splitlines()) if int(m) > threshold_mb}
    except Exception as e:
        log(f"nvidia-smi failed ({e}); assuming no external load")
        return set()


def attempts(job_id: str) -> int:
    return len(list(LOGS.glob(f"{job_id}.*.log")))


def launch(job: dict, gpus: list[int]) -> None:
    n = attempts(job["id"]) + 1
    logf = LOGS / f"{job['id']}.{n}.log"
    exitf = st(job["id"], "exit")
    exitf.unlink(missing_ok=True)
    # The wrapper records the exit code even if the runner is gone by then. The command runs in a
    # subshell so an `exit` inside it cannot skip the record, and the file is renamed into place
    # so the runner never reads a half-written code.
    q = shlex.quote(str(exitf))
    wrapped = f"(\n{job['cmd']}\n); echo $? > {q}.tmp && mv {q}.tmp {q}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)) if gpus else "",
               PYTHONUNBUFFERED="1")
    with open(logf, "a") as f:
        f.write(f"# {time.ctime()} attempt {n} gpus={gpus}\n# {job['cmd']}\n")
        p = subprocess.Popen(["bash", "-c", wrapped], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
    CHILDREN[p.pid] = p
    st(job["id"], "running").write_text(json.dumps({"pid": p.pid, "gpus": gpus, "start": time.time(),
                                                    "attempt": n}))
    log(f"launched {job['id']} attempt {n} on gpus {gpus} (pid {p.pid})")


def reap(job: dict) -> str | None:
    """Return 'done' / 'failed' / 'retry' if a running job finished, else None."""
    run = st(job["id"], "running")
    info = json.loads(run.read_text())
    exitf = st(job["id"], "exit")
    if exitf.exists():
        code = int(exitf.read_text().strip() or 1)
    elif pid_alive(info["pid"]):
        return None
    elif exitf.exists():  # finished between the two checks
        code = int(exitf.read_text().strip() or 1)
    else:
        code = -1  # died without writing an exit code (killed, reboot)
    CHILDREN.pop(info["pid"], None)
    run.unlink()
    exitf.unlink(missing_ok=True)
    if code == 0:
        st(job["id"], "done").write_text(json.dumps({"end": time.time(), "attempt": info["attempt"]}))
        log(f"done {job['id']} ({(time.time() - info['start']) / 3600:.2f} h)")
        return "done"
    if info["attempt"] <= job["retries"]:
        log(f"{job['id']} exited {code}; will retry")
        return "retry"
    st(job["id"], "failed").write_text(json.dumps({"code": code, "end": time.time()}))
    log(f"FAILED {job['id']} (exit {code}); see {LOGS}/{job['id']}.*.log")
    return "failed"


def cmd_run(a) -> None:
    STATE.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    if a.gpus is None:
        all_gpus = _detect_gpus()
    else:
        all_gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
    log(f"runner started; gpus={all_gpus} queue={a.queue}")
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    while not stop["flag"]:
        jobs = read_queue(Path(a.queue))
        for j in jobs:  # collect finished jobs BEFORE computing done/failed, or they get relaunched
            if st(j["id"], "running").exists():
                reap(j)
        done = {j["id"] for j in jobs if st(j["id"], "done").exists()}
        failed = {j["id"] for j in jobs if st(j["id"], "failed").exists()}
        running = [j for j in jobs if st(j["id"], "running").exists()]
        used = {g for j in running for g in json.loads(st(j["id"], "running").read_text())["gpus"]}
        cpu_running = sum(1 for j in running if j["cpu"])
        external = gpu_busy_external(a.busy_mb) - used if all_gpus else set()
        free = [g for g in all_gpus if g not in used and g not in external]
        pending = [j for j in jobs if j["id"] not in done | failed and not st(j["id"], "running").exists()]
        if (JOBS_DIR / "STOP").exists():
            if pending:
                log(f"STOP file present; {len(pending)} pending jobs held")
        else:
            for j in pending:
                if any(d in failed for d in j["after"]):
                    continue  # blocked by a failed dependency (shown in status)
                if not all(d in done for d in j["after"]):
                    continue
                if j["cpu"]:
                    if cpu_running < a.cpu_slots:
                        launch(j, [])
                        cpu_running += 1
                elif len(free) >= j["gpus"]:
                    launch(j, free[: j["gpus"]])
                    free = free[j["gpus"]:]
        if not pending and not running and a.exit_when_empty:
            log("queue empty; exiting")
            return
        time.sleep(a.poll)


def _detect_gpus() -> list[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def cmd_status(a) -> None:
    jobs = read_queue(Path(a.queue))
    counts: dict[str, int] = {}
    for j in jobs:
        if st(j["id"], "done").exists():
            s = "done"
        elif st(j["id"], "failed").exists():
            s = "FAILED"
        elif st(j["id"], "running").exists():
            info = json.loads(st(j["id"], "running").read_text())
            s = f"running gpus={info['gpus']} {(time.time() - info['start']) / 3600:.1f}h"
        else:
            s = "pending"
        counts[s.split()[0]] = counts.get(s.split()[0], 0) + 1
        if not a.summary:
            print(f"{s:<32} {j['id']}  (attempts {attempts(j['id'])})")
    print("summary:", counts)


def cmd_retry(a) -> None:
    for kind in ("failed", "done"):
        st(a.id, kind).unlink(missing_ok=True)
    print(f"cleared state for {a.id}; it will run on the next poll")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--queue", default=str(HERE / "queue.jsonl"))
    r.add_argument("--gpus", default=None, help="comma list; default: all from nvidia-smi")
    r.add_argument("--poll", type=int, default=30)
    r.add_argument("--busy-mb", type=int, default=1500, help="treat GPUs using more than this as busy")
    r.add_argument("--cpu-slots", type=int, default=2)
    r.add_argument("--exit-when-empty", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--queue", default=str(HERE / "queue.jsonl"))
    s.add_argument("--summary", action="store_true")
    t = sub.add_parser("retry")
    t.add_argument("id")
    a = ap.parse_args()
    {"run": cmd_run, "status": cmd_status, "retry": cmd_retry}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

"""A082: reproduction of the HLTCOE JHU submission to the VoicePrivacy Challenge 2024.

Importing the package caps the CPU thread pools at the CPUs the process may actually use. In rented
containers the CPU count of the whole host is visible (208 on the AutoDL machine of 2026-10-19) while the
cgroup quota is far lower (16), and torch and onnxruntime then start about 100 threads each, which made
Piper about 15 times slower. The cap goes into the environment before torch is imported, so subprocesses
(the VPC evaluation) inherit it. `A082_THREADS` overrides it, e.g. when several jobs share one machine.
It also switches torch to the cuDNN v7 API (see below).
"""
from __future__ import annotations

import os


def cpu_limit() -> int:
    """CPUs this process may use: the cgroup quota (v2, then v1) and the affinity mask, whichever is lower."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for quota_file, period_file in (("/sys/fs/cgroup/cpu.max", None),
                                    ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            with open(quota_file) as f:
                parts = f.read().split()
            if period_file is None:
                quota, period = parts[0], parts[1]
            else:
                with open(period_file) as f:
                    quota, period = parts[0], f.read().strip()
        except (OSError, IndexError):
            continue
        if quota not in ("max", "-1"):
            n = min(n, max(1, int(quota) // int(period)))
        break
    return n


def threads() -> int:
    return max(1, int(os.environ.get("A082_THREADS") or cpu_limit()))


for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(threads()))

# torch 2.0's cuDNN v8 API keeps an execution plan for every input shape it has seen, without a limit: with
# utterances of all lengths a kNN-VC process grew by ~9 MB per new length (to 14-26 GB) and spent time building
# plans. The v7 API does neither (measured 2026-10-20: 600 lengths, +5.45 GB in 152 s vs +0 GB in 75 s).
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

"""Task queue for the batch runs on one machine with several GPUs (replaces the fixed lanes of lanes.sh).

    python scripts/batch_queue.py <work dir>            # in tmux; restartable: finished tasks have markers in <work>/done

Every task waits for its dependencies (markers in <work>/done/). A free GPU takes the ready task with the
highest priority (lowest number). Anonymization tasks are split into shards: a GPU slot runs `procs` shard
processes at once, and several GPUs can work on one system, so the long cascade does not hold up the end.
Evaluations take one GPU each and run in separate framework copies (--workspace auto). CPU-only steps
(admixture links, transcoding) run beside the GPU jobs. Failed jobs are retried twice; a task that still
fails gets <id>.failed and blocks only its dependants.

When nothing runs and nothing more can start, the queue writes logs/queue_end.txt and, after a grace period,
shuts the machine down (AutoDL stops billing the GPUs). Create <work>/no_shutdown to prevent that.
Status: logs/queue.log (events), logs/queue_status.json (running, done, failed, waiting).
Tasks added while the queue runs: <work>/tasks_extra.json, a list of objects with the fields of `Task` (commands may
use {W}, {R}, {LOGS}, {POOL}, {PIPER}); the file is read again every cycle, new ids are added. Finished shards leave
<id>.shard<i> in <work>/done, so a restarted queue does not hand them out again.
Training of ID 5-8 is not here: it waits for the Huawei vouchers (until 2026-10-26).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/a082")
R = W / "vpc2024"
CODE = Path(__file__).resolve().parent.parent
DONE, LOGS = W / "done", W / "logs"
PIPER = W / "cache/piper/en_US-libritts_r-medium.onnx"
POOL = W / "cache/pool_vox1"
GRACE_S = 600
POLL_S = 20
RETRIES = 2


@dataclass
class Task:
    id: str
    cmd: str                       # shell command run from the code dir with the framework environment
    deps: list[str] = field(default_factory=list)
    prio: int = 50
    gpu: bool = True
    shards: int = 0                # > 0: anonymization split into this many shards (cmd gets --shard i/n)
    procs: int = 3                 # shard processes per GPU slot
    finalize: str = ""             # after all shards: completeness check and wav.scp
    after: str = ""                # extra command run after success (CPU)
    timeout_h: float = 10.0


def knn(sid: str, config: str, deps: list[str], prio: int, source: str = "") -> Task:
    src = f" --source-suffix {source}" if source else ""
    base = f"python -m a082.knnvc.anonymize --config {config} --pool {POOL} --vpc-root {R} --sets all{src}"
    return Task(f"anon_{sid}", base, ["pool.done"] + deps, prio, shards=12, procs=3,
                finalize=f"python -m a082.knnvc.anonymize --config {config} --vpc-root {R} --sets all{src} --finalize-only")


def evaluate(suffix: str, deps: list[str], prio: int, row: str = "") -> Task:
    r = f" --row {row}" if row else ""
    return Task(f"eval{suffix}", f"python -m a082.eval.vpc run --vpc-root {R} --suffix {suffix} --workspace auto",
                deps, prio, after=f"python -m a082.eval.vpc collect --vpc-root {R} --suffix {suffix}{r} "
                                  f"> {LOGS}/result{suffix}.txt 2>&1", timeout_h=12)


def separation(mix: str, deps: list[str], prio: int, level: str = "utt") -> list[Task]:
    """Classify the admixture's utterances, train four attackers on the split, analyse by trial type."""
    name = mix.lstrip("_")
    lv = f" --level {level}" if level != "utt" else ""
    t = [Task(f"split{mix}", f"python -m a082.bonus.sysclf split --vpc-root {R} --mix-suffix {mix} --p 0.4{lv}",
              ["sysclf"] + deps, prio)]
    for s in ("pred_knn", "pred_tts", "true_knn", "true_tts"):
        t.append(Task(f"attack{mix}_{s}", f"python -m a082.bonus.attack train --vpc-root {R} --mix-suffix {mix} "
                      f"--subset {s} --workspace auto", [f"split{mix}"], prio, timeout_h=12))
    t.append(Task(f"analyse{mix}", f"mkdir -p {W}/bonus && python -m a082.bonus.attack analyse --vpc-root {R} "
                  f"--mix-suffix {mix} --labels work/bonus/labels/{name} --models single=exp/asv_anon{mix} "
                  f"knn=exp/asv_anon{mix}__pred_knn tts=exp/asv_anon{mix}__pred_tts --json {W}/bonus/attack{mix}.json",
                  [f"attack{mix}_{s}" for s in ("pred_knn", "pred_tts", "true_knn", "true_tts")] + [f"eval{mix}"],
                  prio))
    return t


def tasks() -> list[Task]:
    T = [
        Task("anon_tts", f"python -m a082.cascade.anonymize --vpc-root {R} --piper-model {PIPER} "
             f"--transcripts {W}/transcripts --sets all", [], 1, shards=12, procs=3,
             finalize=f"python -m a082.cascade.anonymize --vpc-root {R} --piper-model {PIPER} "
                      f"--transcripts {W}/transcripts --sets all --finalize-only",
             after=f"python scripts/whisper_wer.py --vpc-root {R} --transcripts {W}/transcripts "
                   f"> {LOGS}/whisper_wer.txt 2>&1"),
        knn("knn1", "configs/knn1.yaml", [], 0),
        # calibrate system 3's noise on dev, then write the closest setting into <work>/knn3.yaml
        Task("g9", f"python scripts/calibrate_noise.py --vpc-root {R} --pool {POOL} > {LOGS}/g9.txt 2>&1",
             ["pool.done"], 2, after=f"python - <<'PY'\n"
             f"import json, re, yaml\n"
             f"t = open('{LOGS}/g9.txt').read(); j = json.loads(t[t.rindex('\\n{{\\n') + 1:])\n"
             f"where, delta = j['closest'].split('/')\n"
             f"c = yaml.safe_load(open('configs/knn3.yaml'))\n"
             f"c['noise'] = {{'where': where, 'delta': float(delta)}}\n"
             f"open('{W}/knn3.yaml', 'w').write(yaml.safe_dump(c, sort_keys=False))\n"
             f"print('knn3 noise:', c['noise'], j['wer'])\nPY"),
        Task("sysclf", f"python -m a082.bonus.sysclf train --vpc-root {R}", ["anon_knn1", "anon_tts"], 3),
        Task("sysclf_eval", f"mkdir -p {W}/bonus && python -m a082.bonus.sysclf eval --vpc-root {R} --sets eval "
             f"--json {W}/bonus/sysclf_eval.json", ["sysclf"], 3),   # needs a GPU: the model is saved on one
        evaluate("_knn1", ["anon_knn1"], 0, "1"),   # first: the first full and parallel evaluation, early warning
        evaluate("_tts", ["anon_tts"], 4, "4"),
        Task("admix", f"python -m a082.admix --vpc-root {R} --p 0.2 0.325 0.4 0.1 0.15 0.25 0.3 0.35 0.5",
             ["anon_knn1", "anon_tts"], 4, gpu=False),
    ]
    for p, row in (("0p4", "mix0.4"), ("0p2", "mix0.2"), ("0p325", "mix0.325")):
        T.append(evaluate(f"_mix{p}", ["admix"], 5, row))
    T += separation("_mix0p4", ["admix"], 6)
    T += [
        knn("knn1m", "configs/knn1m.yaml", [], 7),
        knn("knn2", "configs/knn2.yaml", [], 7),
        knn("knn3", f"{W}/knn3.yaml", ["g9"], 7),
        evaluate("_knn1m", ["anon_knn1m"], 8),
        evaluate("_knn2", ["anon_knn2"], 8, "2"),
        evaluate("_knn3", ["anon_knn3"], 8, "3"),
    ]
    for p in ("0p1", "0p15", "0p25", "0p3", "0p35", "0p5"):
        T.append(evaluate(f"_mix{p}", ["admix"], 9))
    # improvement A: kNN-VC on the cascade's output, mixed with ID 1
    T += [
        knn("ttsknn", "configs/ttsknn.yaml", ["anon_tts"], 10, source="_tts"),
        Task("admixA", f"python -m a082.admix --vpc-root {R} --tts-suffix _ttsknn --out-suffix _mixA0p4 --p 0.4",
             ["anon_knn1", "anon_ttsknn"], 10, gpu=False),
        evaluate("_mixA0p4", ["admixA"], 10),
    ]
    T += separation("_mixA0p4", ["admixA"], 11)
    # improvement B: the system is fixed per speaker
    T += [
        Task("admixB", f"python -m a082.admix --vpc-root {R} --level spk --p 0.4", ["anon_knn1", "anon_tts"], 10,
             gpu=False),
        evaluate("_mix0p4spk", ["admixB"], 10),
    ]
    T += separation("_mix0p4spk", ["admixB"], 11, level="spk")
    # real-world condition: Opus 12 kb/s transcoding of the evaluation sets
    for suf, dep in (("_knn1", "anon_knn1"), ("_orig", ""), ("_tts", "anon_tts"), ("_mix0p4", "admix")):
        T.append(Task(f"tcmake{suf}", f"python -m a082.bonus.transcode make --vpc-root {R} --suffix {suf} "
                      f"--codec opus12k", [dep] if dep else [], 12, gpu=False))
        T.append(Task(f"tceval{suf}", f"python -m a082.bonus.transcode evaluate --vpc-root {R} --suffix {suf} "
                      f"--codec opus12k --workspace auto", [f"tcmake{suf}"], 12, timeout_h=12))
    return T


# ---------------------------------------------------------------------------------------------------------

def log(msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOGS / "queue.log", "a") as f:
        f.write(line + "\n")


def ngpu() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return len([l for l in out.splitlines() if l.startswith("GPU ")])


def cpu_share(n: int) -> int:
    sys.path.insert(0, str(CODE))
    import a082  # noqa: E402
    return max(1, a082.cpu_limit() // n)


def launch(cmd: str, logfile: Path, gpu: int | None, threads: int) -> subprocess.Popen:
    env = dict(os.environ, A082_THREADS=str(threads))
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # huggingface.co is unreachable from the machine
    # loading and freeing ~60 MB matching sets fragmented glibc's heap (kNN processes grew to 16-26 GB); a fixed
    # mmap threshold returns large blocks to the system when they are freed
    env.setdefault("MALLOC_MMAP_THRESHOLD_", "1048576")
    env.setdefault("MALLOC_ARENA_MAX", "4")
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    full = f"cd {shlex.quote(str(R))} && set +u && source env.sh && set -u && cd {shlex.quote(str(CODE))} && {cmd}"
    f = open(logfile, "a")
    f.write(f"\n### {time.strftime('%F %T')} gpu={gpu} threads={threads}\n### {cmd}\n")
    f.flush()
    return subprocess.Popen(["bash", "-c", full], stdout=f, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def run_sync(cmd: str, logfile: Path, threads: int) -> bool:
    p = launch(cmd, logfile, None, threads)
    return p.wait() == 0


class Job:
    def __init__(self, task: Task, gpu: int | None, procs: list[subprocess.Popen], shards: list[int]):
        self.task, self.gpu, self.procs, self.shards, self.start = task, gpu, procs, shards, time.time()

    def finished(self) -> bool:
        return all(p.poll() is not None for p in self.procs)

    def ok(self) -> bool:
        return all(p.returncode == 0 for p in self.procs)

    def kill(self) -> None:
        for p in self.procs:
            try:
                os.killpg(p.pid, 15)
            except ProcessLookupError:
                pass


def extra_tasks(known: set[str]) -> list[Task]:
    f = W / "tasks_extra.json"
    if not f.exists():
        return []
    try:
        items = json.loads(f.read_text())
    except (OSError, ValueError) as e:
        log(f"tasks_extra.json unreadable: {e}")
        return []
    fmt = dict(W=W, R=R, LOGS=LOGS, POOL=POOL, PIPER=PIPER)
    out = []
    for d in items:
        if d.get("id") in known:
            continue
        d = {k: (v.format(**fmt) if isinstance(v, str) and k in ("cmd", "finalize", "after") else v) for k, v in d.items()}
        out.append(Task(**d))
    return out


def main() -> None:
    DONE.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    T = tasks()
    by = {t.id: t for t in T}
    G = ngpu()
    share = cpu_share(G)
    is_done = lambda k: (DONE / k).exists()
    is_failed = lambda k: (DONE / f"{k}.failed").exists()
    jobs: list[Job] = []
    shard_done = lambda t, i: (DONE / f"{t.id}.shard{i}").exists()
    claimed: dict[str, set[int]] = {t.id: {i for i in range(t.shards) if shard_done(t, i)} for t in T if t.shards}
    tries: dict[str, int] = {t.id: 0 for t in T}
    idle_since = None
    log(f"queue start: {G} GPUs, {share} CPU threads per GPU, {len(T)} tasks, "
        f"{sum(is_done(t.id) for t in T)} already done")

    def ready(t: Task) -> bool:
        return not is_done(t.id) and not is_failed(t.id) and all(is_done(d) for d in t.deps)

    def blocked(t: Task) -> bool:
        return any(is_failed(d) or (d in by and blocked(by[d])) for d in t.deps)

    while True:
        # 0. tasks added while running
        for t in extra_tasks(set(by)):
            T.append(t); by[t.id] = t; tries[t.id] = 0
            if t.shards:
                claimed[t.id] = {i for i in range(t.shards) if shard_done(t, i)}
            log(f"added task {t.id}")
        # 1. collect finished jobs
        for j in [j for j in jobs if j.finished() or time.time() - j.start > j.task.timeout_h * 3600]:
            t = j.task
            if not j.finished():
                log(f"timeout {t.id} after {t.timeout_h} h, killing")
                j.kill()
                time.sleep(5)
            jobs.remove(j)
            if t.shards:
                if j.ok() and j.finished():
                    for i in j.shards:
                        (DONE / f"{t.id}.shard{i}").touch()
                    log(f"{t.id}: shards {j.shards} finished on GPU {j.gpu}")
                    continue
                log(f"{t.id}: shards {j.shards} failed on GPU {j.gpu} (exit codes {[p.returncode for p in j.procs]}); "
                    f"they go back to the pool of shards")
                claimed[t.id] -= set(j.shards)
                tries[t.id] += 1
                if tries[t.id] > RETRIES * 3:
                    (DONE / f"{t.id}.failed").touch()
                    log(f"FAILED {t.id} (shards keep failing)")
                continue
            if j.ok() and j.finished():
                if t.after and not run_sync(t.after, LOGS / f"q_{t.id}.log", share):
                    log(f"{t.id}: post step failed (see q_{t.id}.log); marking done anyway")
                (DONE / t.id).touch()
                log(f"done {t.id} ({(time.time() - j.start) / 3600:.2f} h)")
            else:
                tries[t.id] += 1
                log(f"{t.id} failed (try {tries[t.id]}, exit codes {[p.returncode for p in j.procs]}), see q_{t.id}.log")
                if tries[t.id] > RETRIES:
                    (DONE / f"{t.id}.failed").touch()
                    log(f"FAILED {t.id}")

        # 2. sharded tasks whose shards are all claimed and finished: finalize
        for t in T:
            if t.shards and not is_done(t.id) and not is_failed(t.id) and len(claimed[t.id]) == t.shards \
                    and not any(j.task.id == t.id for j in jobs):
                if run_sync(t.finalize, LOGS / f"q_{t.id}_finalize.log", share):
                    if t.after and not run_sync(t.after, LOGS / f"q_{t.id}.log", share):
                        log(f"{t.id}: post step failed (see q_{t.id}.log)")
                    (DONE / t.id).touch()
                    log(f"done {t.id}")
                else:
                    tries[t.id] += 1
                    log(f"{t.id}: finalize found missing outputs; running all shards again (they skip finished files)")
                    claimed[t.id] = set()
                    for i in range(t.shards):
                        (DONE / f"{t.id}.shard{i}").unlink(missing_ok=True)
                    if tries[t.id] > RETRIES * 3:
                        (DONE / f"{t.id}.failed").touch()
                        log(f"FAILED {t.id}")

        # 3. CPU tasks
        for t in T:
            if not t.gpu and ready(t) and not any(j.task.id == t.id for j in jobs):
                jobs.append(Job(t, None, [launch(t.cmd, LOGS / f"q_{t.id}.log", None, share)], []))
                log(f"start {t.id} (CPU)")

        # 4. GPU tasks onto free GPUs, highest priority first
        busy = {j.gpu for j in jobs if j.gpu is not None}
        for g in [g for g in range(G) if g not in busy]:
            cands = []
            for t in T:
                if not t.gpu or not ready(t):
                    continue
                if t.shards:
                    if len(claimed[t.id]) < t.shards:
                        cands.append(t)
                elif not any(j.task.id == t.id for j in jobs):
                    cands.append(t)
            if not cands:
                break
            t = min(cands, key=lambda x: x.prio)
            if t.shards:
                free = [i for i in range(t.shards) if i not in claimed[t.id]][: t.procs]
                claimed[t.id] |= set(free)
                n = len(free)
                procs = [launch(f"{t.cmd} --shard {i}/{t.shards}", LOGS / f"q_{t.id}_{i}.log", g, max(1, share // n))
                         for i in free]
                jobs.append(Job(t, g, procs, free))
                log(f"start {t.id} shards {free} on GPU {g}")
            else:
                jobs.append(Job(t, g, [launch(t.cmd, LOGS / f"q_{t.id}.log", g, share)], []))
                log(f"start {t.id} on GPU {g}")

        # 5. status, and the end
        status = {
            "time": time.strftime("%F %T"),
            "running": [{"task": j.task.id, "gpu": j.gpu, "shards": j.shards,
                         "hours": round((time.time() - j.start) / 3600, 2)} for j in jobs],
            "done": [t.id for t in T if is_done(t.id)],
            "failed": [t.id for t in T if is_failed(t.id)],
            "waiting": [t.id for t in T if not is_done(t.id) and not is_failed(t.id)
                        and not any(j.task.id == t.id for j in jobs)],
        }
        (LOGS / "queue_status.json").write_text(json.dumps(status, indent=1))
        if not jobs:
            idle_since = idle_since or time.time()
            stuck = [t.id for t in T if not is_done(t.id) and not is_failed(t.id) and blocked(t)]
            if time.time() - idle_since > GRACE_S:
                (LOGS / "queue_end.txt").write_text(json.dumps({**status, "blocked_by_failure": stuck}, indent=1))
                if (W / "no_shutdown").exists():
                    log("nothing left to run; no_shutdown present, queue exits")
                    return
                log("nothing left to run; shutting the machine down")
                subprocess.run(["sync"])
                subprocess.run(["shutdown"])
                return
        else:
            idle_since = None
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()

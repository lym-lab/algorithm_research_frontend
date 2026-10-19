"""Separate working copies of the VPC framework, so that several evaluations can run at the same time.

run_evaluation.py works relative to its current directory (data/, exp/) and every run rebuilds the same
directories: data/libri_{dev,test}_asr, exp/asv_orig/cosine_out (where --force_compute deletes every
folder whose name merely contains the suffix, so _knn1 also hits _knn1m), and the cached ASR outputs. Two
runs in one tree therefore overwrite each other. A workspace is a directory in which

- every top-level entry of the framework except data/ and exp/ is a symlink (code, env.sh, corpora, ...);
- data/ holds a symlink per data set, except the *_asr folders, which the run rebuilds for itself;
- exp/ holds a real folder per model folder with a symlink per entry, except the entries this run
  writes: names ending in the run's suffix (or `<suffix>.`), the folders listed in `fresh` (the
  attacker model being trained) and cosine_out (ASV scoring output, rebuilt per run);

so everything a run writes lands inside the workspace. `merge` then moves the files the run created
(real files, not symlinks) into the framework's exp/ at the same relative paths, which is where a serial
run would have written them, and removes the workspace. The framework's code is not changed.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

SCORING_DIRS = ("cosine_out", "plda_out")


def _own(name: str, suffix: str) -> bool:
    return bool(suffix) and (name.endswith(suffix) or f"{suffix}." in name)


def create(vpc_root: Path, ws: Path, suffix: str, fresh: tuple[str, ...] = ()) -> Path:
    vpc_root, ws = Path(vpc_root).resolve(), Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for entry in vpc_root.iterdir():
        if entry.name not in ("data", "exp"):
            (ws / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (ws / "data").mkdir()
    for entry in (vpc_root / "data").iterdir():
        if "_asr" not in entry.name:
            (ws / "data" / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (ws / "exp").mkdir()
    exp = vpc_root / "exp"
    for entry in (exp.iterdir() if exp.exists() else []):
        if entry.name in fresh or _own(entry.name, suffix):
            continue
        dst = ws / "exp" / entry.name
        if not entry.is_dir():
            dst.symlink_to(entry)
            continue
        dst.mkdir()
        for child in entry.iterdir():
            if child.name in SCORING_DIRS or _own(child.name, suffix):
                continue
            (dst / child.name).symlink_to(child, target_is_directory=child.is_dir())
    return ws


def merge(ws: Path, vpc_root: Path) -> list[Path]:
    """Move the files the run created under ws/exp into vpc_root/exp; return them (relative paths)."""
    ws, vpc_root = Path(ws), Path(vpc_root)
    moved = []
    for root, dirs, files in os.walk(ws / "exp"):          # os.walk does not descend into symlinked dirs
        for f in files:
            src = Path(root) / f
            if src.is_symlink():
                continue
            rel = src.relative_to(ws / "exp")
            dst = vpc_root / "exp" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            moved.append(rel)
    shutil.rmtree(ws)
    return moved

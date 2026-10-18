"""The loop every anonymizer shares: for each VPC data set, convert each utterance and write
`data/<set><suffix>/wav/<utt>.wav`; skip finished files (restartable); split work into shards.

An anonymizer supplies `convert(utt, wav[1, T] at 16 kHz) -> wav[1, T'] at 16 kHz` and, optionally,
`order(utts) -> utts` to group utterances that share expensive state (e.g. a matching set).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable

import torch
from tqdm import tqdm

from . import audio, kaldi

log = logging.getLogger(__name__)


def resolve_sets(spec: str) -> list[str]:
    if spec == "all":
        return kaldi.ALL_SETS
    if spec == "eval":
        return kaldi.EVAL_SETS
    return spec.split(",")


def parse_shard(spec: str) -> tuple[int, int]:
    i, n = map(int, spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard {spec}")
    return i, n


def anonymize_sets(vpc_root: Path, sets: Iterable[str], suffix: str,
                   convert: Callable[[str, torch.Tensor], torch.Tensor] | None,
                   order: Callable[[list[str]], list[str]] | None = None,
                   shard=(0, 1), finalize_only: bool = False) -> None:
    data = Path(vpc_root) / "data"
    sets = list(sets)
    if not finalize_only:
        for name in sets:
            src, out = data / name, data / f"{name}{suffix}"
            kaldi.prepare_output_dir(src, out)
            scp = kaldi.read_kaldi(src / "wav.scp")
            utts = sorted(scp)[shard[0]::shard[1]]
            todo = [u for u in utts if not kaldi.wav_path(out, u).exists()]
            if order is not None:
                todo = order(todo)
            for u in tqdm(todo, desc=f"{name}{suffix} [{shard[0]}/{shard[1]}]"):
                wav, sr = audio.load_scp_entry(scp[u], root=vpc_root)
                wav = audio.resample(wav, sr)
                audio.save_wav16(kaldi.wav_path(out, u), convert(u, wav))
    if finalize_only or shard[1] == 1:
        for name in sets:
            kaldi.finalize_output_dir(data / name, data / f"{name}{suffix}")


def add_common_args(ap) -> None:
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--sets", default="all", help="all | eval | comma-separated set names")
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles every n-th utterance from i")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--finalize-only", action="store_true",
                    help="after all shards finished: check completeness, write wav.scp and utt2dur")

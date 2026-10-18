"""Kaldi-style data directories as used by the VPC 2024 framework.

An anonymized copy of `data/<set>` is written to `data/<set><suffix>`: all metadata files are copied,
audio goes to `wav/<utt>.wav`, and `wav.scp` lists absolute paths. The ASV training script takes the
speaker id from the file name (split on '-'), so file names must stay `<utt>.wav`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import soundfile as sf

# The data sets every system anonymizes (evaluation plan §2.1, §4.2, §6).
EVAL_SETS = [
    "libri_dev_enrolls", "libri_dev_trials_f", "libri_dev_trials_m",
    "libri_test_enrolls", "libri_test_trials_f", "libri_test_trials_m",
    "IEMOCAP_dev", "IEMOCAP_test",
]
TRAIN_SET = "train-clean-360"
ALL_SETS = EVAL_SETS + [TRAIN_SET]


def read_kaldi(path: str | Path) -> dict[str, str]:
    """Map the first field of each line to the rest of the line (kept as one string)."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            key, _, value = line.strip().partition(" ")
            out[key] = value.strip()
    return out


def write_kaldi(data: dict[str, str], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for key in sorted(data):
            f.write(f"{key} {data[key]}\n")


def wav_path(out_dir: str | Path, utt: str) -> Path:
    return Path(out_dir) / "wav" / f"{utt}.wav"


def prepare_output_dir(src_dir: str | Path, out_dir: str | Path) -> None:
    """Create `out_dir` and copy every metadata file of `src_dir` except `wav.scp`."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        if f.is_file() and f.name != "wav.scp":
            shutil.copy2(f, out_dir / f.name)


def finalize_output_dir(src_dir: str | Path, out_dir: str | Path) -> None:
    """Write `wav.scp` (absolute paths) and a fresh `utt2dur`, after checking that every utterance of
    `src_dir` has its output. Raises if any is missing: the VPC result preparation fails on an incomplete
    set (README FAQ 4), so it is better to fail here."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    utts = list(read_kaldi(src_dir / "wav.scp"))
    missing = [u for u in utts if not wav_path(out_dir, u).is_file()]
    if missing:
        raise RuntimeError(f"{out_dir}: {len(missing)} of {len(utts)} outputs missing, e.g. {missing[:3]}")
    scp, dur = {}, {}
    for u in utts:
        p = wav_path(out_dir, u).absolute()  # not resolve(): keep the link's own path
        info = sf.info(p)
        if info.samplerate != 16000 or info.subtype != "PCM_16" or info.channels != 1:
            raise RuntimeError(f"{p}: {info.samplerate} Hz {info.subtype} {info.channels} ch; need 16 kHz PCM_16 mono")
        scp[u] = str(p)
        dur[u] = f"{info.frames / info.samplerate:.3f}"
    write_kaldi(scp, out_dir / "wav.scp")
    write_kaldi(dur, out_dir / "utt2dur")  # lengths change in systems 2-4; the copied utt2dur would be stale


UTT_KEYED = ("wav.scp", "utt2spk", "text", "utt2dur", "utt2emo", "text_no_norm")
SPK_KEYED = ("spk2gender", "spk2fold")


def subset_dir(src_dir: str | Path, out_dir: str | Path, keep: set[str]) -> int:
    """Copy a Kaldi directory keeping only the utterances in `keep`; speaker files follow.
    Returns the number of utterances kept."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    utt2spk = {u: s for u, s in read_kaldi(src_dir / "utt2spk").items() if u in keep}
    speakers = set(utt2spk.values())
    for name in UTT_KEYED:
        if (src_dir / name).exists():
            write_kaldi({u: v for u, v in read_kaldi(src_dir / name).items() if u in keep}, out_dir / name)
    for name in SPK_KEYED:
        if (src_dir / name).exists():
            write_kaldi({s: v for s, v in read_kaldi(src_dir / name).items() if s in speakers}, out_dir / name)
    spk2utt: dict[str, list[str]] = {}
    for u, s in sorted(utt2spk.items()):
        spk2utt.setdefault(s, []).append(u)
    write_kaldi({s: " ".join(us) for s, us in spk2utt.items()}, out_dir / "spk2utt")
    return len(utt2spk)


def link_output_dir(src_dir: str | Path, out_dir: str | Path, choose: dict[str, Path]) -> None:
    """Build `out_dir` from existing outputs without re-anonymizing: `choose[utt]` is the wav to use.
    Used by the admixture systems. Links are absolute symlinks (copies where symlinks are unavailable)."""
    prepare_output_dir(src_dir, out_dir)
    for utt, target in choose.items():
        dst = wav_path(out_dir, utt)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(Path(target).resolve())
        except OSError:
            shutil.copy2(target, dst)
    finalize_output_dir(src_dir, out_dir)

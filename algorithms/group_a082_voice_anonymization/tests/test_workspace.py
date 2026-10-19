"""Separate working copies for concurrent evaluations (a082/eval/workspace.py), with a fake framework that
writes where run_evaluation.py writes, including its substring-matching cleanup of cosine_out."""
import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from a082.eval import vpc

FAKE = r'''
import json, shutil, sys, time
from pathlib import Path
o = json.loads(sys.argv[sys.argv.index("--overwrite") + 1]); s = o["anon_data_suffix"]
cos = Path("exp/asv_orig/cosine_out")
for d in (list(cos.iterdir()) if cos.exists() else []):
    if s in d.name:                                   # the framework's force_compute rule
        shutil.rmtree(d)
time.sleep(0.5)                                       # let the two runs overlap
for d in ("data/libri_dev_asr", f"data/libri_dev_asr{s}", f"exp/asr/libri_dev_asr{s}",
          f"cos/x{s}".replace("cos", str(cos)), f"exp/asv_anon{s}", f"exp/results_summary/eval_anon{s}"):
    Path(d).mkdir(parents=True, exist_ok=True)
Path("data/libri_dev_asr/text").write_text(s)         # shared name: rebuilt by every run
Path(f"exp/asr/libri_dev_asr{s}/text").write_text(Path("exp/asr/libri_dev_asr/text").read_text() + s)
Path(f"exp/asr/results{s}.csv").write_text(s)
Path(f"{cos}/x{s}/scores").write_text(s)
Path(f"exp/asv_anon{s}/CKPT").write_text(Path(f"data/libri_dev{s}/wav.scp").read_text())
Path(f"exp/results_summary/eval_anon{s}/results.txt").write_text(s)
'''


def make_root(root: Path):
    root.mkdir(parents=True)
    (root / "env.sh").write_text("")
    (root / "run_evaluation.py").write_text(FAKE)
    for s in ("", "_knn1", "_knn1m"):
        (root / "data" / f"libri_dev{s}").mkdir(parents=True)
        (root / "data" / f"libri_dev{s}" / "wav.scp").write_text(f"u1 /x/u1{s}.wav\n")
    (root / "exp/asr/libri_dev_asr").mkdir(parents=True)
    (root / "exp/asr/libri_dev_asr/text").write_text("ORIG")        # cached original ASR output
    (root / "exp/asv_orig/cosine_out/x_knn1m").mkdir(parents=True)
    (root / "exp/asv_orig/cosine_out/x_knn1m/scores").write_text("old")
    (root / "exp/asv_orig/CKPT").write_text("model")


@pytest.fixture
def fake_bash(monkeypatch):
    real = subprocess.run

    def run(argv, cwd, check):  # `bash -c "source env.sh && python ..."` -> this interpreter, no bash needed
        args = shlex.split(argv[-1].split("&& ", 1)[1])
        return real([sys.executable] + args[1:], cwd=cwd, check=check)
    monkeypatch.setattr(vpc.subprocess, "run", run)


def cmd(s):
    return [f"python run_evaluation.py --overwrite {shlex.quote(json.dumps({'anon_data_suffix': s}))}"]


def test_concurrent_runs_are_isolated(tmp_path, fake_bash):
    root = tmp_path / "vpc"
    make_root(root)
    try:
        (tmp_path / "probe").symlink_to(root, target_is_directory=True)
    except OSError as e:
        pytest.skip(f"symlink not permitted: {e}")
    errors = []

    def go(s):
        try:
            vpc.framework(root, cmd(s), "auto", s, fresh=(f"asv_anon{s}",))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    th = [threading.Thread(target=go, args=(s,)) for s in ("_knn1", "_knn1m")]
    [t.start() for t in th]; [t.join() for t in th]
    assert not errors
    exp = root / "exp"
    for s in ("_knn1", "_knn1m"):
        assert (exp / f"asr/results{s}.csv").read_text() == s
        assert (exp / f"asr/libri_dev_asr{s}/text").read_text() == "ORIG" + s     # read the shared cache
        assert (exp / f"asv_orig/cosine_out/x{s}/scores").read_text() == s
        assert (exp / f"asv_anon{s}/CKPT").read_text() == f"u1 /x/u1{s}.wav\n"
        assert (exp / f"results_summary/eval_anon{s}/results.txt").read_text() == s
    assert (exp / "asr/libri_dev_asr/text").read_text() == "ORIG"                # cache untouched
    assert (exp / "asv_orig/CKPT").read_text() == "model"
    assert not (root / "data/libri_dev_asr").exists()                             # rebuilt per workspace only
    assert not (tmp_path / "evalws").exists() or not any((tmp_path / "evalws").iterdir())


def test_same_files_as_a_serial_run(tmp_path, fake_bash):
    a, b = tmp_path / "a" / "vpc", tmp_path / "b" / "vpc"
    make_root(a); make_root(b)
    try:
        vpc.framework(a, cmd("_knn1"), "auto", "_knn1")
    except OSError as e:
        pytest.skip(f"symlink not permitted: {e}")
    vpc.framework(b, cmd("_knn1"))                                                 # in place
    files = lambda r: sorted(p.relative_to(r / "exp").as_posix() for p in (r / "exp").rglob("*") if p.is_file())
    other = "asv_orig/cosine_out/x_knn1m/scores"
    # in place, the framework's substring rule deletes another system's scoring output (_knn1 hits _knn1m);
    # in a workspace it survives. Everything else is identical.
    assert other in files(a) and other not in files(b)
    assert [f for f in files(a) if f != other] == files(b)
    for f in files(b):
        assert (a / "exp" / f).read_text() == (b / "exp" / f).read_text()


def test_failed_run_keeps_workspace(tmp_path, fake_bash):
    root = tmp_path / "vpc"
    make_root(root)
    (root / "run_evaluation.py").write_text("raise SystemExit(3)")
    try:
        with pytest.raises(subprocess.CalledProcessError):
            vpc.framework(root, cmd("_knn1"), "auto", "_knn1")
    except OSError as e:
        pytest.skip(f"symlink not permitted: {e}")
    assert (tmp_path / "evalws" / "knn1" / "run_evaluation.py").exists()

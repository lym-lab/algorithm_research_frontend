"""IEMOCAP from the Hugging Face copy: layout, format check, link and coverage check (synthetic bytes)."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_iemocap_hf as prep  # noqa: E402

UTTS = ["Ses01F_impro01_F000", "Ses03M_impro05a_M001", "Ses05F_script01_1_F002"]


def wav_bytes(sr=16000, channels=1, subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, np.zeros((sr // 10, channels), dtype="float32"), sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def test_target_layout(tmp_path):
    assert prep.target(tmp_path, "Ses05F_script01_1_F002") == \
        tmp_path / "Session5/sentences/wav/Ses05F_script01_1/Ses05F_script01_1_F002.wav"
    assert prep.target(tmp_path, "Ses03M_impro05a_M001").parent.name == "Ses03M_impro05a"


def test_write_link_check(tmp_path):
    data = wav_bytes()
    assert prep.write_all([(u + ".wav", data) for u in UTTS], tmp_path / "out") == 3
    assert prep.target(tmp_path / "out", UTTS[0]).read_bytes() == data   # bytes kept unchanged
    vpc = tmp_path / "vpc"
    for name, utts in (("IEMOCAP_dev", UTTS[:2]), ("IEMOCAP_test", UTTS[2:])):
        (vpc / "data" / name).mkdir(parents=True)
        (vpc / "data" / name / "wav.scp").write_text("".join(
            f"{u} {prep.target(Path('data/IEMOCAP/wav'), u).as_posix()}\n" for u in utts))
    try:
        assert prep.link_and_check(vpc, tmp_path / "out") == 3
    except OSError as e:  # Windows without the symlink privilege
        pytest.skip(f"symlink not permitted: {e}")
    (vpc / "data" / "IEMOCAP_test" / "wav.scp").write_text(
        "Ses02F_impro01_F000 data/IEMOCAP/wav/Session2/sentences/wav/Ses02F_impro01/Ses02F_impro01_F000.wav\n")
    with pytest.raises(SystemExit, match="1 of 3"):
        prep.link_and_check(vpc, tmp_path / "out")


@pytest.mark.parametrize("kw", [dict(sr=22050), dict(channels=2), dict(subtype="FLOAT")])
def test_rejects_other_formats(tmp_path, kw):
    with pytest.raises(SystemExit, match="expected"):
        prep.write_all([(UTTS[0] + ".wav", wav_bytes(**kw))], tmp_path)

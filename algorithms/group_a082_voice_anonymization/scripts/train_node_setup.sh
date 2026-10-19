#!/bin/bash
# Setup for a second machine that only trains the WavLM conversion models (rows 5-8), e.g. a Huawei Cloud
# GPU server paid by a voucher. Needs root on Ubuntu with a working NVIDIA driver; no evaluation framework.
#
#   bash scripts/train_node_setup.sh <work dir>      # use the same <work dir> path as on the main machine
#
# Copy these from the main machine first (same absolute paths, so the pool manifest stays valid), e.g.
# relayed through the local PC with `scp -3`:
#   <work dir>/vpc2024/.data.zip               (the framework's data.zip, 10 MB)
#   <work dir>/wc/target.pt, <work dir>/pool_vox1.json
#   the target speaker's audio files listed in pool_vox1.json under that speaker
#   ~/.cache/torch/hub/checkpoints/WavLM-Large.pt   (optional; downloaded from GitHub otherwise)
# Then train as in docs/REPRODUCE.md §7 with --vpc-root <work dir>/vpc2024, and copy <work dir>/wc/*/last.pt back.
#
#   OPENSLR    LibriSpeech source (default: the CN mirror https://openslr.magicdatatech.com)
#   PIP_INDEX  pip index (default: Huawei Cloud's PyPI mirror)
set -euo pipefail

WORK=${1:?usage: train_node_setup.sh <work dir>}
CODE=$(cd "$(dirname "$0")/.." && pwd)
OPENSLR=${OPENSLR:-https://openslr.magicdatatech.com}
PIP_INDEX=${PIP_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple}
VPC=$WORK/vpc2024
mkdir -p "$WORK"/{corpora,logs,wc} "$VPC/corpora"

# 1. Python environment: the framework's torch version (2.0.1, CUDA 11.7 wheels) and what training imports
command -v flac >/dev/null && python3 -m venv --help >/dev/null 2>&1 || \
  { apt-get update -q && apt-get install -y -q flac python3-venv curl unzip; }
if [ ! -x "$WORK/venv/bin/python" ]; then
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q -i "$PIP_INDEX" torch==2.0.1 torchaudio==2.0.2 soundfile numpy pyyaml tqdm
  "$WORK/venv/bin/pip" install -q -i "$PIP_INDEX" --no-deps -e "$CODE"
fi
"$WORK/venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"

# 2. train-clean-360 (23 GB) with its md5, laid out as the framework expects
if [ ! -d "$WORK/corpora/LibriSpeech/train-clean-360" ]; then
  curl -fsSL -o "$WORK/corpora/md5sum_12.txt" "$OPENSLR/resources/12/md5sum.txt"
  curl -fSL --retry 5 -C - -o "$WORK/corpora/train-clean-360.tar.gz" "$OPENSLR/resources/12/train-clean-360.tar.gz"
  (cd "$WORK/corpora" && grep ' train-clean-360.tar.gz$' md5sum_12.txt | md5sum -c -)
  tar -xzf "$WORK/corpora/train-clean-360.tar.gz" -C "$WORK/corpora" && rm "$WORK/corpora/train-clean-360.tar.gz"
fi
[ -e "$VPC/corpora/LibriSpeech" ] || ln -s "$WORK/corpora/LibriSpeech" "$VPC/corpora/LibriSpeech"

# 3. the Kaldi lists of train-clean-360 from data.zip
if [ ! -f "$VPC/data/train-clean-360/wav.scp" ]; then
  [ -f "$VPC/.data.zip" ] || curl -fsSL -o "$VPC/.data.zip" \
    https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/data.zip/data.zip
  (cd "$VPC" && unzip -q -o .data.zip 'data/train-clean-360/*')
fi

# 4. checks: lists readable, first utterance decodes, target and its audio present
"$WORK/venv/bin/python" - "$VPC" "$WORK" <<'PY'
import json, sys
from pathlib import Path
from a082 import audio, kaldi
vpc, work = Path(sys.argv[1]), Path(sys.argv[2])
scp = kaldi.read_kaldi(vpc / "data" / kaldi.TRAIN_SET / "wav.scp")
u, e = next(iter(scp.items()))
wav, sr = audio.load_scp_entry(e, vpc)
print(f"train-clean-360: {len(scp)} utterances; {u}: {wav.shape[-1] / sr:.1f} s at {sr} Hz")
t = work / "wc" / "target.pt"
if t.exists():
    import torch
    spk = torch.load(t, map_location="cpu")["speaker"]
    files = json.loads((work / "pool_vox1.json").read_text())[spk]
    missing = [f for f in files if not Path(f).exists()]
    print(f"target {spk}: {len(files)} files, {len(missing)} missing")
    if missing:
        sys.exit(f"copy the target speaker's audio first, e.g. {missing[0]}")
else:
    print("target.pt not copied yet: pretrain-spk can run now, train needs it")
PY
echo "train node ready: source $WORK/venv/bin/activate, then train with --vpc-root $VPC"

#!/bin/bash
# One-time setup on the GPU machine (Linux, CUDA). Idempotent: each step skips what is already there.
#
#   bash scripts/cloud_setup.sh <work dir on the data disk>     e.g. /data/a082
#
# Needs: git, curl, unzip, flac; ~600 GB free under the work dir; HF_TOKEN in the environment for
# VoxCeleb (a read-only token of an account that accepted the ProgramComputer/voxceleb terms).
# IEMOCAP is linked later by hand (licence; see docs/REPRODUCE.md).
set -euo pipefail

WORK=${1:?usage: cloud_setup.sh <work dir>}
CODE=$(cd "$(dirname "$0")/.." && pwd)
VPC=$WORK/vpc2024
VPC_COMMIT=0ef5506
mkdir -p "$WORK"/{corpora,cache,logs}

# 1. VPC 2024 framework at the pinned commit, its own environment (Python 3.11, torch 2.0.1 + CUDA 11.7)
if [ ! -d "$VPC/.git" ]; then
  git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024.git "$VPC"
  git -C "$VPC" checkout -q $VPC_COMMIT
fi
if [ ! -f "$VPC/.done-python-requirements" ]; then
  (cd "$VPC" && bash 00_install.sh) 2>&1 | tee "$WORK/logs/00_install.log"
fi

# 2. our code and its extra dependencies, in the framework's environment
(
  cd "$VPC" && source env.sh
  pip install -e "$CODE"
  pip install "openai-whisper @ git+https://github.com/openai/whisper@86098128c0b4f24f0e2aa2994de830614b474227" \
              piper-tts==1.8.0 fastapi uvicorn python-multipart matplotlib huggingface_hub
)

# 3. evaluation data lists (data.zip), then libri_dev / libri_test rebuilt from OpenSLR (no sftp password)
if [ ! -d "$VPC/data/libri_dev_enrolls" ]; then
  (cd "$VPC" && curl -sSL -o .data.zip https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/data.zip/data.zip && unzip -q .data.zip)
fi
for part in dev-clean test-clean; do
  if [ ! -d "$WORK/corpora/LibriSpeech/$part" ]; then
    curl -sSL "https://www.openslr.org/resources/12/$part.tar.gz" | tar -xz -C "$WORK/corpora"
  fi
done
(cd "$VPC" && source env.sh && python "$CODE/scripts/rebuild_libri.py" --vpc-root "$VPC" --librispeech "$WORK/corpora/LibriSpeech")

# 4. train-clean-360 and the pretrained evaluation models via the framework's script.
#    It ends with an error while IEMOCAP is missing; everything before that is done by then.
(cd "$VPC" && bash 01_download_data_model.sh) 2>&1 | tee "$WORK/logs/01_download.log" || true
test -d "$VPC/exp/asv_orig" && test -d "$VPC/exp/asr" && test -d "$VPC/corpora/LibriSpeech/train-clean-360"

# 5. kNN-VC target pool: VoxCeleb1 dev + test from the Hugging Face mirror
if [ ! -d "$WORK/corpora/voxceleb1/wav" ]; then
  : "${HF_TOKEN:?set HF_TOKEN for the VoxCeleb mirror}"
  (cd "$VPC" && source env.sh && python - "$WORK/corpora/voxceleb1" <<'PY'
import sys
from huggingface_hub import hf_hub_download
out = sys.argv[1]
for f in ("vox1/vox1_test_wav.zip", "vox1/vox1_dev_wav.zip", "vox1/vox1_meta.csv"):
    hf_hub_download("ProgramComputer/voxceleb", f, repo_type="dataset", local_dir=out)
PY
  )
  mkdir -p "$WORK/corpora/voxceleb1/wav"
  for z in "$WORK"/corpora/voxceleb1/vox1/vox1_{test,dev}_wav.zip; do unzip -q -n "$z" -d "$WORK/corpora/voxceleb1"; done
fi

# 6. public demo targets: LibriTTS-R test-clean (CC BY 4.0) and the Piper voice
if [ ! -d "$WORK/corpora/LibriTTS_R/test-clean" ]; then
  curl -sSL https://www.openslr.org/resources/141/test_clean.tar.gz | tar -xz -C "$WORK/corpora"
fi
mkdir -p "$WORK/cache/piper"
for f in en_US-libritts_r-medium.onnx en_US-libritts_r-medium.onnx.json; do
  [ -f "$WORK/cache/piper/$f" ] || curl -sSL -o "$WORK/cache/piper/$f" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/$f"
done
(cd "$WORK/cache/piper" && sha256sum en_US-libritts_r-medium.onnx | grep -q 10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195 \
  || echo "WARNING: Piper voice differs from the file recorded on 2026-10-18")

echo "setup done: VPC at $VPC, corpora under $WORK/corpora"

#!/bin/bash
# One-time setup on the GPU machine (Linux, CUDA). Idempotent: each step skips what is already there.
#
#   bash scripts/cloud_setup.sh <work dir on the data disk>     e.g. /data/a082
#
# Needs: git, curl, unzip, flac; ~600 GB free under the work dir; HF_TOKEN in the environment for
# VoxCeleb (a read-only token of an account that accepted the ProgramComputer/voxceleb terms).
# IEMOCAP sentence audio comes from the Hugging Face copy AbstractTTS/IEMOCAP (scripts/prepare_iemocap_hf.py).
# env.sh (micromamba activation) may read unset variables, so nounset is lifted around each `source`.
#
# Download sources (check them first with scripts/net_test.sh):
#   OPENSLR=https://openslr.magicdatatech.com   OpenSLR mirror (default https://www.openslr.org)
#   NETWORK_TURBO=off                           do not source /etc/network_turbo (AutoDL's GitHub/HF proxy)
#   HF_ENDPOINT=...                             Hugging Face mirror, read by huggingface_hub itself
set -euo pipefail

OPENSLR=${OPENSLR:-https://www.openslr.org}
if [ "${NETWORK_TURBO:-auto}" != off ] && [ -f /etc/network_turbo ]; then
  set +u; source /etc/network_turbo; set -u
  export no_proxy="${no_proxy:+$no_proxy,}${OPENSLR#https://}"   # the OpenSLR mirror is reached directly
fi

# fetch <url> <file>: resumable download into a file (large archives are checked against OpenSLR's md5sum.txt)
fetch() { curl -fSL --retry 5 -C - -o "$2" "$1"; }

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
  cd "$VPC" && set +u; source env.sh; set -u
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
    curl -fsSL "$OPENSLR/resources/12/$part.tar.gz" | tar -xz -C "$WORK/corpora"
  fi
done
(cd "$VPC" && set +u; source env.sh; set -u && python "$CODE/scripts/rebuild_libri.py" --vpc-root "$VPC" --librispeech "$WORK/corpora/LibriSpeech")

# 3b. IEMOCAP: sentence audio from the Hugging Face copy, linked as data/IEMOCAP/wav (needs data.zip from step 3)
if [ ! -d "$VPC/data/IEMOCAP/wav/Session1" ]; then
  (cd "$VPC" && set +u; source env.sh; set -u && python "$CODE/scripts/prepare_iemocap_hf.py" \
     --vpc-root "$VPC" --out "$WORK/corpora/IEMOCAP_full_release" --parquet-dir "$WORK/corpora/iemocap_hf")
fi

# 4a. train-clean-360 (23 GB) from $OPENSLR next to dev/test-clean; the framework's script links it
if [ ! -d "$WORK/corpora/LibriSpeech/train-clean-360" ]; then
  fetch "$OPENSLR/resources/12/md5sum.txt" "$WORK/corpora/md5sum_12.txt"
  fetch "$OPENSLR/resources/12/train-clean-360.tar.gz" "$WORK/corpora/train-clean-360.tar.gz"
  (cd "$WORK/corpora" && grep ' train-clean-360.tar.gz$' md5sum_12.txt | md5sum -c -)
  tar -xzf "$WORK/corpora/train-clean-360.tar.gz" -C "$WORK/corpora" && rm "$WORK/corpora/train-clean-360.tar.gz"
fi
export librispeech_corpus="$WORK/corpora/LibriSpeech"

# 4. train-clean-360 and the pretrained evaluation models via the framework's script.
#    It stops with an error if data/IEMOCAP/wav is missing; everything before that is done by then.
(cd "$VPC" && bash 01_download_data_model.sh) 2>&1 | tee "$WORK/logs/01_download.log" || true
test -d "$VPC/exp/asv_orig" && test -d "$VPC/exp/asr" && test -d "$VPC/corpora/LibriSpeech/train-clean-360"

# 5. kNN-VC target pool: VoxCeleb1 dev + test from the Hugging Face mirror
if [ ! -d "$WORK/corpora/voxceleb1/wav" ]; then
  : "${HF_TOKEN:?set HF_TOKEN for the VoxCeleb mirror}"
  (cd "$VPC" && set +u; source env.sh; set -u && python - "$WORK/corpora/voxceleb1" <<'PY'
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
  curl -fsSL "$OPENSLR/resources/141/test_clean.tar.gz" | tar -xz -C "$WORK/corpora"
fi
mkdir -p "$WORK/cache/piper"
for f in en_US-libritts_r-medium.onnx en_US-libritts_r-medium.onnx.json; do
  [ -f "$WORK/cache/piper/$f" ] || curl -sSL -o "$WORK/cache/piper/$f" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/$f"
done
(cd "$WORK/cache/piper" && sha256sum en_US-libritts_r-medium.onnx | grep -q 10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195 \
  || echo "WARNING: Piper voice differs from the file recorded on 2026-10-18")

echo "setup done: VPC at $VPC, corpora under $WORK/corpora"

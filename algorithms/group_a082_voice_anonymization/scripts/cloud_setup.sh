#!/bin/bash
# One-time setup on the GPU machine (Linux, CUDA). Idempotent: each step skips what is already there.
#
#   bash scripts/cloud_setup.sh <work dir on the data disk>     e.g. /data/a082
#
# Needs: git, curl, unzip, flac; ~600 GB free under the work dir; a Hugging Face login (or HF_TOKEN) for
# VoxCeleb (a read-only token of an account that accepted the ProgramComputer/voxceleb terms).
# IEMOCAP sentence audio comes from the Hugging Face copy AbstractTTS/IEMOCAP (scripts/prepare_iemocap_hf.py).
# env.sh (micromamba activation) may read unset variables, so nounset is lifted around each `source`.
#
# Download sources (check them first with scripts/net_test.sh):
#   OPENSLR=https://openslr.magicdatatech.com   OpenSLR mirror (default https://www.openslr.org)
#   NETWORK_TURBO=off                           do not source /etc/network_turbo (AutoDL's GitHub/HF proxy)
#   HF_ENDPOINT=https://hf-mirror.com           Hugging Face mirror (huggingface_hub reads it; also used for curl)
#   PROXY=off                                   ignore a platform-set http(s)_proxy (Compshare containers set one)
#   LIBRISPEECH=hf                              LibriSpeech archives from <HF>/datasets/k2-fsa/LibriSpeech instead
#                                               of OpenSLR (same files: checked against OpenSLR's md5 below)
#   GH_ASSETS=<dir>                             GitHub files fetched elsewhere and copied here (for machines where
#                                               GitHub is slow): vpc2024.tar (framework checkout with .git),
#                                               micromamba-linux-64, data.zip, asv_orig.zip, asr.zip, ser.zip,
#                                               WavLM-Large.pt, prematch_g_02500000.pt, whisper-<commit>.tar.gz
#   DEMO_DATA=skip                              skip LibriTTS-R and the Piper voice (only the demo needs them)
#   PIP_INDEX_URL=...                           pip mirror, read by pip itself
set -euo pipefail

OPENSLR=${OPENSLR:-https://www.openslr.org}
HF=${HF_ENDPOINT:-https://huggingface.co}
if [ "${PROXY:-on}" = off ]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
elif [ "${NETWORK_TURBO:-auto}" != off ] && [ -f /etc/network_turbo ]; then
  set +u; source /etc/network_turbo; set -u
  export no_proxy="${no_proxy:+$no_proxy,}${OPENSLR#https://}"   # the OpenSLR mirror is reached directly
fi

# fetch <url> <file>: resumable download into a file
fetch() { curl -fSL --retry 5 -C - -o "$2" "$1"; }
# gh <name> <url> <file>: a GitHub download, taken from $GH_ASSETS when it is there
gh() { if [ -n "${GH_ASSETS:-}" ] && [ -f "$GH_ASSETS/$1" ]; then cp "$GH_ASSETS/$1" "$3"; else fetch "$2" "$3"; fi; }
# LibriSpeech parts with OpenSLR's md5 (resources/12/md5sum.txt, checked 2026-10-19)
declare -A LS_MD5=([dev-clean]=42e2234ba48799c1f50f24a7926300a1 [test-clean]=32fa31d27d2e1cad72775fee3f4849a9
                   [train-clean-360]=c0e676e450a7ff2f54aeade5171606fa)
librispeech() {  # librispeech <part>: download, check, unpack into $WORK/corpora/LibriSpeech/<part>
  local url="$OPENSLR/resources/12/$1.tar.gz" tgz="$WORK/corpora/$1.tar.gz"
  [ "${LIBRISPEECH:-openslr}" = hf ] && url="$HF/datasets/k2-fsa/LibriSpeech/resolve/main/$1.tar.gz"
  fetch "$url" "$tgz"
  (cd "$WORK/corpora" && echo "${LS_MD5[$1]}  $1.tar.gz" | md5sum -c -)
  tar -xzf "$tgz" -C "$WORK/corpora" && rm "$tgz"
}
WHISPER_COMMIT=86098128c0b4f24f0e2aa2994de830614b474227

WORK=${1:?usage: cloud_setup.sh <work dir>}
CODE=$(cd "$(dirname "$0")/.." && pwd)
VPC=$WORK/vpc2024
VPC_COMMIT=0ef5506
mkdir -p "$WORK"/{corpora,cache,logs}

# 1. VPC 2024 framework at the pinned commit, its own environment (Python 3.11, torch 2.0.1 + CUDA 11.7)
if [ ! -d "$VPC/.git" ]; then
  if [ -n "${GH_ASSETS:-}" ] && [ -f "$GH_ASSETS/vpc2024.tar" ]; then
    mkdir -p "$VPC" && tar -xf "$GH_ASSETS/vpc2024.tar" -C "$VPC" --strip-components=1
  else
    git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024.git "$VPC"
  fi
  git -C "$VPC" checkout -q $VPC_COMMIT
fi
if [ ! -f "$VPC/.done-python-requirements" ]; then
  # micromamba placed beforehand; NEW_INSTALL_TRIGGER=no keeps 00_install.sh from deleting it on a first run
  mkdir -p "$VPC/.micromamba"
  [ -x "$VPC/.micromamba/micromamba" ] || { gh micromamba-linux-64 \
    https://github.com/mamba-org/micromamba-releases/releases/download/1.5.1-0/micromamba-linux-64 "$VPC/.micromamba/micromamba"
    chmod +x "$VPC/.micromamba/micromamba"; }
  (cd "$VPC" && NEW_INSTALL_TRIGGER=no bash 00_install.sh) 2>&1 | tee "$WORK/logs/00_install.log"
  test -f "$VPC/.done-python-requirements"
fi

# 2. our code and its extra dependencies, in the framework's environment
(
  cd "$VPC" && set +u; source env.sh; set -u
  pip install -e "$CODE"
  whisper="openai-whisper @ git+https://github.com/openai/whisper@$WHISPER_COMMIT"
  [ -n "${GH_ASSETS:-}" ] && [ -f "$GH_ASSETS/whisper-$WHISPER_COMMIT.tar.gz" ] && whisper="$GH_ASSETS/whisper-$WHISPER_COMMIT.tar.gz"
  pip install "$whisper" piper-tts==1.8.0 fastapi uvicorn python-multipart matplotlib huggingface_hub
)

# 3. evaluation data lists (data.zip), then libri_dev / libri_test rebuilt from OpenSLR (no sftp password)
if [ ! -d "$VPC/data/libri_dev_enrolls" ]; then
  gh data.zip https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/data.zip/data.zip "$VPC/.data.zip"
  (cd "$VPC" && unzip -q -o .data.zip)
fi
for part in dev-clean test-clean; do
  [ -d "$WORK/corpora/LibriSpeech/$part" ] || librispeech $part
done
(cd "$VPC" && set +u; source env.sh; set -u && python "$CODE/scripts/rebuild_libri.py" --vpc-root "$VPC" --librispeech "$WORK/corpora/LibriSpeech")

# 3b. IEMOCAP: sentence audio from the Hugging Face copy, linked as data/IEMOCAP/wav (needs data.zip from step 3)
if [ ! -d "$VPC/data/IEMOCAP/wav/Session1" ]; then
  (cd "$VPC" && set +u; source env.sh; set -u && python "$CODE/scripts/prepare_iemocap_hf.py" \
     --vpc-root "$VPC" --out "$WORK/corpora/IEMOCAP_full_release" --parquet-dir "$WORK/corpora/iemocap_hf")
fi

# 4a. train-clean-360 (23 GB) from $OPENSLR next to dev/test-clean; the framework's script links it
[ -d "$WORK/corpora/LibriSpeech/train-clean-360" ] || librispeech train-clean-360
export librispeech_corpus="$WORK/corpora/LibriSpeech"
# pretrained evaluation models (the framework's script unzips .<model>.zip when it finds one) and kNN-VC weights
for m in asv_orig asr ser; do
  [ -d "$VPC/exp/$m" ] || [ -f "$VPC/.$m.zip" ] || \
    gh $m.zip https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/pre_model.zip/$m.zip "$VPC/.$m.zip"
done
HUB=${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints
mkdir -p "$HUB"
for f in WavLM-Large.pt prematch_g_02500000.pt; do
  [ -f "$HUB/$f" ] || gh $f https://github.com/bshall/knn-vc/releases/download/v0.1/$f "$HUB/$f"
done

# 4. train-clean-360 and the pretrained evaluation models via the framework's script.
#    It stops with an error if data/IEMOCAP/wav is missing; everything before that is done by then.
(cd "$VPC" && bash 01_download_data_model.sh) 2>&1 | tee "$WORK/logs/01_download.log" || true
test -d "$VPC/exp/asv_orig" && test -d "$VPC/exp/asr" && test -d "$VPC/corpora/LibriSpeech/train-clean-360"

# 5. kNN-VC target pool: VoxCeleb1 dev + test from the Hugging Face mirror
if [ ! -d "$WORK/corpora/voxceleb1/wav" ]; then
  [ -n "${HF_TOKEN:-}" ] || [ -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ] || \
    { echo "log in to Hugging Face first (huggingface-cli login) or set HF_TOKEN, for the VoxCeleb mirror"; exit 1; }
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
[ "${DEMO_DATA:-yes}" = skip ] && { echo "setup done (demo data skipped): VPC at $VPC"; exit 0; }
if [ ! -d "$WORK/corpora/LibriTTS_R/test-clean" ]; then
  curl -fsSL "$OPENSLR/resources/141/test_clean.tar.gz" | tar -xz -C "$WORK/corpora"
fi
mkdir -p "$WORK/cache/piper"
for f in en_US-libritts_r-medium.onnx en_US-libritts_r-medium.onnx.json; do
  [ -f "$WORK/cache/piper/$f" ] || curl -sSL -o "$WORK/cache/piper/$f" \
    "$HF/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/$f"
done
(cd "$WORK/cache/piper" && sha256sum en_US-libritts_r-medium.onnx | grep -q 10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195 \
  || echo "WARNING: Piper voice differs from the file recorded on 2026-10-18")

echo "setup done: VPC at $VPC, corpora under $WORK/corpora"

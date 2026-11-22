#!/bin/bash
# One-time setup on the GPU machine (Linux, CUDA, run as root). Idempotent: each step skips what is done,
# so after a failure fix the cause and run the same command again.
#
#   bash scripts/cloud_setup.sh <work dir on the data disk>      e.g. /root/autodl-tmp/a082
#
# Order: disk check -> big downloads start in the background (aria2, retried) -> framework environment
# -> our code -> evaluation data and models -> wait for the downloads -> IEMOCAP, VoxCeleb unpacking.
# Needs ~350 GB free under the work dir and a Hugging Face login (`huggingface-cli login`, or HF_TOKEN) of an
# account that accepted the ProgramComputer/voxceleb terms. Log of each background download: <work>/logs/.
#
# Problems met on a first run (2026-10-19, RTX 4090), handled here:
#   - a slow disk makes everything unusable: step 0 measures it and stops below DISK_MIN MB/s;
#   - the framework's requirements have only lower bounds; pip then pulls 2026 majors (numpy 2 via pyarrow,
#     pandas 3, transformers 5) that break it: they are pinned back to 2024 versions (PINS);
#   - torch 2.0.1+cu117 fails on RTX 40xx (cuFFT CUFFT_INTERNAL_ERROR): the same torch built for CUDA 11.8
#     is installed instead, on every GPU, so all machines run the same build;
#   - one connection to hf-mirror stalls when its signed link expires: aria2 with --lowest-speed-limit and
#     retries; a machine's whole downlink was ~17 MB/s, so the big files start first, in parallel with installs.
#
# Options (environment variables):
#   HF_ENDPOINT=https://hf-mirror.com   Hugging Face mirror (default huggingface.co)
#   PROXY=off                           unset a platform-set http(s)_proxy (Compshare containers set one)
#   NETWORK_TURBO=on                    source /etc/network_turbo (AutoDL's GitHub/HF proxy) for GitHub only
#   LIBRISPEECH=openslr                 LibriSpeech from $OPENSLR (default: the identical archives on HF
#                                       k2-fsa/LibriSpeech; either way checked against OpenSLR's md5)
#   OPENSLR=https://www.openslr.org     OpenSLR host
#   GH_ASSETS=<dir>                     GitHub files fetched elsewhere (for machines where GitHub is slow):
#                                       vpc2024.tar, micromamba-linux-64, data.zip, asv_orig.zip, asr.zip,
#                                       ser.zip, WavLM-Large.pt, prematch_g_02500000.pt,
#                                       whisper-<commit>.tar.gz, knn-vc.tar.gz
#   DISK_MIN=200                        minimum direct read and write speed of the work dir in MB/s (0 = skip)
#   VPC_DIR=/root/vpc2024               put the framework checkout and its Python environment there (e.g. on the
#                                       system disk, so that a saved machine image carries the environment);
#                                       its data/, exp/ and corpora/ are then links into <work dir>, which keeps
#                                       all corpora, outputs and results on the data disk (default <work>/vpc2024)
#   DEMO_DATA=skip                      skip LibriTTS-R (an optional demo target pool; the deployed demo uses
#                                       6 LibriSpeech test-clean speakers, downloaded with the evaluation data)
#   GPU_CHECK=skip                      set up on a machine without a GPU (e.g. a cheap CPU server whose data disk
#                                       then moves to the GPU server); run the script once more on the GPU server,
#                                       which only repeats the quick steps and the GPU check
#   VOXCELEB=skip                       skip VoxCeleb1 (~31 GB; only the target pools need it): a machine for
#                                       rows 5-8 alone gets the target speaker from scripts/wc_bundle.py
#   PIP_INDEX_URL=...                   pip mirror, read by pip itself
#   TORCH_WHEELS=https://mirrors.aliyun.com/pytorch-wheels/cu118  (fetched with curl into <work>/wheels, sha256-checked)
#                                       torch wheels from this directory listing instead of download.pytorch.org
#                                       (there about 0.4 MB/s from AutoDL on 2026-10-19, Aliyun 11 MB/s); wheels
#                                       placed in $GH_ASSETS are used first
set -euo pipefail

WORK=${1:?usage: cloud_setup.sh <work dir>}
CODE=$(cd "$(dirname "$0")/.." && pwd)
VPC=${VPC_DIR:-$WORK/vpc2024}
VPC_COMMIT=0ef5506
WHISPER_COMMIT=86098128c0b4f24f0e2aa2994de830614b474227
HF=${HF_ENDPOINT:-https://huggingface.co}
OPENSLR=${OPENSLR:-https://www.openslr.org}
C=$WORK/corpora
PINS=("numpy==1.26.4" "pyarrow==15.0.2" "pandas==2.2.2" "transformers==4.40.2" "huggingface_hub==0.23.4")
# file:sha256 from https://download.pytorch.org/whl/cu118/<name>/ (read 2026-10-22)
TORCH_WHL_SHA=("torch-2.0.1+cu118-cp311-cp311-linux_x86_64.whl:143b6c658c17d43376e2dfbaa2c106d35639d615e5e8dec4429cf1e510dd8d61"
               "torchaudio-2.0.2+cu118-cp311-cp311-linux_x86_64.whl:7bc0b50a0d83a24bdc3916270c23934345ac84351df76bd3bb08834bdb271df6"
               "torchvision-0.15.2+cu118-cp311-cp311-linux_x86_64.whl:def9af47ebc2cad55c5aa2dad1230dcf4261833ed6df8a73e839bc233764f09e")
mkdir -p "$WORK"/{corpora,cache,logs}
[ "${PROXY:-on}" = off ] && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
GH_PROXY_ENV=()
if [ "${NETWORK_TURBO:-off}" = on ] && [ -f /etc/network_turbo ]; then
  GH_PROXY_ENV=(env $(set +u; source /etc/network_turbo >/dev/null; env | grep -i -E '^(https?_proxy|no_proxy)='))
fi
log() { echo "== $(date +%T) $*"; }

# fetch <url> <file>: resumable download; gh <name> <url> <file>: a GitHub file, from $GH_ASSETS when there
fetch() { curl -fSL --retry 5 -C - -o "$2" "$1"; }
gh() { if [ -n "${GH_ASSETS:-}" ] && [ -f "$GH_ASSETS/$1" ]; then cp "$GH_ASSETS/$1" "$3"; else "${GH_PROXY_ENV[@]}" curl -fSL --retry 5 -C - -o "$3" "$2"; fi; }
# big <url> <dir> <name> [header]: aria2 with 16 connections, restarted with a fresh link when it stalls
big() {
  local h=(); [ -n "${4:-}" ] && h=(--header="$4")
  for i in 1 2 3 4 5 6 7 8; do
    aria2c -x16 -s16 -k16M --file-allocation=none --console-log-level=warn --summary-interval=60 \
      --lowest-speed-limit=200K --auto-file-renaming=false --allow-overwrite=true -c "${h[@]}" \
      -d "$2" -o "$3" "$1" && return 0
    echo "retry $i for $3"; sleep 5
  done
  return 1
}
declare -A LS_MD5=([dev-clean]=42e2234ba48799c1f50f24a7926300a1 [test-clean]=32fa31d27d2e1cad72775fee3f4849a9
                   [train-clean-360]=c0e676e450a7ff2f54aeade5171606fa)   # OpenSLR resources/12/md5sum.txt
librispeech() {  # librispeech <part>: download, check md5, unpack into $C/LibriSpeech/<part>
  [ -f "$C/LibriSpeech/$1.done" ] && return 0
  local url="$HF/datasets/k2-fsa/LibriSpeech/resolve/main/$1.tar.gz"
  [ "${LIBRISPEECH:-hf}" = openslr ] && url="$OPENSLR/resources/12/$1.tar.gz"
  big "$url" "$C" "$1.tar.gz"
  (cd "$C" && echo "${LS_MD5[$1]}  $1.tar.gz" | md5sum -c -)
  tar -xzf "$C/$1.tar.gz" -C "$C" && rm "$C/$1.tar.gz" && touch "$C/LibriSpeech/$1.done"
}
voxceleb() {  # VoxCeleb1 dev + test into $C/voxceleb1/wav
  [ -f "$C/voxceleb1/.done" ] && return 0
  local tok=${HF_TOKEN:-$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}
  [ -n "$tok" ] || { echo "no Hugging Face login: huggingface-cli login, or set HF_TOKEN"; return 1; }
  mkdir -p "$C/voxceleb1/vox1"
  for f in vox1_test_wav.zip vox1_dev_wav.zip vox1_meta.csv; do
    big "$HF/datasets/ProgramComputer/voxceleb/resolve/main/vox1/$f" "$C/voxceleb1/vox1" "$f" "Authorization: Bearer $tok"
  done
  for z in vox1_test_wav.zip vox1_dev_wav.zip; do unzip -q -n "$C/voxceleb1/vox1/$z" -d "$C/voxceleb1"; done
  touch "$C/voxceleb1/.done"
}

# 0. tools, disk speed
command -v aria2c >/dev/null && command -v flac >/dev/null && command -v unzip >/dev/null && command -v tmux >/dev/null || {
  log "apt: aria2 flac unzip git tmux ffmpeg"
  apt-get update -q >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -q aria2 flac unzip git tmux ffmpeg >/dev/null; }
if [ "${DISK_MIN:-200}" != 0 ] && [ ! -f "$WORK/.disk-ok" ]; then
  w=$(dd if=/dev/zero of="$WORK/.ddtest" bs=4M count=256 oflag=direct 2>&1 | awk '/copied/{print int($(NF-1)) " " $NF}')
  r=$(dd if="$WORK/.ddtest" of=/dev/null bs=4M iflag=direct 2>&1 | awk '/copied/{print int($(NF-1)) " " $NF}')
  rm -f "$WORK/.ddtest"
  log "disk under $WORK: write $w, read $r"
  mbs() { case "$2" in GB/s) echo $(( $1 * 1000 ));; MB/s) echo "$1";; *) echo 0;; esac; }
  if [ "$(mbs $w)" -lt "${DISK_MIN:-200}" ] || [ "$(mbs $r)" -lt "${DISK_MIN:-200}" ]; then
    echo "disk slower than ${DISK_MIN:-200} MB/s: give this machine back (DISK_MIN=0 to go on anyway)"; exit 2
  fi
  touch "$WORK/.disk-ok"
fi

# 1. big downloads in the background (they need no Python environment)
log "background downloads: logs/dl_*.log"
( librispeech train-clean-360 ) > "$WORK/logs/dl_ls360.log" 2>&1 & P_LS=$!
P_VOX=
[ "${VOXCELEB:-yes}" = skip ] || { ( voxceleb ) > "$WORK/logs/dl_vox.log" 2>&1 & P_VOX=$!; }

# 2. VPC 2024 framework at the pinned commit, its own environment (Python 3.11, torch 2.0.1)
if [ ! -d "$VPC/.git" ]; then
  if [ -n "${GH_ASSETS:-}" ] && [ -f "$GH_ASSETS/vpc2024.tar" ]; then
    mkdir -p "$VPC" && tar -xf "$GH_ASSETS/vpc2024.tar" -C "$VPC" --strip-components=1
  else
    "${GH_PROXY_ENV[@]}" git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024.git "$VPC"
  fi
  chown -R "$(id -u):$(id -g)" "$VPC"
  git -C "$VPC" config core.autocrlf false
  git -C "$VPC" checkout -q $VPC_COMMIT && git -C "$VPC" reset -q --hard $VPC_COMMIT   # LF line endings
fi
if [ "$VPC" != "$WORK/vpc2024" ]; then   # bulky directories stay on the data disk
  for d in data exp corpora; do
    mkdir -p "$WORK/vpc_$d"
    [ -L "$VPC/$d" ] || { [ -d "$VPC/$d" ] && cp -a "$VPC/$d/." "$WORK/vpc_$d/" && rm -rf "$VPC/$d"; ln -s "$WORK/vpc_$d" "$VPC/$d"; }
  done
  ln -sfn "$VPC" "$WORK/vpc2024"   # so that <work>/vpc2024 works as the framework path everywhere
fi
if [ ! -f "$VPC/.done-python-requirements" ] || [ -f "$VPC/.done-python-requirements.pending" ]; then
  # 00_install.sh only for the environment and the CUDA toolkit: its torch (cu117) and requirements steps are
  # marked done beforehand and replaced below (cu118 build first, so requirements do not pull another torch)
  log "framework environment (00_install.sh: micromamba env, CUDA toolkit)"
  mkdir -p "$VPC/.micromamba"   # placed beforehand; NEW_INSTALL_TRIGGER=no keeps 00_install.sh from deleting it
  [ -x "$VPC/.micromamba/micromamba" ] || { gh micromamba-linux-64 \
    https://github.com/mamba-org/micromamba-releases/releases/download/1.5.1-0/micromamba-linux-64 "$VPC/.micromamba/micromamba"
    chmod +x "$VPC/.micromamba/micromamba"; }
  touch "$VPC/.done-pytorch" "$VPC/.done-python-requirements.pending"
  rm -f "$VPC/.done-python-requirements"
  cp "$VPC/requirements.txt" "$WORK/requirements.vpc.txt"
  : > "$VPC/requirements.txt"   # empty for 00_install.sh; restored right after
  (cd "$VPC" && NEW_INSTALL_TRIGGER=no bash 00_install.sh) > "$WORK/logs/00_install.log" 2>&1
  git -C "$VPC" checkout -q -- requirements.txt 00_install.sh
  (
    cd "$VPC" && set +u && source env.sh && set -u
    log "torch 2.0.1+cu118, framework requirements with pins"
    torch_src=(--index-url https://download.pytorch.org/whl/cu118)
    if [ -n "${TORCH_WHEELS:-}" ]; then
      # pip itself fetched from the Aliyun mirror at ~0.2 MB/s on Huawei Cloud (2026-10-21) while curl got
      # ~18 MB/s from the same files: fetch the three wheels with curl, check them against download.pytorch.org
      mkdir -p "$WORK/wheels"
      for whl in "${TORCH_WHL_SHA[@]}"; do
        f=${whl%%:*}; [ -f "$WORK/wheels/$f" ] || curl -fsS --retry 5 -o "$WORK/wheels/$f" "$TORCH_WHEELS/$f"
        echo "${whl##*:}  $WORK/wheels/$f" | sha256sum -c - >/dev/null || { echo "bad checksum: $f"; rm -f "$WORK/wheels/$f"; exit 1; }
      done
      torch_src=(-f "$WORK/wheels")
    fi
    [ -n "${GH_ASSETS:-}" ] && ls "$GH_ASSETS"/torch-2.0.1+cu118-*.whl >/dev/null 2>&1 && torch_src=(-f "$GH_ASSETS")
    pip install -q torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 "${torch_src[@]}"
    pip install -q -r requirements.txt "${PINS[@]}" Cython
  ) >> "$WORK/logs/00_install.log" 2>&1
  rm -f "$VPC/.done-python-requirements.pending"; touch "$VPC/.done-python-requirements"
  (cd "$VPC" && set +u && source env.sh && set -u && pip cache purge >/dev/null 2>&1; micromamba clean -a -y >/dev/null 2>&1) || true
fi

# 3. pins, torch for CUDA 11.8, our code and its extra dependencies, in the framework's environment
(
  cd "$VPC" && set +u && source env.sh && set -u
  python -c "import torch, sys; sys.exit(torch.version.cuda != '11.8')" || {
    echo "torch is not the CUDA 11.8 build; remove $VPC/.done-python-requirements and run again"; exit 1; }
  log "our code"
  whisper="openai-whisper @ git+https://github.com/openai/whisper@$WHISPER_COMMIT"
  [ -n "${GH_ASSETS:-}" ] && [ -f "$GH_ASSETS/whisper-$WHISPER_COMMIT.tar.gz" ] && whisper="$GH_ASSETS/whisper-$WHISPER_COMMIT.tar.gz"
  pip install -q -e "$CODE" "$whisper" piper-tts==1.8.0 jiwer fastapi uvicorn python-multipart "${PINS[@]}"
  GPU_CHECK=${GPU_CHECK:-yes} python - <<'PY'
import os, torch
import speechbrain, whisper, piper, a082            # noqa: F401
if os.environ["GPU_CHECK"] == "skip":
    print("environment ok without the GPU check:", torch.__version__)
else:
    assert torch.cuda.is_available(), "no CUDA"
    x = torch.randn(2, 16000, device="cuda")
    torch.stft(x, 400, return_complex=True)           # the cuFFT check
    print("environment ok:", torch.__version__, torch.cuda.get_device_name(0))
PY
)

# 4. evaluation lists (data.zip), libri_dev / libri_test rebuilt from LibriSpeech (no sftp password),
#    pretrained evaluation models, kNN-VC weights and code, Piper voice
if [ ! -d "$VPC/data/libri_dev_enrolls" ]; then
  gh data.zip https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/data.zip/data.zip "$VPC/.data.zip"
  (cd "$VPC" && unzip -q -o .data.zip)
fi
for part in dev-clean test-clean; do librispeech $part; done
(cd "$VPC" && set +u && source env.sh && set -u && python "$CODE/scripts/rebuild_libri.py" --vpc-root "$VPC" --librispeech "$C/LibriSpeech")
for m in asv_orig asr ser; do
  if [ ! -d "$VPC/exp/$m" ]; then
    [ -f "$VPC/.$m.zip" ] || gh $m.zip https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/pre_model.zip/$m.zip "$VPC/.$m.zip"
    (cd "$VPC" && unzip -q -n ".$m.zip")
  fi
done
HUB=${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints
mkdir -p "$HUB"
for f in WavLM-Large.pt prematch_g_02500000.pt; do
  [ -f "$HUB/$f" ] || gh $f https://github.com/bshall/knn-vc/releases/download/v0.1/$f "$HUB/$f"
done
if [ ! -f "$WORK/knn-vc/hubconf.py" ]; then   # official kNN-VC code, for scripts/check_knnvc_reference.py --hub-dir
  mkdir -p "$WORK/knn-vc"
  gh knn-vc.tar.gz https://github.com/bshall/knn-vc/archive/refs/heads/master.tar.gz "$WORK/knn-vc.tar.gz"
  tar -xzf "$WORK/knn-vc.tar.gz" -C "$WORK/knn-vc" --strip-components=1 && rm "$WORK/knn-vc.tar.gz"
fi
mkdir -p "$WORK/cache/piper"
for f in en_US-libritts_r-medium.onnx en_US-libritts_r-medium.onnx.json; do
  [ -f "$WORK/cache/piper/$f" ] || fetch "$HF/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/$f" "$WORK/cache/piper/$f"
done
(cd "$WORK/cache/piper" && sha256sum en_US-libritts_r-medium.onnx | grep -q 10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195 \
  || echo "WARNING: Piper voice differs from the file recorded on 2026-10-18")

# 5. IEMOCAP sentence audio from the Hugging Face copy, linked as data/IEMOCAP/wav (needs data.zip)
if [ ! -d "$VPC/data/IEMOCAP/wav/Session1" ]; then
  log "IEMOCAP"
  (cd "$VPC" && set +u && source env.sh && set -u && python "$CODE/scripts/prepare_iemocap_hf.py" \
     --vpc-root "$VPC" --out "$C/IEMOCAP_full_release" --parquet-dir "$C/iemocap_hf")
fi

# 6. wait for the big downloads; link train-clean-360 where the framework expects it
log "waiting for train-clean-360"
wait $P_LS || { echo "train-clean-360 failed: logs/dl_ls360.log"; exit 1; }
mkdir -p "$VPC/corpora"
[ -e "$VPC/corpora/LibriSpeech" ] || ln -s "$C/LibriSpeech" "$VPC/corpora/LibriSpeech"
test -f "$VPC/data/train-clean-360/wav.scp"
if [ -n "$P_VOX" ]; then
  log "waiting for VoxCeleb"
  wait $P_VOX || { echo "VoxCeleb failed: logs/dl_vox.log"; exit 1; }
fi

# 7. optional public demo targets: LibriTTS-R test-clean (CC BY 4.0)
if [ "${DEMO_DATA:-yes}" != skip ] && [ ! -d "$C/LibriTTS_R/test-clean" ]; then
  fetch "$OPENSLR/resources/141/test_clean.tar.gz" "$C/test_clean.tar.gz"
  tar -xzf "$C/test_clean.tar.gz" -C "$C" && rm "$C/test_clean.tar.gz"
fi

log "setup done: VPC at $VPC, corpora under $C"

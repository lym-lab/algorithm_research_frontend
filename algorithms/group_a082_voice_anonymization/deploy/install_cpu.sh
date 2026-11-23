#!/bin/bash
# Install the web demo on a CPU-only Linux server (Huawei Cloud ECS, Ubuntu 22.04/24.04) as a systemd service.
#
#   sudo bash deploy/install_cpu.sh /opt/a082 <directory holding the demo assets>
#
# NOTE: huggingface_hub is pinned to 0.25.2 on purpose. SpeechBrain 0.5.16 calls
# hf_hub_download(use_auth_token=...), which huggingface_hub >= 0.26 removed; installing
# the latest makes the speaker-similarity readout fail with
#   TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'
#
# The assets directory must contain (copied from the GPU machine or the network drive):
#   pool_demo/            matching sets of the demo targets (6 LibriSpeech test-clean speakers), built by a082.knnvc.pool
#   piper/en_US-libritts_r-medium.onnx{,.json}
#   wc/wc5.pt ... wc8.pt   (optional) WavLM-conversion checkpoints
#   table1.json, bonus.json (optional) results for the results page
#   samples/*.wav          (optional) example clips, public data only (no IEMOCAP)
#   SHA256SUMS             checksums of all of the above (`sha256sum -c` format); the install stops on a mismatch
#
# WHISPER_MODEL=medium.en makes the cascade use the paper's ASR model (about 3 GB more RAM; 8 GB servers only).
set -euo pipefail
DEST=${1:?usage: install_cpu.sh <install dir> <assets dir>}
ASSETS=${2:?usage: install_cpu.sh <install dir> <assets dir>}
CODE=$(cd "$(dirname "$0")/.." && pwd)

# The assets come with a SHA256SUMS file; refuse to install anything that does not match it (a wrong or
# truncated checkpoint otherwise only shows up as wrong audio). SKIP_ASSET_CHECK=1 skips this on purpose.
if [ "${SKIP_ASSET_CHECK:-0}" != 1 ]; then
  [ -f "$ASSETS/SHA256SUMS" ] || { echo "no $ASSETS/SHA256SUMS: use the asset package from the network drive"; exit 1; }
  (cd "$ASSETS" && sha256sum --quiet -c SHA256SUMS) || { echo "asset check failed: re-download the files listed above"; exit 1; }
  echo "assets match SHA256SUMS"
fi

apt-get update -y && apt-get install -y python3.11 python3.11-venv ffmpeg || apt-get install -y python3 python3-venv ffmpeg
PY=$(command -v python3.11 || command -v python3)
mkdir -p "$DEST"
rsync -a --delete --exclude .git --exclude cache --exclude work "$CODE/" "$DEST/code/"
rsync -a "$ASSETS/" "$DEST/assets/"
$PY -m venv "$DEST/venv"
"$DEST/venv/bin/pip" install --upgrade pip
"$DEST/venv/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch==2.0.1 torchaudio==2.0.2
"$DEST/venv/bin/pip" install -e "$DEST/code" "numpy<2" soundfile pyyaml tqdm matplotlib fastapi uvicorn python-multipart \
  speechbrain==0.5.16 "huggingface_hub==0.25.2" piper-tts==1.8.0 \
  "openai-whisper @ git+https://github.com/openai/whisper@86098128c0b4f24f0e2aa2994de830614b474227"

cat > "$DEST/config.yaml" <<EOF
device: cpu
pool: $DEST/assets/pool_demo
pool_min_frames: 3000
piper_model: $DEST/assets/piper/en_US-libritts_r-medium.onnx
whisper_model: ${WHISPER_MODEL:-base.en}   # medium.en (the paper's, ~3 GB in fp32) does not fit next to WavLM in 4 GB RAM
wavlm_conv:
  wc5: $DEST/assets/wc/wc5.pt
  wc6: $DEST/assets/wc/wc6.pt
  wc7: $DEST/assets/wc/wc7.pt
  wc8: $DEST/assets/wc/wc8.pt
speaker_encoder: speechbrain/spkrec-ecapa-voxceleb
speaker_encoder_dir: $DEST/assets/spk_ecapa
results_json: $DEST/assets/table1.json
bonus_json: $DEST/assets/bonus.json
samples_dir: $DEST/assets/samples
similarity_threshold: 0.0   # 0 = 不画参考竖线；该模型的余弦尺度随目标说话人变化，固定阈值会误导
EOF

id -u a082 >/dev/null 2>&1 || useradd --system --home "$DEST" --shell /usr/sbin/nologin a082
chown -R a082:a082 "$DEST"
sed "s#@DEST@#$DEST#g" "$CODE/deploy/a082-web.service" > /etc/systemd/system/a082-web.service
systemctl daemon-reload
systemctl enable a082-web
systemctl restart a082-web   # `enable --now` would leave an already running service on the old code
echo "installed; check with: systemctl status a082-web; curl -s localhost:8000/api/systems"
# With medium.en the process peaks at about 6.7 GiB (measured on 8 GB). Some cloud images ship vm.swappiness=0:
# the kernel then kills the service at a peak instead of moving idle model pages to swap (seen on 11-23).
if [ "$(cat /proc/sys/vm/swappiness)" = 0 ] && [ -n "$(swapon --noheadings)" ]; then
  echo "WARNING: vm.swappiness is 0, so the swap is not used and memory peaks kill the service."
  echo "         Set it to 10: put 'vm.swappiness = 10' in /etc/sysctl.conf, then run: sysctl -p"
fi

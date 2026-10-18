#!/bin/bash
# Install the web demo on a CPU-only Linux server (Huawei Cloud ECS, Ubuntu 22.04/24.04) as a systemd service.
#
#   sudo bash deploy/install_cpu.sh /opt/a082 <directory holding the demo assets>
#
# The assets directory must contain (copied from the GPU machine or the network drive):
#   pool_demo/            matching sets of the demo targets (LibriTTS-R), built by a082.knnvc.pool
#   piper/en_US-libritts_r-medium.onnx{,.json}
#   wc/wc5.pt ... wc8.pt   (optional) WavLM-conversion checkpoints
#   table1.json, bonus.json (optional) results for the results page
#   samples/*.wav          (optional) example clips, public data only (no IEMOCAP)
set -euo pipefail
DEST=${1:?usage: install_cpu.sh <install dir> <assets dir>}
ASSETS=${2:?usage: install_cpu.sh <install dir> <assets dir>}
CODE=$(cd "$(dirname "$0")/.." && pwd)

apt-get update -y && apt-get install -y python3.11 python3.11-venv ffmpeg || apt-get install -y python3 python3-venv ffmpeg
PY=$(command -v python3.11 || command -v python3)
mkdir -p "$DEST"
rsync -a --delete --exclude .git --exclude cache --exclude work "$CODE/" "$DEST/code/"
rsync -a "$ASSETS/" "$DEST/assets/"
$PY -m venv "$DEST/venv"
"$DEST/venv/bin/pip" install --upgrade pip
"$DEST/venv/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch==2.0.1 torchaudio==2.0.2
"$DEST/venv/bin/pip" install -e "$DEST/code" "numpy<2" soundfile pyyaml tqdm matplotlib fastapi uvicorn python-multipart \
  speechbrain==0.5.16 piper-tts==1.8.0 \
  "openai-whisper @ git+https://github.com/openai/whisper@86098128c0b4f24f0e2aa2994de830614b474227"

cat > "$DEST/config.yaml" <<EOF
device: cpu
pool: $DEST/assets/pool_demo
pool_min_frames: 3000
piper_model: $DEST/assets/piper/en_US-libritts_r-medium.onnx
whisper_model: medium.en
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
EOF

id -u a082 >/dev/null 2>&1 || useradd --system --home "$DEST" --shell /usr/sbin/nologin a082
chown -R a082:a082 "$DEST"
sed "s#@DEST@#$DEST#g" "$CODE/deploy/a082-web.service" > /etc/systemd/system/a082-web.service
systemctl daemon-reload
systemctl enable --now a082-web
echo "installed; check with: systemctl status a082-web; curl -s localhost:8000/api/systems"

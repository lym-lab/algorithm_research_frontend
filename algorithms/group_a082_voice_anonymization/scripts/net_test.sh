#!/bin/bash
# Download speed from every source cloud_setup.sh uses, to choose between direct access, the platform's
# proxy (/etc/network_turbo on AutoDL) and mirrors. Each test reads at most 200 MB for at most 20 s.
#
#   bash scripts/net_test.sh            # direct, then through /etc/network_turbo if it exists
set -u

URLS=(
  "GitHub release (VPC data.zip)|https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/data.zip/data.zip"
  "GitHub release (WavLM-Large)|https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt"
  "GitHub clone (VPC repo)|https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/archive/refs/heads/main.zip"
  "Hugging Face (IEMOCAP parquet)|https://huggingface.co/datasets/AbstractTTS/IEMOCAP/resolve/main/data/train-00000-of-00003.parquet"
  "Hugging Face (Piper voice)|https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx"
  "OpenSLR www|https://www.openslr.org/resources/12/train-clean-360.tar.gz"
  "OpenSLR CN mirror|https://openslr.magicdatatech.com/resources/12/train-clean-360.tar.gz"
  "OpenSLR EU mirror (trmal)|https://openslr.trmal.net/resources/12/train-clean-360.tar.gz"
  "PyPI (torch wheel index)|https://pypi.org/simple/torch/"
)

probe() {
  for entry in "${URLS[@]}"; do
    name=${entry%%|*}; url=${entry#*|}
    r=$(curl -sL -o /dev/null -r 0-209715199 --max-time 20 -w "%{http_code} %{speed_download}" "$url" 2>/dev/null || true)
    code=${r%% *}; speed=${r##* }
    printf "  %-34s HTTP %-4s %8.2f MB/s\n" "$name" "${code:-000}" "$(awk -v s="${speed:-0}" 'BEGIN{print s/1048576}')"
  done
}

echo "== direct"
( unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; probe )
if [ -f /etc/network_turbo ]; then
  echo "== through /etc/network_turbo"
  ( source /etc/network_turbo >/dev/null; probe )
fi

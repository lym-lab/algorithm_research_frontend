#!/bin/bash
# Load sampler beside the queue: every 30 s one CSV line with CPU use (all cores, from /proc/stat), the 1-minute
# load, memory in use, and per GPU the utilisation and memory in use. Read by tools/huawei/hw_watch.py (averages
# over the last 10 minutes) and by hand to decide how many jobs a GPU can take.
#   tmux new -d -s metrics "bash scripts/metrics.sh <work dir>"      -> <work dir>/logs/metrics.csv
W=${1:?usage: metrics.sh <work dir>}
OUT=$W/logs/metrics.csv
[ -f "$OUT" ] || echo "time,cpu_pct,load1,ncpu,mem_used_pct,gpu_util,gpu_mem_used_mb" > "$OUT"
read -r _ a b c d e f g h _ < /proc/stat
prev_total=$((a + b + c + d + e + f + g + h)); prev_idle=$((d + e))
while true; do
  sleep 30
  read -r _ a b c d e f g h _ < /proc/stat
  total=$((a + b + c + d + e + f + g + h)); idle=$((d + e))
  cpu=$(( 100 * ( (total - prev_total) - (idle - prev_idle) ) / (total - prev_total) ))
  prev_total=$total; prev_idle=$idle
  load=$(cut -d' ' -f1 /proc/loadavg)
  mem=$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%d", 100*(t-a)/t}' /proc/meminfo)
  gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd' ')
  gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd' ')
  echo "$(date '+%F %T'),$cpu,$load,$(nproc),$mem,$gu,$gm" >> "$OUT"
done

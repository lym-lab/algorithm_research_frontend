#!/bin/bash
# Batch runs on one machine with several GPUs: one "lane" per GPU, each a fixed sequence of tasks that
# waits for the outputs it needs (marker files in <work>/done/). Start each lane in its own tmux session:
#
#   tmux new -d -s lane1 "bash scripts/lanes.sh /root/autodl-tmp/a082 1"
#
# Wave 1 (docs/REPRODUCE.md §2, §4, §5, §7 preparation):
#   lane 0  cascade ID 4 (3 processes) -> evaluation ID 4
#   lane 1  pool shards -> kNN ID 1 (2 processes) -> evaluation ID 1 -> kNN ID 2 -> evaluation ID 2
#   lane 2  pool shards -> kNN ID 1M (2 processes) -> evaluation ID 1M
#   lane 3  pool shards, then the single-process pass that fills what the shards left (out of memory on
#           long VoxCeleb files) -> pool.done -> WavLM conversion: target features, speaker classifier
# Each GPU's lane gets an equal share of the CPU quota (A082_THREADS), split among its processes.
# Logs: <work>/logs/lane<i>.log and <work>/logs/<task>*.log. Re-running a lane skips finished tasks.
set -uo pipefail
W=${1:?usage: lanes.sh <work dir> <lane>}
LANE=${2:?usage: lanes.sh <work dir> <lane>}
R=$W/vpc2024
CODE=$(cd "$(dirname "$0")/.." && pwd)
cd "$R" && set +u && source env.sh && set -u
cd "$CODE"
export CUDA_VISIBLE_DEVICES=$LANE
NGPU=$(nvidia-smi -L | wc -l)
CPUS=$(CUDA_VISIBLE_DEVICES= python -c "import a082; print(a082.cpu_limit())")
SHARE=$(( CPUS / NGPU ))
mkdir -p "$W/done" "$W/logs"
log() { echo "== $(date '+%F %T') lane $LANE: $*"; }
done_() { [ -f "$W/done/$1" ]; }
mark() { touch "$W/done/$1"; log "done $1"; }
waitfor() { for m in "$@"; do until done_ "$m"; do sleep 60; done; done; }

# parallel <name> <n> <command...>: n processes of the command with --shard i/n, each with SHARE/n threads
parallel() {
  local name=$1 n=$2; shift 2
  local pids=() i rc=0
  for i in $(seq 0 $((n - 1))); do
    A082_THREADS=$(( SHARE / n )) "$@" --shard $i/$n > "$W/logs/${name}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || rc=1; done
  return $rc
}

pool_shards() {  # 6 shards over lanes 1-3, two per GPU; failures (out of memory) are left to the fill pass
  done_ pool.done && return 0
  for i in 0 1; do
    local s=$(( (LANE - 1) * 2 + i ))
    A082_THREADS=$(( SHARE / 2 )) python -m a082.knnvc.pool build "$W/pool_vox1.json" "$W/cache/pool_vox1" \
      --shard $s/6 > "$W/logs/pool6_$s.log" 2>&1 &
  done
  wait
  mark pool.lane$LANE
}

knn() {  # knn <config name> <processes>
  local s=$1 n=$2
  done_ "anon_$s" && return 0
  parallel "anon_$s" $n python -m a082.knnvc.anonymize --config configs/$s.yaml --pool "$W/cache/pool_vox1" \
    --vpc-root "$R" --sets all || { log "anon_$s failed"; return 1; }
  python -m a082.knnvc.anonymize --config configs/$s.yaml --vpc-root "$R" --sets all --finalize-only \
    > "$W/logs/anon_${s}_finalize.log" 2>&1 || { log "anon_$s finalize failed"; return 1; }
  mark "anon_$s"
}

evaluate() {  # evaluate <suffix> [row]
  local suf=$1 row=${2:-}
  done_ "eval$suf" && return 0
  A082_THREADS=$SHARE python -m a082.eval.vpc run --vpc-root "$R" --suffix "$suf" --workspace auto \
    > "$W/logs/eval$suf.log" 2>&1 || { log "eval$suf failed"; return 1; }
  python -m a082.eval.vpc collect --vpc-root "$R" --suffix "$suf" ${row:+--row $row} > "$W/logs/result$suf.txt" 2>&1
  mark "eval$suf"
}

log "start on GPU $LANE, $SHARE CPU threads"
case $LANE in
  0)
    if ! done_ anon_tts; then
      parallel anon_tts 3 python -m a082.cascade.anonymize --vpc-root "$R" \
        --piper-model "$W/cache/piper/en_US-libritts_r-medium.onnx" --transcripts "$W/transcripts" --sets all \
        && python -m a082.cascade.anonymize --vpc-root "$R" --piper-model "$W/cache/piper/en_US-libritts_r-medium.onnx" \
             --transcripts "$W/transcripts" --sets all --finalize-only > "$W/logs/anon_tts_finalize.log" 2>&1 \
        && mark anon_tts || { log "anon_tts failed"; exit 1; }
      python scripts/whisper_wer.py --vpc-root "$R" --transcripts "$W/transcripts" > "$W/logs/whisper_wer.txt" 2>&1
    fi
    evaluate _tts 4
    ;;
  1)
    pool_shards; waitfor pool.done
    knn knn1 2 && evaluate _knn1 1
    knn knn2 2 && evaluate _knn2 2
    ;;
  2)
    pool_shards; waitfor pool.done
    knn knn1m 2 && evaluate _knn1m
    ;;
  3)
    pool_shards; waitfor pool.lane1 pool.lane2
    if ! done_ pool.done; then
      A082_THREADS=$SHARE python -m a082.knnvc.pool build "$W/pool_vox1.json" "$W/cache/pool_vox1" \
        > "$W/logs/pool_fill.log" 2>&1 || { log "pool fill failed"; exit 1; }
      n=$(ls "$W"/cache/pool_vox1/*.frames | wc -l)
      [ "$n" -eq "$(python -c "import json; print(len(json.load(open('$W/pool_vox1.json'))))")" ] \
        || { log "pool incomplete: $n"; exit 1; }
      mark pool.done
    fi
    done_ wc_target || { python -m a082.wavlm_conv.train prepare-target --manifest "$W/pool_vox1.json" \
      --pool "$W/cache/pool_vox1" --out "$W/wc/target.pt" > "$W/logs/wc_target.log" 2>&1 && mark wc_target; }
    done_ wc_spk || { A082_THREADS=$SHARE python -m a082.wavlm_conv.train pretrain-spk --vpc-root "$R" \
      --out "$W/wc/spk.pt" > "$W/logs/wc_spk.log" 2>&1 && mark wc_spk; }
    ;;
esac
log "lane finished"

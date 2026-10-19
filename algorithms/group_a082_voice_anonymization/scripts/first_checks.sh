#!/bin/bash
# First checks on a new GPU machine, after scripts/cloud_setup.sh: correctness first, then speed.
#
#   bash scripts/first_checks.sh <work dir>        e.g. /root/autodl-tmp/a082   (about 1-1.5 h on one GPU)
#
#   1. kNN-VC against the official implementation (bshall/knn-vc), one real utterance
#   2. row 0 (original speech) through the official evaluation: EER and WER must match
#   3. speed of each kind of work on a small share, for scheduling
#      kNN-VC (row 1 config, 1/20 of libri_dev_trials_f), cascade (row 4, same share), WavLM conversion
#      training (200 steps of the row 5 config); the attacker ASV training time is read from the first
#      full evaluation instead (it is logged as "ASV training time").
# Speed runs use throwaway suffixes (_speedknn, _speedtts) and are removed at the end. The VoxCeleb1 test
# speakers' matching sets are built into the real pool cache, so the full pool build reuses them.
# Everything is logged to <work>/logs/first_checks.log; the numbers go to <work>/logs/first_checks.txt.
set -uo pipefail
W=${1:?usage: first_checks.sh <work dir>}
R=$W/vpc2024
CODE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$W/logs/first_checks.txt
cd "$R" && set +u && source env.sh && set -u
cd "$CODE"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
: > "$OUT"
say() { echo "$*" | tee -a "$OUT"; }
now() { date +%s; }
rtf() {  # rtf <set> <suffix> <seconds>: audio seconds of the utterances written, divided by wall time
  python - "$R" "$1" "$2" "$3" <<'PY'
import sys
from pathlib import Path
r, s, suf, sec = Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
dur = dict(l.split() for l in (r / "data" / s / "utt2dur").read_text().splitlines())
done = [p.stem for p in (r / "data" / f"{s}{suf}" / "wav").glob("*.wav")]
a = sum(float(dur[u]) for u in done if u in dur)
print(f"{len(done)} utterances, {a / 3600:.3f} h of audio in {sec / 3600:.3f} h: {a / max(sec, 1):.1f} x real time")
PY
}

say "== $(date '+%F %T') on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 1. kNN-VC reference check
SRC=$(ls "$R"/data/libri_dev/wav/1272-128104-0000/*.wav | head -1)
TGT=$(ls "$W"/corpora/voxceleb1/wav/id10270/*/*.wav | head -5)
python scripts/check_knnvc_reference.py --hub-dir "$W/knn-vc" --source "$SRC" --target $TGT 2>&1 | tail -3 | tee -a "$OUT"
say "1. kNN reference check: exit ${PIPESTATUS[0]} (0 = equal within 1e-3)"

# 2. row 0
t=$(now)
python -m a082.eval.vpc identity --vpc-root "$R" --sets libri_dev_enrolls,libri_dev_trials_f,libri_dev_trials_m,libri_test_enrolls,libri_test_trials_f,libri_test_trials_m,IEMOCAP_dev,IEMOCAP_test
python -m a082.eval.vpc run --vpc-root "$R" --suffix _orig --stages pre
say "2. row 0 evaluation (eval_pre): $(( $(now) - t )) s"
python -m a082.eval.vpc collect --vpc-root "$R" --suffix _orig --row 0 | tee -a "$OUT"
say "   paper: EER 10.511 / 0.931 / 8.761 / 0.418, WER 1.807 / 1.844 (each within 0.05)"

# 3a. VoxCeleb1 test speakers into the pool cache
python -m a082.knnvc.pool scan-voxceleb "$W/corpora/voxceleb1/wav" "$W/pool_vox1.json"
python - "$W" <<'PY'
import json, sys
from pathlib import Path
w = Path(sys.argv[1]); m = json.loads((w / "pool_vox1.json").read_text())
meta = (w / "corpora/voxceleb1/vox1/vox1_meta.csv").read_text().splitlines()
test = {l.split("\t")[0] for l in meta[1:] if l.strip().endswith("test")}
sub = {k: v for k, v in m.items() if k in test} or dict(list(m.items())[:40])
(w / "pool_speed.json").write_text(json.dumps(sub))
print(f"{len(sub)} test speakers")
PY
t=$(now)
python -m a082.knnvc.pool build "$W/pool_speed.json" "$W/cache/pool_vox1"
say "3a. pool build, VoxCeleb1 test speakers: $(( $(now) - t )) s"
mkdir -p "$W/cache/pool_speed"
for f in "$W"/cache/pool_vox1/*.pt; do   # the speed run uses only these speakers
  ln -sf "$f" "$W/cache/pool_speed/"; ln -sf "${f%.pt}.frames" "$W/cache/pool_speed/"
done

# 3b. kNN-VC speed
sed 's/^suffix:.*/suffix: _speedknn/' configs/knn1.yaml > "$W/knn_speed.yaml"
t=$(now)
python -m a082.knnvc.anonymize --config "$W/knn_speed.yaml" --pool "$W/cache/pool_speed" --vpc-root "$R" \
  --sets libri_dev_trials_f --shard 0/20 > "$W/logs/speed_knn.log" 2>&1
say "3b. kNN-VC (row 1): $(rtf libri_dev_trials_f _speedknn $(( $(now) - t )))"

# 3c. cascade speed (Whisper medium.en is downloaded on first use)
t=$(now)
python -m a082.cascade.anonymize --vpc-root "$R" --piper-model "$W/cache/piper/en_US-libritts_r-medium.onnx" \
  --suffix _speedtts --transcripts "$W/work/speed_transcripts" --sets libri_dev_trials_f --shard 0/20 \
  > "$W/logs/speed_tts.log" 2>&1
say "3c. cascade (row 4): $(rtf libri_dev_trials_f _speedtts $(( $(now) - t ))) (includes the Whisper download)"

# 3d. WavLM conversion training speed and memory
SPK=$(python -c "import json; m=json.load(open('$W/pool_speed.json')); print(max(m, key=lambda k: len(m[k])))")
python -m a082.wavlm_conv.train prepare-target --manifest "$W/pool_speed.json" --pool "$W/cache/pool_speed" \
  --out "$W/wc/speed_target.pt" --speaker "$SPK" > "$W/logs/speed_wc.log" 2>&1
sed 's/^steps:.*/steps: 200/; s/^name:.*/name: wc_speed/' configs/wc5.yaml > "$W/wc_speed.yaml"
python -m a082.wavlm_conv.train train --config "$W/wc_speed.yaml" --vpc-root "$R" --target "$W/wc/speed_target.pt" \
  --manifest "$W/pool_speed.json" --out "$W/wc/speed" --log-every 50 --save-every 1000 >> "$W/logs/speed_wc.log" 2>&1 &
PID=$!; PEAK=0
while kill -0 $PID 2>/dev/null; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -gt "$PEAK" ] && PEAK=$m; sleep 5
done
say "3d. WavLM conversion training (row 5 config): $(grep -o '[0-9.]* it/s' "$W/logs/speed_wc.log" | tail -1), peak GPU memory ${PEAK} MiB"
grep -E "Error|error" "$W/logs/speed_wc.log" | tail -2 | tee -a "$OUT"

# clean up the throwaway outputs
rm -rf "$R"/data/*_speedknn "$R"/data/*_speedtts "$W/cache/pool_speed" "$W/wc/speed" "$W/wc/speed_target.pt"
say "== done $(date '+%F %T'); full log: logs/first_checks.log"

#!/bin/bash
# Safety net beside scripts/batch_queue.py: shut the machine down when no queue and no computation has run for
# 20 minutes (e.g. the queue died), so an unattended machine does not keep billing. The queue shuts down by
# itself when it finishes; this covers the cases where it cannot. `touch <work>/no_shutdown` disables both.
#   tmux new -d -s guard "bash scripts/guard.sh /root/autodl-tmp/a082"
W=${1:?usage: guard.sh <work dir>}
idle=0
while true; do
  # anchored on the interpreter: the tmux server's command line is the `tmux new ... batch_queue.py` that started it
  if pgrep -f '^[^ ]*python[^ ]* scripts/batch_queue[.]py' >/dev/null || pgrep -f '^python -m a082[.]' >/dev/null \
     || pgrep -f 'run_evaluation[.]py' >/dev/null; then
    idle=0
  else
    idle=$((idle + 1))
  fi
  if [ $idle -ge 4 ] && [ ! -f "$W/no_shutdown" ]; then
    echo "$(date '+%F %T') guard: nothing ran for 20 min, shutting down" >> "$W/logs/queue.log"
    sync; shutdown; exit 0
  fi
  sleep 300
done

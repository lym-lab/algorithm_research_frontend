"""Batch counters for display only; no tensors or random-number operations."""
import json
import os
import time
from pathlib import Path


class EpochProgress:
    def __init__(self, output, epoch, epochs):
        self.path = Path(output)/'progress.json'
        self.epoch, self.epochs = epoch + 1, epochs
        self.started = time.monotonic()
        self.last_write = float('-inf')

    def update(self, phase, completed, total, force=False):
        now = time.monotonic()
        if not force and completed != total and now-self.last_write < 2:
            return
        payload = dict(pid=os.getpid(), epoch=self.epoch, epochs=self.epochs,
                       phase=phase, completed=completed, total=total,
                       percent=100*completed/total if total else 0,
                       elapsed_seconds=now-self.started, updated=time.time())
        temporary = self.path.with_suffix('.json.tmp')
        self.last_write = now
        try:
            temporary.write_text(json.dumps(payload), encoding='utf-8')
            os.replace(temporary, self.path)
        except OSError:
            # A dashboard read can briefly lock the destination on Windows.
            # Display failures must not interrupt training; retry next update.
            pass

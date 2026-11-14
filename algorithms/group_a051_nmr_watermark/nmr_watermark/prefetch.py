"""Prepare one CPU batch ahead, preserving sampler order and RNG state."""
from concurrent.futures import ThreadPoolExecutor


def prefetch_batches(loader):
    # Construct the iterator on the caller thread (DataLoader seeds it here).
    iterator = iter(loader)
    end = object()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(next, iterator, end)
        while True:
            batch = pending.result()
            if batch is end:
                return
            pending = pool.submit(next, iterator, end)
            yield batch

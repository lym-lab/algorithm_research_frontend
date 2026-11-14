import hashlib
import json
import sys
from pathlib import Path


def verify(registry):
    registry = Path(registry)
    entries = json.loads(registry.read_text())
    if set(entries) != {'MSE256','NMR256','MSE512','NMR512'}:
        raise ValueError('Final deployment requires all four model entries')
    for name, entry in entries.items():
        path = registry.parent/entry['checkpoint']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get('sha256'):
            raise ValueError(f'Weight integrity check failed: {name}')
    return True


if __name__ == '__main__':
    verify(sys.argv[1])
    print('All four weight checksums verified')

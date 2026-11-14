"""Package four selected inference states without optimizer or absolute paths."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('selected', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
selected = json.loads(args.selected.read_text())
if set(selected) != {'MSE256','NMR256','MSE512','NMR512'}:
    raise ValueError('All four paper models must be selected before release')
args.output.mkdir(parents=True, exist_ok=True)
result = {}
for name, record in selected.items():
    source = Path(record['checkpoint'])
    if not source.is_absolute():
        source = args.selected.parent/source
    ckpt = torch.load(source, map_location='cpu', weights_only=False)
    if ckpt['config'].get('pilot', False):
        raise ValueError('Do not release a pilot model as a paper result')
    target = args.output/f'{name}.pt'
    torch.save({key:ckpt[key] for key in ['model','config','metrics','epoch']}, target)
    result[name] = {'checkpoint':target.name, 'metrics':ckpt['metrics'], 'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(args.output/'selected.json').write_text(json.dumps(result, indent=2))

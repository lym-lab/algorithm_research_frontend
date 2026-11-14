"""Copy an explicit source/evidence allowlist into the user-authorized Fork.

No raw datasets, checkpoints, private listening responses, or project report.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repository', type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    target = args.repository / 'algorithms/group_a051_nmr_watermark'
    target.mkdir(parents=True, exist_ok=True)
    files = [*Path('nmr_watermark').glob('*.py'), *Path('tests').glob('*.py')]
    files += [p for p in Path('scripts').glob('*.py')
              if p.name not in {'build_report.py', 'report_body_docx.py'}]
    files += list(Path('scripts').glob('*.m'))
    files.append(Path('scripts/progress_charts.js'))
    files += [Path('pyproject.toml'), Path('requirements-windows.lock')]
    files += [Path('docs') / name for name in [
        'reproduction.md', 'audit-20261021.md', 'deployment-and-troubleshooting.md',
        'presentation-and-listening.md', 'completion-20261030.md']]
    files += [Path('outputs') / name for name in [
        'frozen_config.json', 'reference_validation.json', 'final_training_audit.json']]
    files += list(Path('outputs/final_analysis').glob('*'))
    files += [p for p in Path('outputs/case').rglob('*') if p.is_file() and p.suffix != '.wav']
    files += [Path('outputs/paper_audit_20261021') / name for name in ['contract.json', 'training_runtime.json']]
    for folder in ['cloud_validation_20261101', 'cloud_reboot_validation_20261101']:
        files += [p for p in (Path('outputs')/folder).glob('*.json') if p.name != 'reboot-availability.json']
    for name in ['MSE256', 'NMR256', 'MSE512', 'NMR512']:
        folder = Path('outputs/evaluation') / name
        files += [folder / file for file in ['metrics.csv', 'summary.json', 'peaq.csv'] if (folder/file).exists()]
        files += list(folder.glob('peaq.csv.undefined_*.txt'))
    for file in files:
        dest = target / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        if file.suffix == '.json':
            # Make local path references portable; weights remain external.
            content = file.read_text(encoding='utf-8')
            for prefix in [str(root), str(root).replace('\\', '/')]:
                content = content.replace(json.dumps(prefix)[1:-1], '.')
            dest.write_text(content, encoding='utf-8')
        else:
            shutil.copyfile(file, dest)
    samples = args.repository / 'public/samples/group_a051_nmr_watermark'
    samples.mkdir(parents=True, exist_ok=True)
    for file in Path('outputs/service_samples').glob('*'):
        if file.is_file():
            shutil.copyfile(file, samples / file.name)
    # Remove only the RWC excerpts copied by the initial packaging revision.
    for name in ['MSE256.wav', 'NMR256.wav', 'original.wav']:
        (target/'outputs/case'/name).unlink(missing_ok=True)
    records = {str(p.relative_to(target)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in target.rglob('*') if p.is_file() and p.name != 'source-manifest.json'}
    (target/'source-manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Packaged {len(records)} files; weights and datasets excluded')


if __name__ == '__main__':
    main()

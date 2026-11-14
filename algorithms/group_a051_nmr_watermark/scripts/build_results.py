"""Generate report-ready tables from real final validation/test records."""
import csv
import json
from pathlib import Path
from nmr_watermark.evaluation import figures
from scripts.analyze_final import collect


def main():
    root = Path('outputs/evaluation')
    selected = json.loads(Path('runs/paper/selected.json').read_text())
    collected = {r['name']: r for r in collect(root, selected)}
    rows = []
    for name in ['MSE256','NMR256','MSE512','NMR512']:
        summary = json.loads((root/name/'summary.json').read_text())
        if summary['config'].get('pilot') or summary['samples'] != 3000:
            raise ValueError('Only complete final RWC evaluations may enter report tables')
        record = collected[name]
        val = selected[name]['metrics']
        row = dict(model=name, validation_ber=val['ber'], validation_snr=val['snr'],
                   validation_nmr_db=val['nmr_db'], test_ber=summary['ber'],
                   test_snr=summary['snr'], test_nmr_db=summary['nmr_db'],
                   test_odg=record['odg'], test_samples=3000,
                   peaq_common_samples=record['peaq_valid_samples'])
        rows.append(row)
        summary['peaq'] = 'completed'
        summary['odg_mean'] = row['test_odg']
        summary['peaq_common_samples'] = record['peaq_valid_samples']
        summary['peaq_undefined_indices'] = record['peaq_undefined_indices']
        (root/name/'summary.json').write_text(json.dumps(summary, indent=2))
    with (root/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['|模型|验证BER|验证SNR|验证NMR|测试BER|测试SNR|测试NMR|测试ODG|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        keys = ['validation_ber', 'validation_snr', 'validation_nmr_db', 'test_ber',
                'test_snr', 'test_nmr_db', 'test_odg']
        lines.append('|'+r['model']+'|'+'|'.join(f'{r[k]:.6g}' for k in keys)+'|')
    lines.append(f'\nBER/SNR/NMR每模型3000段；ODG采用四模型共有的{rows[0]["peaq_common_samples"]}个有限结果，不填补未定义分数。')
    (root/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    figures(root)
    from scripts.analyze_final import main as analyze_final
    analyze_final()


if __name__ == '__main__':
    main()

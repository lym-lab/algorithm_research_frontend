"""Strict post-training analysis. No writes to training or source evaluation data."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

NAMES = ['MSE256', 'NMR256', 'MSE512', 'NMR512']
PAPER = [(0.00030, 25.8, 13.7), (0.00025, 22.4, -18.0),
         (0.11100, 25.0, 13.7), (0.09700, 18.6, -13.3)]


def indexed(path, columns, allow_undefined=False):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [int(r['index']) for r in rows]
    if len(ids) != 3000 or set(ids) != set(range(3000)):
        raise ValueError(f'{path}: require exactly 3000 unique sample indices')
    rows.sort(key=lambda row: int(row['index']))
    for row in rows:
        if not all(np.isfinite(float(row[k])) for k in columns):
            diagnostic = Path(f'{path}.undefined_{row["index"]}.txt')
            if not (allow_undefined and columns == ['odg'] and np.isnan(float(row['odg']))
                    and row.get('status') == 'undefined_reference_output'
                    and diagnostic.is_file() and diagnostic.stat().st_size > 0):
                raise ValueError(f'{path}: nonfinite metric without a documented reference failure')
    return rows


def collect(root, selected):
    records, manifest, tracks = [], None, None
    if set(selected) != set(NAMES):
        raise ValueError('Require exactly four selected models')
    peaq_rows = {name: indexed(Path(root)/name/'peaq.csv', ['odg'], allow_undefined=True) for name in NAMES}
    common = np.logical_and.reduce([
        np.array([np.isfinite(float(row['odg'])) for row in peaq_rows[name]]) for name in NAMES])
    if not common.any():
        raise ValueError('No common finite PEAQ samples across the four models')
    for name, paper in zip(NAMES, PAPER):
        directory = Path(root)/name
        summary = json.loads((directory/'summary.json').read_text())
        if summary['config'].get('pilot') or summary['samples'] != 3000:
            raise ValueError('Incomplete or pilot evaluation')
        if (summary['config'].get('bits') != int(name[-3:])
                or summary['config'].get('loss') != name[:-3].lower()):
            raise ValueError('Evaluation model identity does not match its directory')
        identity = summary['manifest_sha256']
        if not identity or (manifest is not None and manifest != identity):
            raise ValueError('Four models must use the same manifest')
        manifest = identity
        metrics = indexed(directory/'metrics.csv', ['ber', 'snr', 'nmr'])
        odg = peaq_rows[name]
        ids = [r['track_id'] for r in metrics]
        if tracks is not None and tracks != ids:
            raise ValueError('Track/index alignment differs across models')
        tracks = ids
        from collections import Counter
        counts = Counter(ids)
        if len(counts) != 100 or set(counts.values()) != {30}:
            raise ValueError('Expected 100 tracks with 30 samples each')
        ber = np.array([float(r['ber']) for r in metrics])
        nmr = np.array([float(r['nmr']) for r in metrics])
        if (ber < 0).any() or (ber > 1).any() or (nmr < 0).any():
            raise ValueError('Invalid BER or linear NMR')
        val = selected[name]['metrics']
        if not all(np.isfinite(val[k]) for k in ('ber', 'snr', 'nmr_db')):
            raise ValueError('Invalid validation metrics')
        snr = np.array([float(r['snr']) for r in metrics])
        odg_values = np.array([float(r['odg']) for r in odg])[common]
        aggregate = dict(ber=float(ber.mean()), snr=float(snr.mean()),
                         nmr_db=float(10*np.log10(max(nmr.mean(), 1e-30))))
        if any(not np.isclose(summary[k], aggregate[k], rtol=1e-5, atol=1e-7) for k in aggregate):
            raise ValueError('Summary disagrees with per-sample CSV')
        records.append(dict(name=name, paper=paper, validation=val, test=aggregate,
                            odg=float(odg_values.mean()), snr_values=snr, odg_values=odg_values,
                            peaq_valid_samples=int(common.sum()),
                            peaq_excluded_indices=np.flatnonzero(~common).tolist(),
                            peaq_undefined_indices=[int(r['index']) for r in odg if not np.isfinite(float(r['odg']))]))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs/evaluation')
    p.add_argument('--selected', default='runs/paper/selected.json')
    p.add_argument('--output', default='outputs/final_analysis')
    args = p.parse_args()
    records = collect(args.root, json.loads(Path(args.selected).read_text()))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        row = {'model': r['name']}
        for key, original in zip(('ber', 'snr', 'nmr_db'), r['paper']):
            row.update({f'paper_validation_{key}': original, f'validation_{key}': r['validation'][key],
                        f'delta_validation_{key}': r['validation'][key]-original,
                        f'rwc_{key}': r['test'][key]})
        row['rwc_odg_mean'] = r['odg']
        row['rwc_odg_common_samples'] = r['peaq_valid_samples']
        rows.append(row)
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field, label in [('snr_values', 'SNR (dB)'), ('odg_values', 'PEAQ ODG')]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot([r[field] for r in records], tick_labels=NAMES, whis=(5, 95), showfliers=False)
        ax.set_ylabel(label); fig.tight_layout()
        fig.savefig(out/(field+'.png'), dpi=180); plt.close(fig)
    (out/'peaq_cohort.json').write_text(json.dumps({
        'policy': 'All 3000 samples attempted; compare ODG only on the common finite subset. No imputation.',
        'common_samples': records[0]['peaq_valid_samples'],
        'excluded_indices': records[0]['peaq_excluded_indices'],
        'undefined_by_model': {r['name']: r['peaq_undefined_indices'] for r in records}}, indent=2))
    (out/'README.md').write_text('仅使用真实结果。paper_validation列来自论文表1；validation列为本次验证选模指标；rwc列为独立测试，不与原文表1混作同集合比较。BER/SNR/NMR保留全部3000段；ODG只比较四模型共有的有限结果，数量及排除索引见peaq_cohort.json，未定义结果保留诊断，不填补分数。箱线图须线为5%和95%分位数。\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()

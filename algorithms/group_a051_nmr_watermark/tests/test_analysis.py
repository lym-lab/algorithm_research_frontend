import csv
import json
import pytest
from scripts.analyze_final import collect, NAMES


def fixture(root):
    selected = {}
    for name in NAMES:
        folder = root/name; folder.mkdir()
        summary = {'config': {'bits': int(name[-3:]), 'loss': name[:-3].lower()},
                   'samples': 3000, 'manifest_sha256': 'fixture-only',
                   'ber': .01, 'snr': 20., 'nmr_db': 0.}
        (folder/'summary.json').write_text(json.dumps(summary))
        selected[name] = {'metrics': {'ber': .01, 'snr': 20., 'nmr_db': 0.}}
        with (folder/'metrics.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['index','track_id','ber','snr','nmr'])
            writer.writerows((i, f'G{i//30:03}', .01, 20, 1) for i in range(3000))
        with (folder/'peaq.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['index','odg'])
            writer.writerows((i, -1.) for i in reversed(range(3000)))
    return selected


def test_reorders_peaq_and_rejects_duplicate_indices(tmp_path):
    selected = fixture(tmp_path)
    assert collect(tmp_path, selected)[0]['odg'] == -1.
    path = tmp_path/'NMR512/peaq.csv'
    text = path.read_text().replace('2999,-1.0', '2998,-1.0')
    path.write_text(text)
    with pytest.raises(ValueError, match='unique'):
        collect(tmp_path, selected)


def test_rejects_mismatched_manifest_and_summary(tmp_path):
    selected = fixture(tmp_path)
    path = tmp_path/'NMR256/summary.json'
    summary = json.loads(path.read_text())
    summary['manifest_sha256'] = 'other'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='same manifest'):
        collect(tmp_path, selected)
    summary.update(manifest_sha256='fixture-only', snr=21)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='disagrees'):
        collect(tmp_path, selected)


def test_cli_generates_tables_and_percentile_plots(tmp_path, monkeypatch):
    from scripts.analyze_final import main
    selected = fixture(tmp_path)
    registry = tmp_path/'selected.json'; registry.write_text(json.dumps(selected))
    output = tmp_path/'generated'
    monkeypatch.setattr('sys.argv', ['analyze_final', '--root', str(tmp_path),
                                    '--selected', str(registry), '--output', str(output)])
    main()
    assert 'paper_validation_ber' in (output/'comparison.csv').read_text(encoding='utf-8-sig')
    assert (output/'snr_values.png').stat().st_size > 1000
    assert (output/'odg_values.png').stat().st_size > 1000


def test_rejects_wrong_model_directory(tmp_path):
    selected = fixture(tmp_path)
    path = tmp_path/'NMR256/summary.json'
    summary = json.loads(path.read_text())
    summary['config']['loss'] = 'mse'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='model identity'):
        collect(tmp_path, selected)


def test_documented_undefined_peaq_uses_common_cohort_without_imputation(tmp_path):
    selected = fixture(tmp_path)
    path = tmp_path/'NMR256/peaq.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['index', 'odg', 'status'])
        writer.writerows((i, 'NaN' if i == 109 else -1.,
                          'undefined_reference_output' if i == 109 else 'ok') for i in range(3000))
    with pytest.raises(ValueError, match='documented'):
        collect(tmp_path, selected)
    tmp_path.joinpath('NMR256/peaq.csv.undefined_109.txt').write_text('Reference implementation returned NaN')
    records = collect(tmp_path, selected)
    assert all(r['peaq_valid_samples'] == 2999 and r['odg'] == -1. for r in records)
    assert all(r['peaq_excluded_indices'] == [109] for r in records)
    assert records[1]['peaq_undefined_indices'] == [109]

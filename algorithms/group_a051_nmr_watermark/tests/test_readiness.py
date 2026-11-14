import json
import pytest
import torch
from scripts.check_readiness import audit
from scripts.verify_final_training import verify_run


def test_missing_evidence_is_pending_not_complete(tmp_path):
    result = audit(tmp_path)
    assert all(c['status'] == 'pending' for c in result['checks'])
    assert not list(tmp_path.iterdir())


def test_final_checkpoint_consistency_and_incomplete_rejection(tmp_path):
    config = {'bits': 256, 'loss': 'mse'}
    metrics = {'loss': .1, 'ber': .01}
    (tmp_path/'result.json').write_text(json.dumps(dict(config=config, metrics=metrics, epochs=3)))
    (tmp_path/'history.json').write_text(json.dumps([dict(epoch=1, **metrics), dict(epoch=2, loss=.2)]))
    torch.save(dict(config=config, metrics=metrics, epoch=1), tmp_path/'best.pt')
    last = dict(config=config, complete=True, epoch=2, best=.1, optimizer={}, rng_cpu=torch.get_rng_state())
    torch.save(last, tmp_path/'last.pt')
    assert verify_run(tmp_path)['best_epoch'] == 1
    last['complete'] = False
    torch.save(last, tmp_path/'last.pt')
    with pytest.raises(ValueError, match='incomplete'):
        verify_run(tmp_path)


def test_final_audit_checks_selection_without_rewriting_it(tmp_path, monkeypatch):
    from nmr_watermark.training import candidates
    from scripts.verify_final_training import main
    selected = {}
    for index, config in enumerate(candidates()):
        folder = tmp_path/str(index)
        folder.mkdir()
        mse = config['loss'] == 'mse'
        metrics = dict(loss=.1, ber=.1 if mse else .09, snr=config['alpha'], nmr=config['alpha'])
        (folder/'result.json').write_text(json.dumps(dict(config=config, metrics=metrics, epochs=4)))
        (folder/'history.json').write_text(json.dumps([dict(epoch=3, **metrics)]))
        torch.save(dict(config=config, metrics=metrics, epoch=3), folder/'best.pt')
        torch.save(dict(config=config, complete=True, epoch=3, best=.1,
                        optimizer={}, rng_cpu=torch.get_rng_state()), folder/'last.pt')
        if config['seed'] == 2026 and config['alpha'] == (.98 if mse else .1):
            selected[f"{config['loss'].upper()}{config['bits']}"] = dict(checkpoint=str(folder/'best.pt'), metrics=metrics)
    registry = tmp_path/'selected.json'
    registry.write_text(json.dumps(selected))
    original = registry.read_bytes()
    output = tmp_path/'audit.json'
    monkeypatch.setattr('sys.argv', ['verify', '--runs', str(tmp_path), '--output', str(output)])
    main()
    assert len(json.loads(output.read_text())['runs']) == 60
    assert registry.read_bytes() == original
    selected['NMR256']['metrics']['ber'] = .1
    registry.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match='selection rule'):
        main()

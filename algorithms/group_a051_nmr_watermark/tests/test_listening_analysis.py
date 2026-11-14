import json
import pytest
from scripts.analyze_listening import collect, summarize, TRACKS, MODELS, CONDITIONS


def fixture(root):
    mapping = {f'{track}/C{i+1}': name for track in TRACKS for i, name in enumerate(MODELS)}
    review = {}
    for person in range(2):
        name = f'fixture-{person}.json'
        review[name] = dict(include=True, participant_id=f'fixture-person-{person}', reason='unit test only')
        trials = [dict(type='mushra', id=track, responses=[
            dict(stimulus=label, score=40+20*person)
            for label in ('C1', 'C2', 'C3', 'C4', 'reference', 'anchor35', 'anchor70')
        ]) for track in TRACKS]
        (root/name).write_text(json.dumps(dict(testId='a051', uuid=f'fixture-session-{person}', trials=trials)))
    return mapping, review


def test_participant_bootstrap_preserves_pairs(tmp_path):
    mapping, review = fixture(tmp_path)
    values = collect(tmp_path, mapping, review)
    assert values.shape == (2, 7, 7)
    rows = summarize(values, repetitions=1000)
    assert len(rows) == 8*len(CONDITIONS)
    assert all(r['mean'] == 50 and r['participants'] == 2 for r in rows)
    assert all(r['ci95_low'] == 40 and r['ci95_high'] == 60 for r in rows)
    assert rows == summarize(values, repetitions=1000)


def test_rejects_unreviewed_duplicate_people_and_missing_trials(tmp_path):
    mapping, review = fixture(tmp_path)
    with pytest.raises(ValueError, match='every response'):
        collect(tmp_path, mapping, {})
    review['fixture-1.json']['participant_id'] = 'fixture-person-0'
    with pytest.raises(ValueError, match='unique anonymous'):
        collect(tmp_path, mapping, review)
    review['fixture-1.json']['participant_id'] = 'fixture-person-1'
    path = tmp_path/'fixture-1.json'
    session = json.loads(path.read_text())
    session['trials'].pop()
    path.write_text(json.dumps(session))
    with pytest.raises(ValueError, match='seven unique'):
        collect(tmp_path, mapping, review)


def test_rejects_invalid_rating(tmp_path):
    mapping, review = fixture(tmp_path)
    path = tmp_path/'fixture-0.json'
    session = json.loads(path.read_text())
    session['trials'][0]['responses'][0]['score'] = float('nan')
    path.write_text(json.dumps(session))
    with pytest.raises(ValueError, match='finite'):
        collect(tmp_path, mapping, review)

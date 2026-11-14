"""Analyze real webMUSHRA sessions only after explicit inclusion review.

The review file maps response filenames to {include, participant_id, reason}.
It is prepared by the study organizer, not inferred from scores or session UUIDs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

TRACKS = ('G001', 'G021', 'G030', 'G050', 'G087', 'G082', 'G059')
MODELS = ('MSE256', 'NMR256', 'MSE512', 'NMR512')
CONDITIONS = (*MODELS, 'reference', 'anchor35', 'anchor70')


def collect(responses, mapping, review):
    files = sorted(Path(responses).glob('*.json'))
    if not files:
        raise ValueError('No real responses available')
    if set(review) != {p.name for p in files}:
        raise ValueError('Explicitly review every response file; do not silently drop records')
    for track in TRACKS:
        if {mapping.get(f'{track}/C{i}') for i in range(1, 5)} != set(MODELS):
            raise ValueError('Invalid private condition mapping')
    participants, sessions, scores = set(), set(), []
    for path in files:
        decision = review[path.name]
        if type(decision.get('include')) is not bool or not decision.get('reason'):
            raise ValueError('Every review requires a boolean include and reason')
        if not decision['include']:
            continue
        participant = decision.get('participant_id')
        if not isinstance(participant, str) or not participant.strip() or participant in participants:
            raise ValueError('Require a unique anonymous participant_id for each included person')
        session = json.loads(path.read_text(encoding='utf-8'))
        if session.get('testId') != 'a051' or not session.get('uuid') or session['uuid'] in sessions:
            raise ValueError('Wrong study or duplicate session UUID')
        trials = [t for t in session['trials'] if t['type'] == 'mushra']
        if len(trials) != 7 or {t['id'] for t in trials} != set(TRACKS):
            raise ValueError('Each included session must contain all seven unique trials')
        matrix = np.empty((len(TRACKS), len(CONDITIONS)))
        for trial in trials:
            seen = set()
            for response in trial['responses']:
                label = response['stimulus']
                condition = mapping.get(f"{trial['id']}/{label}", label)
                if condition not in CONDITIONS or condition in seen:
                    raise ValueError('Unknown or duplicate stimulus')
                score = response['score']
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score) or not 0 <= score <= 100:
                    raise ValueError('Ratings must be finite numbers in [0,100]')
                matrix[TRACKS.index(trial['id']), CONDITIONS.index(condition)] = score
                seen.add(condition)
            if seen != set(CONDITIONS):
                raise ValueError('Missing model, hidden reference or anchor rating')
        participants.add(participant)
        sessions.add(session['uuid'])
        scores.append(matrix)
    if len(scores) < 2:
        raise ValueError('At least two reviewed participants needed for a confidence interval')
    return np.stack(scores)


def summarize(scores, seed=2026, repetitions=10000):
    # Resample participants, keeping all their paired conditions/tracks together.
    rng = np.random.default_rng(seed)
    n = len(scores)
    indices = rng.integers(n, size=(repetitions, n))
    rows = []
    for track, values in [*(zip(TRACKS, scores.transpose(1, 0, 2))), ('ALL', scores.mean(axis=1))]:
        bootstrap = values[indices].mean(axis=1)
        low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        for j, condition in enumerate(CONDITIONS):
            rows.append(dict(track=track, condition=condition, participants=n,
                             mean=float(values[:, j].mean()), ci95_low=float(low[j]), ci95_high=float(high[j])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=Path('outputs/listening'))
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/listening_analysis'))
    args = parser.parse_args()
    mapping_path = args.study/'private_mapping.json'
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    review = json.loads(args.review.read_text(encoding='utf-8'))
    scores = collect(args.study/'responses', mapping, review)
    rows = summarize(scores)
    # No results are written before all records pass review/validation.
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    sources = [args.review, mapping_path, *sorted((args.study/'responses').glob('*.json'))]
    metadata = dict(participants=len(scores), target=20, target_reached=len(scores) >= 20,
                    seed=2026, bootstrap_repetitions=10000, method='paired participant percentile bootstrap',
                    caveat='Session UUID is not proof of a unique human; organizer review is required. No automatic score-based exclusions.',
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'{len(scores)} reviewed participants; target 20; no participants were fabricated.')


if __name__ == '__main__':
    main()

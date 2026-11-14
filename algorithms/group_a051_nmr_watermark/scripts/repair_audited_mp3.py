"""Build an isolated repaired dataset revision; never mutate active training data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf


def main():
    sys.path.insert(0, str(Path('.cache/audio-audit-tools').resolve()))
    import imageio_ffmpeg
    audit = json.loads(Path('outputs/mp3-audit/scan.json').read_text(encoding='utf-8'))
    if audit['errors']:
        raise RuntimeError('Resolve unreadable files before preparing this revision')
    output = Path('data/repairs/mp3-v1')
    output.mkdir(parents=True, exist_ok=True)
    replacements, records = {}, []
    for warning in audit['warnings']:
        source = Path(warning['path'])
        target = output/(source.stem+'.wav')
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(source),
                   '-f', 'f32le', '-acodec', 'pcm_f32le', '-']
        result = subprocess.run(command, capture_output=True, check=True)
        if result.stderr:
            raise RuntimeError(result.stderr.decode(errors='replace'))
        info = sf.info(source)
        audio = np.frombuffer(result.stdout, dtype='<f4').reshape(-1, info.channels)
        if len(audio) != info.frames or not np.isfinite(audio).all():
            raise ValueError('Decoded length or samples invalid: '+str(source))
        sf.write(target, audio, info.samplerate, subtype='FLOAT')
        reread, rate = sf.read(target, dtype='float32', always_2d=True)
        assert rate == info.samplerate and np.array_equal(audio, reread)
        replacements[str(source)] = str(target.resolve())
        records.append(dict(source=str(source), target=str(target.resolve()),
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            frames=len(audio), sample_rate=rate, channels=info.channels,
                            decoder_stderr='', float_wav_roundtrip_exact=True))
    manifests = output/'manifests'
    manifests.mkdir(exist_ok=True)
    counts = {}
    for split in ('train', 'validation'):
        rows = json.loads(Path(f'data/manifests/full/{split}.json').read_text())
        count = 0
        for row in rows:
            if row['path'] in replacements:
                row['path'] = replacements[row['path']]
                count += 1
        (manifests/f'{split}.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
        counts[split] = count
    summary = json.loads(Path('data/manifests/full/summary.json').read_text())
    (manifests/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report = dict(records=records, replaced_manifest_rows=counts,
                  active_training_data_changed=False,
                  activation_requirement='New mask cache and separate fresh full-search run. Do not resume old checkpoints on repaired data.')
    (output/'repair.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

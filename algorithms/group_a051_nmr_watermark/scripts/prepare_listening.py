"""Prepare paper Table 2 stimuli after final models and RWC are available."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from nmr_watermark.audio import read_audio, save_audio
from nmr_watermark.inference import load_model, embed_array
from nmr_watermark.model import SAMPLE_RATE, SEGMENT_SAMPLES

CLIPS = {'G001': 240, 'G021': 20, 'G030': 58, 'G050': 210,
         'G087': 93, 'G082': 70, 'G059': 20}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('tracks', help='JSON mapping G001 etc. to verified RWC WAV paths')
    p.add_argument('registry')
    p.add_argument('--output', default='outputs/listening')
    args = p.parse_args()
    tracks = json.loads(Path(args.tracks).read_text(encoding='utf-8'))
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text())
    names = ['MSE256', 'NMR256', 'MSE512', 'NMR512']
    if set(registry) != set(names):
        raise ValueError('Require four final models')
    models = {}
    for name in names:
        path = Path(registry[name]['checkpoint'])
        model, cfg = load_model(path if path.is_absolute() else registry_path.parent/path)
        if cfg.get('pilot'):
            raise ValueError('Pilot models cannot be used for the paper listening study')
        models[name] = model
    output = Path(args.output)
    (output/'audio').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    pages, mapping = [], {}
    for track, start in CLIPS.items():
        audio = read_audio(tracks[track])[start*SAMPLE_RATE:(start+10)*SAMPLE_RATE]
        if len(audio) != 10*SAMPLE_RATE:
            raise ValueError(f'{track}: specified ten-second excerpt is unavailable')
        padded = np.pad(audio, (0, 10*SEGMENT_SAMPLES-len(audio)))
        save_audio(output/'audio'/f'{track}_reference.wav', audio)
        permutation = rng.permutation(names)
        stimuli = {}
        message = rng.integers(0, 2, 512).astype('float32')
        for i, name in enumerate(permutation):
            label = f'C{i+1}'
            marked, _ = embed_array(models[name], padded, message[:models[name].bits])
            filename = f'{track}_{label}.wav'
            save_audio(output/'audio'/filename, marked[:len(audio)])
            stimuli[label] = 'study/audio/'+filename
            mapping[f'{track}/{label}'] = name
        pages.append(dict(type='mushra', id=track, name='音质评分',
                          content='请与参考比较，评价整体音质，0代表极差，100代表无可察觉的差异。',
                          reference=f'study/audio/{track}_reference.wav',
                          createAnchor35=True, createAnchor70=True, stimuli=stimuli,
                          enableLooping=True, showWaveform=True))
    config = dict(testname='A051 音频水印听评', testId='a051', bufferSize=2048,
                  stopOnErrors=True, showButtonPreviousPage=True, remoteService='service/write.php',
                  pages=[dict(type='generic', id='welcome', name='听评说明',
                              content='请佩戴耳机，在安静环境中调节舒适音量。自愿参与，可随时退出。请勿填写姓名。'),
                         ['random', *pages], dict(type='finish', name='完成', content='感谢参与。',
                                                showResults=False, writeResults=True)])
    (output/'a051.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (output/'private_mapping.json').write_text(json.dumps(mapping, indent=2))
    (output/'manifest.json').write_text(json.dumps(dict(clips=CLIPS, seed=2026, models=registry), indent=2))
    print('Materials prepared. No subjective scores have been created.')


if __name__ == '__main__':
    main()

"""Reproduce the Fig.3 comparison style on a fixed RWC case, frame 48."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nmr_watermark.audio import save_audio
from nmr_watermark.data import AudioDataset
from nmr_watermark.inference import load_model
from nmr_watermark.psychoacoustics import NMRLoss


def main():
    torch.set_num_threads(4)
    registry = json.loads(Path('runs/paper/selected.json').read_text())
    dataset = AudioDataset('data/manifests/rwc.json')
    x = dataset[0]['audio'][None]
    message = torch.from_numpy(np.random.default_rng(2026).integers(0,2,(1,256)).astype('float32'))
    out = Path('outputs/case'); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,5),layout='constrained')
    scores = {}
    for name in ('MSE256','NMR256'):
        model, cfg = load_model(registry[name]['checkpoint'])
        if cfg.get('pilot'):
            raise ValueError('Final case requires selected full-data models')
        nmr = NMRLoss(cfg.get('stft_normalized',False))
        with torch.no_grad():
            wave, logits, host, marked = model(x, message)
            mask = nmr.masking(host)
            noise = nmr.noise(marked,host)
            if name == 'MSE256':
                pitch = torch.einsum('cf,bft->bct',nmr.mapping.float(),host.abs().square()*nmr.ear_gain.float()[None,:,None])
                for label,value in [('Host pitch',pitch),('Masking threshold',mask)]:
                    ax.plot(10*np.log10(value[0,:,48].numpy().clip(1e-20)),label=label)
            ax.plot(10*np.log10(noise[0,:,48].numpy().clip(1e-20)), label=name+' noise')
            scores[name] = {'ber':float(((logits>=0)!=message.bool()).float().mean()),
                            'nmr_db':float(10*torch.log10(nmr(marked,host,mask)[0]))}
            save_audio(out/f'{name}.wav',wave[0].numpy())
    save_audio(out/'original.wav',x[0].numpy())
    ax.set(xlabel='Critical band index',ylabel='Level (dB SPL)',title='Fixed RWC case, STFT frame 48')
    ax.legend(); fig.savefig(out/'mask_noise.png',dpi=180); plt.close(fig)
    (out/'case.json').write_text(json.dumps({'sample':dataset.rows[0],'frame':48,'seed':2026,'metrics':scores},indent=2))


if __name__ == '__main__':
    main()

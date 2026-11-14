from pathlib import Path
import numpy as np
from scipy.io import savemat, loadmat
from nmr_watermark.psychoacoustics import NMRLoss
import torch
import json

Path('outputs').mkdir(exist_ok=True)
nmr = NMRLoss()
energy = np.geomspace(1., 1e7, 109)
savemat('outputs/reference_input.mat', {'energy': energy, 'centers': nmr.centers.numpy(), 'frequencies': np.arange(513)*44100/1024})
reference_path = Path('outputs/reference_matlab.mat')
if reference_path.exists():
    r = loadmat(reference_path)
    # Test spreading at exactly the independent reference's center frequencies.
    nmr.centers.copy_(torch.from_numpy(r['fc'].ravel()))
    nmr.normalization.copy_(nmr._spread_raw(torch.ones(1, 109, dtype=torch.float64)))
    actual = (nmr._spread_raw(torch.from_numpy(energy))/nmr.normalization).numpy().ravel()
    np.testing.assert_allclose(actual, r['reference_spread'].ravel(), rtol=1e-9)
    np.testing.assert_allclose(nmr.temporal.numpy(), r['reference_a'].ravel(), rtol=1e-12)
    np.testing.assert_allclose(nmr.internal.numpy(), r['reference_internal'].ravel(), rtol=1e-12)
    relative = np.max(np.abs(actual/r['reference_spread'].ravel()-1))
    Path('outputs/reference_validation.json').write_text(json.dumps({'passed': True, 'spread_max_relative_error': float(relative), 'note': 'Spread checked at Kabal tabulated centers; production uses analytic Bark centers.'}, indent=2))
    print('Independent MATLAB reference passed:', relative)
else:
    print('Input exported. Run MATLAB validate_matlab.m then repeat this command.')

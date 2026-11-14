import numpy as np
import pytest
import torch
from nmr_watermark.model import WatermarkModel, SpectralTransform, SEGMENT_SAMPLES
from nmr_watermark.psychoacoustics import NMRLoss
from nmr_watermark.training import candidates

torch.set_num_threads(2)


def test_stft_roundtrip_and_dimensions():
    stft = SpectralTransform()
    x = torch.randn(2, SEGMENT_SAMPLES) * 0.1
    spectrum = stft(x)
    assert spectrum.shape == (2, 513, 96)
    assert (stft.inverse(spectrum) - x).abs().max() < 2e-7


@pytest.mark.parametrize("bits", [256, 512])
def test_network_contract(bits):
    torch.manual_seed(3)
    model = WatermarkModel(bits).eval()
    x = torch.randn(1, SEGMENT_SAMPLES) * .02
    message = torch.randint(0, 2, (1, bits)).float()
    with torch.no_grad():
        y, logits, host, marked = model(x, message)
        assert y.shape == x.shape
        assert logits.shape == message.shape
        assert torch.isfinite(y).all()
        # Written patch is confined to the specified spectrum region.
        raw = model.embedder(host, message)
        difference = raw - host
        difference[:, 3:131, 16:80] = 0
        assert difference.abs().max() == 0
        # Actual extracted spectrum must be from synthesized audio.
        assert torch.equal(marked, model.stft(y))


def test_nmr_identity_phase_and_gradient():
    torch.manual_seed(5)
    nmr = NMRLoss()
    host = torch.randn(1, 513, 4, dtype=torch.complex128)
    mask = nmr.masking(host)
    assert torch.isfinite(mask).all() and (mask > 0).all()
    assert nmr(host, host, mask).item() == 0
    # Equal magnitudes but different phase must incur a nonzero penalty.
    assert nmr(host * 1j, host, mask).item() > 0
    delta = torch.randn_like(host) * .001
    assert torch.allclose(nmr(host+2*delta, host, mask), 4*nmr(host+delta, host, mask))
    amplitude = torch.tensor(.001, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: nmr(host + a * delta, host, mask), (amplitude,))


def reference_spread(energy, fc):
    """Slow scalar Kabal equations, independent of torch vectorization."""
    count = len(energy)
    al = 10**(-2.7*.25)
    out = np.zeros(count)
    for source in range(count):
        au = 10**((-2.4-23/fc[source])*.25) * max(energy[source], 1e-12)**.05
        total = sum(al**j for j in range(source+1)) + sum(au**j for j in range(count-source)) - 1
        en = energy[source]/total
        for dest in range(count):
            slope = au**(dest-source) if dest >= source else al**(source-dest)
            out[dest] += (en*slope)**.4
    return out**2.5


def test_spread_against_scalar_reference():
    nmr = NMRLoss()
    energy = np.geomspace(1, 1e7, 109)
    actual = nmr._spread_raw(torch.tensor(energy)).numpy()
    expected = reference_spread(energy, nmr.centers.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_mask_temporal_reference():
    nmr = NMRLoss()
    torch.manual_seed(11)
    host = torch.randn(1, 513, 3, dtype=torch.complex128)
    actual = nmr.masking(host)[0].numpy()
    energy = nmr.mapping.numpy() @ (np.abs(host[0].numpy())**2 * nmr.ear_gain.numpy()[:, None])
    energy = np.maximum(energy, 1e-12) + nmr.internal.numpy()[:, None]
    normalization = reference_spread(np.ones(109), nmr.centers.numpy())
    state = np.zeros(109)
    columns = []
    for column in energy.T:
        spread = reference_spread(column, nmr.centers.numpy()) / normalization
        state = nmr.temporal.numpy()*state + (1-nmr.temporal.numpy())*spread
        columns.append(np.maximum(state, spread)*nmr.offset.numpy())
    np.testing.assert_allclose(actual, np.array(columns).T, rtol=1e-10)


def test_search_grid():
    grid = list(candidates())
    assert len(grid) == 60
    assert len({(c['bits'], c['loss'], c['seed'], c['alpha']) for c in grid}) == 60
    assert all(c['epochs'] == 30 for c in grid)


def test_stft_normalization_preserves_physical_nmr():
    raw, unit = SpectralTransform(), SpectralTransform(True)
    x = torch.randn(1, SEGMENT_SAMPLES)*.1
    y = x + torch.randn_like(x)*.001
    torch.testing.assert_close(unit(x)*32, raw(x))
    torch.testing.assert_close(unit.inverse(unit(x)), x, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(NMRLoss()(raw(y), raw(x)),
                               NMRLoss(True)(unit(y), unit(x)), rtol=2e-5, atol=1e-8)

"""Paper Figures 1, 4, 5. No residual embedding or attack layer."""
import torch
from torch import nn

SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 48640
N_FFT = 1024
HOP = 512


class SpectralTransform(nn.Module):
    def __init__(self, normalized=False):
        super().__init__()
        self.normalized = normalized
        self.register_buffer("window", torch.hann_window(N_FFT, periodic=False))

    def forward(self, audio):
        return torch.stft(audio, N_FFT, HOP, window=self.window.to(audio),
                          center=True, pad_mode="reflect", return_complex=True, normalized=self.normalized)

    def inverse(self, spectrum):
        return torch.istft(spectrum, N_FFT, HOP, window=self.window.to(spectrum.real),
                           center=True, length=SEGMENT_SAMPLES, normalized=self.normalized)


def down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 5, 2, 2, bias=False),
                         nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))


def up(cin, cout, dropout=False):
    layers = [nn.ConvTranspose2d(cin, cout, 5, 2, 2, output_padding=1, bias=False),
              nn.BatchNorm2d(cout), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class Embedder(nn.Module):
    def __init__(self, bits=256):
        super().__init__()
        self.bits = bits
        self.encoder = nn.ModuleList([down(2, 32), down(32, 64), down(64, 128), down(128, 256)])
        self.message = nn.Sequential(nn.Conv2d(256 + bits, 256, 5, padding=2), nn.LeakyReLU(0.2))
        self.decoder = nn.ModuleList([up(512, 128, True), up(256, 64, True), up(128, 32), up(64, 16)])
        self.output = nn.Conv2d(18, 2, 5, padding=2)

    def forward(self, spectrum, message):
        patch = torch.view_as_real(spectrum[:, 3:131, 16:80]).permute(0, 3, 1, 2)
        features = []
        x = patch
        for block in self.encoder:
            x = block(x)
            features.append(x)
        m = message[:, :, None, None].expand(-1, -1, 8, 4)
        x = self.message(torch.cat([x, m], dim=1))
        for block, skip in zip(self.decoder, reversed(features)):
            x = block(torch.cat([x, skip], dim=1))
        x = self.output(torch.cat([x, patch], dim=1))
        marked = spectrum.clone()
        marked[:, 3:131, 16:80] = torch.complex(x[:, 0], x[:, 1])
        return marked


class Extractor(nn.Module):
    def __init__(self, bits=256):
        super().__init__()
        self.layers = nn.Sequential(down(2, 32), down(32, 64), down(64, 128))
        # Text-consistent dimensions; printed Fig.5 dimensions are inconsistent.
        self.output = nn.Linear(128 * 20 * 12, bits)

    def forward(self, spectrum):
        x = torch.view_as_real(spectrum[:, :160]).permute(0, 3, 1, 2)
        return self.output(self.layers(x).flatten(1))  # logits, BCEWithLogitsLoss


class WatermarkModel(nn.Module):
    def __init__(self, bits=256, normalized=False):
        super().__init__()
        if bits not in (256, 512):
            raise ValueError("Paper supports 256 or 512 bits")
        self.bits = bits
        self.stft = SpectralTransform(normalized)
        self.embedder = Embedder(bits)
        self.extractor = Extractor(bits)

    def forward(self, audio, message):
        host = self.stft(audio)
        marked_audio = self.stft.inverse(self.embedder(host, message))
        marked = self.stft(marked_audio)
        return marked_audio, self.extractor(marked), host, marked

    def extract(self, audio):
        return self.extractor(self.stft(audio)).sigmoid()

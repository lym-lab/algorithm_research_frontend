"""End-to-end run of the WavLM-conversion commands on a tiny fake VPC directory: random small WavLM,
stub vocoder, 1.5 s synthetic utterances, a few steps. Checks the plumbing, not the learning."""
import json
from argparse import Namespace

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml

from a082 import kaldi
from a082.wavlm_conv import kmeans, train
from third_party.knnvc.wavlm.WavLM import WavLM, WavLMConfig

DIM = 64


def tiny_wavlm(device="cpu", path=None):
    torch.manual_seed(0)
    cfg = WavLMConfig({"encoder_layers": 6, "encoder_embed_dim": DIM, "encoder_ffn_embed_dim": 128,
                       "encoder_attention_heads": 4, "extractor_mode": "layer_norm"})
    return WavLM(cfg).eval()


class StubVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(DIM, 320)

    def forward(self, c):  # [1, T, D] -> [1, 1, T*320]
        return torch.tanh(self.lin(c)).reshape(1, 1, -1)


def write_wav(path, seconds, f0):
    t = np.arange(int(16000 * seconds)) / 16000
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32), 16000, subtype="PCM_16")


def make_vpc(root):
    scp, u2s = {}, {}
    for spk, f0 in (("100", 120.0), ("101", 220.0)):
        for i in range(4):
            u = f"{spk}-1-{i:04d}"
            write_wav(root / "audio" / f"{u}.wav", 1.5, f0 + 10 * i)
            scp[u], u2s[u] = str(root / "audio" / f"{u}.wav"), spk
    for name, keep in (("train-clean-360", list(scp)), ("libri_dev_enrolls", list(scp)[:2])):
        d = root / "data" / name
        d.mkdir(parents=True)
        kaldi.write_kaldi({u: scp[u] for u in keep}, d / "wav.scp")
        kaldi.write_kaldi({u: u2s[u] for u in keep}, d / "utt2spk")
    tgt_files = []
    for i in range(3):
        p = root / "target" / f"t{i}.wav"
        write_wav(p, 1.5, 300.0 + 5 * i)
        tgt_files.append(str(p))
    return tgt_files


def test_commands_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "load_wavlm", tiny_wavlm)
    monkeypatch.setattr(train, "load_hifigan", lambda device="cpu", prematched=True: StubVocoder())
    root = tmp_path / "vpc"
    tgt_files = make_vpc(root)
    manifest = tmp_path / "pool.json"
    manifest.write_text(json.dumps({"idT": tgt_files}))
    matching = torch.randn(300, DIM)
    target = tmp_path / "target.pt"
    torch.save({"speaker": "idT", "matching": matching.half(),
                "centroids": kmeans.spherical_kmeans(matching, k=8, iters=5), "k": 8}, target)

    spk = tmp_path / "spk.pt"
    train.cmd_pretrain_spk(Namespace(vpc_root=root, out=str(spk), steps=2, batch=4, crop=1.0, target_acc=2.0,
                                     eval_every=2, log_every=1, workers=0, device="cpu"))
    assert torch.load(spk)["n_speakers"] == 2

    cfg = {"name": "tiny", "steps": 3, "crop_s": 1.0, "batch": 4, "warmup": 10, "target_recon": 0.25,
           "valid_utts": 4, "adv": True, "ctc": True, "ctc_sigma": 10.0,
           "model": {"feat_dim": DIM, "d": 16, "enc_layers": 1, "dec_layers": 1, "heads": 2, "filt": 24}}
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "wc"
    args = Namespace(config=str(cfg_path), vpc_root=root, target=str(target), manifest=str(manifest),
                     spk=str(spk), init=None, out=str(out), workers=0, device="cpu", log_every=1, save_every=3)
    train.cmd_train(args)
    ck = torch.load(out / "last.pt")
    assert ck["step"] == 3 and "clf" in ck and "ctc" in ck

    cfg["steps"] = 4  # resume continues from the saved step
    cfg_path.write_text(yaml.safe_dump(cfg))
    train.cmd_train(args)
    assert torch.load(out / "last.pt")["step"] == 4

    train.cmd_infer(Namespace(ckpt=str(out / "last.pt"), suffix="_wcT", vpc_root=root, sets="libri_dev_enrolls",
                              shard="0/1", device="cpu", finalize_only=False))
    scp = kaldi.read_kaldi(root / "data" / "libri_dev_enrolls_wcT" / "wav.scp")
    assert len(scp) == 2 and all(sf.info(p).samplerate == 16000 for p in scp.values())


def _train_rank(rank, world, port, kw):
    import os
    os.environ.update(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    train.load_wavlm = tiny_wavlm
    train.cmd_train(Namespace(**kw))


def test_train_on_two_ranks(tmp_path):
    """The training loop under torchrun's environment with two CPU processes (gloo): batch 8 is split 4 + 4 and
    the one target-reconstruction sample goes to rank 0 only, with the adversarial and CTC losses on."""
    import socket
    import torch.multiprocessing as mp
    root = tmp_path / "vpc"
    tgt_files = make_vpc(root)
    manifest = tmp_path / "pool.json"
    manifest.write_text(json.dumps({"idT": tgt_files}))
    matching = torch.randn(300, DIM)
    target = tmp_path / "target.pt"
    torch.save({"speaker": "idT", "matching": matching.half(),
                "centroids": kmeans.spherical_kmeans(matching, k=8, iters=5), "k": 8}, target)
    spk = tmp_path / "spk.pt"
    torch.save({"model": train.SpeakerClassifier(2, DIM).state_dict(), "n_speakers": 2, "feat_dim": DIM}, spk)
    cfg = {"name": "tiny", "steps": 3, "crop_s": 1.0, "batch": 8, "warmup": 10, "target_recon": 0.125,
           "valid_utts": 4, "adv": True, "ctc": True, "ctc_sigma": 10.0,
           "model": {"feat_dim": DIM, "d": 16, "enc_layers": 1, "dec_layers": 1, "heads": 2, "filt": 24}}
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "wc"
    kw = dict(config=str(cfg_path), vpc_root=root, target=str(target), manifest=str(manifest), spk=str(spk),
              init=None, out=str(out), workers=0, device="cpu", log_every=1, save_every=2)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    mp.spawn(_train_rank, args=(2, port, kw), nprocs=2)
    ck = torch.load(out / "last.pt")
    assert ck["step"] == 3 and ck["gpus"] == 2 and "clf" in ck and "ctc" in ck

# Vendored from bshall/knn-vc

Source: https://github.com/bshall/knn-vc, commit `c616845`, MIT licence (`LICENSE`).

Only the network definitions are kept, so that the published checkpoints can be loaded:

- `wavlm/WavLM.py`, `wavlm/modules.py` (WavLM, itself copied from microsoft/unilm, MIT);
- `hifigan/models.py`, `hifigan/utils.py`, `hifigan/config_v1_wavlm.json` (HiFi-GAN generator for WavLM layer-6 features).

The files are unchanged. The kNN matching, length perturbation, feature noise, target pools and everything else in `a082/` are our own code.

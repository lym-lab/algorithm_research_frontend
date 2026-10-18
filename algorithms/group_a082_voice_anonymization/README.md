# HLTCOE JHU VoicePrivacy 2024 方案复现：说话人匿名化

复现论文：Li Xinyuan et al., *HLTCOE JHU Submission to the Voice Privacy Challenge 2024*, Proc. SPSC 2024, pp. 61–66, doi:10.21437/SPSC.2024-11。

（草稿：结果、网盘链接、部署地址在实验完成后补上。）

## 复现了什么

论文表 1 的全部 11 个系统，外加图 4 的混合比例扫描，全部按 VoicePrivacy 2024 官方协议评测 EER、UAR、WER：

| 表 1 | 系统 | 代码 |
|---|---|---|
| 1、2、3（另加 1M） | kNN-VC、+ 长度扰动、+ 长度扰动 + 特征加噪；1M 是多目标池化 | `a082/knnvc/` |
| 4 | Whisper medium.en 转写 → Piper（VITS，904 个音色）合成的级联 | `a082/cascade/` |
| 1+4 | 随机混合，p = 0.2 / 0.325 / 0.4，以及图 4 的 0.1–0.5 | `a082/admix.py` |
| 5–8 | WavLM 转换模型（FastSpeech2 式前馈 Transformer）：基础版、+ 说话人对抗、+ 离散 CTC、两者都加 | `a082/wavlm_conv/` |

评测编排与按容差判定在 `a082/eval/`，加分实验在 `a082/bonus/`，网页在 `webapp/`。

## 自己实现的部分与现成组件

**自己实现：**
- kNN 匹配，多目标池化，长度扰动，特征加噪，目标说话人池；
- 级联系统与混合系统的编排；
- WavLM 转换模型：网络、三种损失、k-means、训练与推断；
- 评测编排与结果汇总；
- 加分实验；
- 网页前后端。

**只加载公开预训练权重的组件：**

| 组件 | 来源 | 许可 |
|---|---|---|
| WavLM-Large | kNN-VC release v0.1 | MIT |
| 预匹配 HiFi-GAN 声码器 | kNN-VC release v0.1 | MIT |
| Whisper medium.en | openai/whisper | MIT |
| Piper `en_US-libritts_r-medium` | rhasspy/piper-voices | MIT；训练数据 CC BY 4.0 |
| 评测模型与脚本 | VoicePrivacy 2024 框架 | GPL-3.0，原样使用 |

WavLM 与 HiFi-GAN 的网络定义取自 `bshall/knn-vc`，放在 `third_party/knnvc/`，文件未作改动，见该目录的 README。

## 运行

**环境：** Python 3.11、PyTorch 2.0.1，与 VPC 2024 框架相同。

```bash
pip install -e .
pip install "openai-whisper @ git+https://github.com/openai/whisper@86098128c0b4f24f0e2aa2994de830614b474227" piper-tts==1.8.0 speechbrain==0.5.16 fastapi uvicorn python-multipart matplotlib pyyaml
pytest -q tests          # 不需要任何权重，几十秒
```

**网页演示：** 把 `webapp/config.example.yaml` 复制成 `webapp/config.yaml`，填好路径后运行：

```bash
python webapp/server.py --config webapp/config.yaml --host 0.0.0.0 --port 8000
```

**完整复现：** 在 GPU 机上按 [`docs/REPRODUCE.md`](docs/REPRODUCE.md) 的顺序执行。

**权重：** [网盘链接，待补]。

## 与论文不同或论文没写的地方

论文没给的细节由我们自己定，逐条列在报告附录中，每条都写了依据和可能的影响。其中会影响数字的有四处：
- 目标说话人池用 VoxCeleb1 dev + test；
- 系统 3 的噪声位置和幅度，用 dev 集标定；
- WavLM 转换模型的训练步数和各项损失的权重；
- 评测集 libri_dev、libri_test 由 OpenSLR 的 LibriSpeech 按官方清单重建。

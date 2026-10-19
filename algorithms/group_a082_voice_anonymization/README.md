# HLTCOE JHU VoicePrivacy 2024 方案复现：说话人匿名化

复现论文：Li Xinyuan et al., *HLTCOE JHU Submission to the Voice Privacy Challenge 2024*, Proc. SPSC 2024, pp. 61–66, doi:10.21437/SPSC.2024-11。

说话人匿名化要把语音里"是谁在说"的信息去掉，同时尽量保留"说了什么"和情感。本项目自己实现了论文表 1 的全部系统，并按 VoicePrivacy 2024 官方协议评测三个指标：
- **EER**：攻击者的说话人验证等错误率，越高越好；
- **WER**：语音识别词错误率，越低越好；
- **UAR**：情感识别的非加权平均召回率，越高越好。

| 入口 | 地址 |
|---|---|
| 在线演示（华为云） | 【待填：部署地址】 |
| 模型权重与演示资产（网盘） | 【待填：网盘链接与提取码】 |
| 报告 | 另行提交 |

## 复现了什么

论文表 1 的全部 11 个系统，外加图 4 的混合比例扫描：

| 表 1 | 系统 | 代码 |
|---|---|---|
| 1、2、3（另加 1M） | kNN-VC、+ 长度扰动、+ 长度扰动 + 特征加噪；1M 是多目标池化 | `a082/knnvc/` |
| 4 | Whisper medium.en 转写 → Piper（VITS，904 个音色）合成的级联 | `a082/cascade/` |
| 1+4 | 随机混合，p = 0.2 / 0.325 / 0.4，以及图 4 的 0.1–0.5 | `a082/admix.py` |
| 5–8 | WavLM 转换模型（FastSpeech2 式前馈 Transformer）：基础版、+ 说话人对抗、+ 离散 CTC、两者都加 | `a082/wavlm_conv/` |

评测编排与按容差判定在 `a082/eval/`，加分实验在 `a082/bonus/`，网页在 `webapp/`。

## 结果

【待填：表 1 复现值与论文值对照（平均 EER / UAR / WER），由 `python -m a082.eval.vpc report` 生成；图 4 的混合比例扫描图；加分实验的主要结论。】

## 目录结构

```
a082/            Python 包
  knnvc/         kNN 匹配、多目标池化、长度扰动、特征加噪、目标说话人池（系统 1、1M、2、3）
  cascade/       Whisper → Piper 级联（系统 4）
  admix.py       随机混合
  wavlm_conv/    WavLM 转换模型：网络、损失、k-means、训练与推断（系统 5–8）
  eval/          调用 VPC 2024 评测框架、汇总结果、与论文对照
  bonus/         加分实验：系统分类器、攻击者分离重训、按试验类型拆分 EER、有损转码
  service.py     网页用的推理服务；viz.py 画图
configs/         每个系统的参数
scripts/         云上安装、数据重建、与官方 kNN-VC 逐数比对、加噪标定、Whisper 自身 WER
webapp/          网页：FastAPI 后端 + 三个静态页面（在线演示、复现结果、方法与系统）
deploy/          CPU 服务器部署脚本与 systemd 服务
tests/           单元测试与接口测试（不需要权重）
third_party/     kNN-VC 发布的 WavLM、HiFi-GAN 网络定义（未改动）
docs/            完整复现步骤
```

## 自己实现的部分与现成组件

**自己实现：**
- kNN 匹配，多目标池化，长度扰动，特征加噪，目标说话人池；
- 级联系统与混合系统的编排；
- WavLM 转换模型：网络、三种损失、k-means、训练与推断；
- 评测编排与结果汇总；
- 加分实验；
- 网页前后端与部署脚本。

**只加载公开预训练权重的组件：**

| 组件 | 来源 | 许可 |
|---|---|---|
| WavLM-Large | kNN-VC release v0.1 | MIT |
| 预匹配 HiFi-GAN 声码器 | kNN-VC release v0.1 | MIT |
| Whisper medium.en | openai/whisper | MIT |
| Piper `en_US-libritts_r-medium` | rhasspy/piper-voices | MIT；训练数据 CC BY 4.0 |
| ECAPA 说话人模型（网页里算相似度） | speechbrain/spkrec-ecapa-voxceleb | Apache-2.0 |
| 评测模型与脚本 | VoicePrivacy 2024 框架 | GPL-3.0，原样使用 |

WavLM 与 HiFi-GAN 的网络定义取自 `bshall/knn-vc`，放在 `third_party/knnvc/`，文件未作改动，见该目录的 README。

## 安装

Python 3.11、PyTorch 2.0.1，与 VPC 2024 框架相同。Linux 或 Windows 均可（完整复现需要 Linux 与 GPU）。

```bash
pip install torch==2.0.1 torchaudio==2.0.2          # 只有 CPU 时加 --index-url https://download.pytorch.org/whl/cpu
pip install -e . "numpy<2" soundfile pyyaml tqdm matplotlib jiwer huggingface_hub fastapi uvicorn python-multipart pytest
pip install "openai-whisper @ git+https://github.com/openai/whisper@86098128c0b4f24f0e2aa2994de830614b474227" piper-tts==1.8.0 speechbrain==0.5.16
pytest -q tests                                    # 不需要任何权重，约一分钟
```

## 运行

**1. 不下载权重，先看网页。** 用一个不加载模型的替身服务（"匿名化"只是把输入放慢到 0.8 倍），能走通上传、选择系统、试听、看图的全部流程：

```bash
python tests/fake_service.py --port 8765           # 浏览器打开 http://127.0.0.1:8765
```

**2. 用真实模型跑网页演示。** 从网盘下载演示资产，解压后把 `webapp/config.example.yaml` 复制成 `webapp/config.yaml`，按注释填好路径：

```bash
python webapp/server.py --config webapp/config.yaml --host 0.0.0.0 --port 8000
```

只有 CPU 时，级联与混合系统里的 Whisper medium.en 最慢，可在配置里换成 `base.en`。

**3. 部署到云服务器（CPU）。** Ubuntu 22.04/24.04 上一条命令装成 systemd 服务，资产目录的内容见脚本开头的注释：

```bash
sudo bash deploy/install_cpu.sh /opt/a082 <资产目录>
```

**4. 完整复现表 1。** 需要 GPU、VPC 2024 评测框架和数据集，按 [`docs/REPRODUCE.md`](docs/REPRODUCE.md) 的顺序执行。

## 数据与许可

- 评测集 libri_dev、libri_test 由 OpenSLR 的 LibriSpeech dev-clean、test-clean 按 VPC 2024 的官方清单重建（`scripts/rebuild_libri.py`），脚本逐句核对文件是否齐全、时长是否与清单一致。
- 目标说话人池用 VoxCeleb1 dev + test；VoxCeleb 与 IEMOCAP 的音频按许可不随项目分发，网页演示的目标说话人改用 LibriTTS-R（CC BY 4.0）。
- VPC 2024 评测框架不随本项目分发，由 `scripts/cloud_setup.sh` 从其官方仓库下载。
- IEMOCAP 的音频由 `scripts/cloud_setup.sh` 从 Hugging Face 上的转传副本 `AbstractTTS/IEMOCAP`（固定版本）下载，经 `scripts/prepare_iemocap_hf.py` 还原成原有目录结构并核对句子齐全。

## 与论文不同或论文没写的地方

论文没给的细节由我们自己定，逐条列在报告附录中，每条都写了依据和可能的影响。其中会影响数字的有四处：
- 目标说话人池用 VoxCeleb1 dev + test；
- 系统 3 的噪声位置和幅度，用 dev 集标定；
- WavLM 转换模型的训练步数和各项损失的权重；
- 评测集 libri_dev、libri_test 由 OpenSLR 的 LibriSpeech 按官方清单重建。

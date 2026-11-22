# HLTCOE JHU VoicePrivacy 2024 方案复现：说话人匿名化

复现论文：Li Xinyuan et al., *HLTCOE JHU Submission to the Voice Privacy Challenge 2024*, Proc. SPSC 2024, pp. 61–66, doi:10.21437/SPSC.2024-11。

说话人匿名化要把语音里"是谁在说"的信息去掉，同时尽量保留"说了什么"和情感。本项目自己实现了论文表 1 的全部系统，并按 VoicePrivacy 2024 官方协议评测三个指标：
- **EER**：攻击者的说话人验证等错误率，越高越好；
- **WER**：语音识别词错误率，越低越好；
- **UAR**：情感识别的非加权平均召回率，越高越好。

| 入口 | 地址 |
|---|---|
| 在线演示（华为云） | http://119.3.187.247:8000 |
| 模型权重与演示资产（网盘） | 百度网盘 https://pan.baidu.com/s/1haKLqx9Ct6gUvpfraoVTEQ?pwd=x9du（提取码 x9du） |
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

表 1 全部 12 行都已评测（平均值，%；论文值在括号里）。判定按报告中逐行设定的容差（例如系统 1、2 与混合：UAR ±2、WER ±0.3；系统 6–8 用定性判据），容差同 `a082/eval/targets.py`：5 行复现，7 行部分复现，没有失败的行。

| 表 1 | 系统 | EER ↑ | UAR ↑ | WER ↓ | 判定 |
|---|---|---|---|---|---|
| 0 | 原始语音 | 5.16 (5.16) | 70.07 (70.07) | 1.82 (1.83) | 复现 |
| 1 | kNN-VC | 12.24 (7.95) | 55.14 (56.70) | 2.70 (3.16) | 部分复现 |
| 2 | + 长度扰动 | 12.59 (8.08) | 55.23 (56.29) | 2.81 (3.33) | 部分复现 |
| 3 | + 长度扰动 + 特征加噪 | 21.11 (20.58) | 44.34 (43.26) | 11.90 (11.00) | 复现 |
| 4 | Whisper → Piper 级联 | 48.50 (48.25) | 30.83 (30.35) | 3.96 (3.75) | 复现 |
| 1+4 | 混合 p = 0.2 | 23.48 (21.91) | 49.87 (51.71) | 2.96 (3.31) | 部分复现 |
| 1+4 | 混合 p = 0.325 | 35.68 (33.91) | 46.75 (49.04) | 3.14 (3.43) | 部分复现 |
| 1+4 | 混合 p = 0.4 | 40.53 (40.81) | 44.88 (47.09) | 3.25 (3.33) | 部分复现 |
| 5 | WavLM 转换（基础） | 7.04 (8.54) | 52.84 (54.75) | 2.53 (3.01) | 复现 |
| 6 | + 说话人对抗 | 12.83 (11.60) | 50.63 (50.62) | 2.64 (4.23) | 复现 |
| 7 | + 离散 CTC | 8.78 (13.73) | 46.81 (43.56) | 2.70 (10.16) | 部分复现 |
| 8 | 两者都加 | 11.45 (13.88) | 46.23 (43.50) | 2.70 (10.83) | 部分复现 |

主要结论：
- 原始语音三项指标与官方数字逐位相同，评测链路正确；隐私—效用的权衡、各方法的隐私排序和混合的超线性收益都复现了：$p = 0.4$ 时 EER 40.53，比两个系统按比例的线性插值 26.74 高 13.8。
- kNN-VC 系列 EER 偏高约 4.3；只换目标说话人池的三组对照能解释其中约三分之一。
- 系统 7、8 没有出现论文中离散目标带来的可懂度下降：论文用它让模型在时间上变形，而我们的转换模型逐帧输出、保留逐帧重建损失，CTC 项几乎不起作用（没训练过 CTC 的系统 5 单元错误率就已是 15.6%，系统 7 为 15.9%）。
- 混合的隐私收益依赖"分配对攻击者不可预测"：攻击者能以 0.9999 的准确率分辨系统，按系统分别训练只让 $p = 0.4$ 的 EER 降 3.0；改成按说话人固定系统后 EER 降到 19.45。
- 说话人对抗训练只让重新训练的攻击者学得更慢：在转换后的特征上从零训练的说话人分类器，3000 步时只能认出系统 6 的 43%–51% 的说话人，训练到 10000 步回升到 76%–78%。

图 4 的混合比例扫描（p = 0.1–0.5）、各项对照与加分实验的完整结果见报告。

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
- 目标说话人池用 VoxCeleb1 dev + test；VoxCeleb 与 IEMOCAP 的音频按许可不随项目分发，网页演示的目标说话人改用 LibriSpeech test-clean 的 6 个说话人（CC BY 4.0）。
- VPC 2024 评测框架不随本项目分发，由 `scripts/cloud_setup.sh` 从其官方仓库下载。
- IEMOCAP 的音频由 `scripts/cloud_setup.sh` 从 Hugging Face 上的转传副本 `AbstractTTS/IEMOCAP`（固定版本）下载，经 `scripts/prepare_iemocap_hf.py` 还原成原有目录结构并核对句子齐全。

## 与论文不同或论文没写的地方

论文没给的细节由我们自己定，逐条列在报告附录中，每条都写了依据和可能的影响。其中会影响数字的有五处：
- 目标说话人池用 VoxCeleb1 dev + test；
- 系统 3 的噪声位置和幅度，用 dev 集标定；
- WavLM 转换模型的训练步数和各项损失的权重：基础模型 100k 步，系统 6–8 从它继续训练 50k 步，学习率调度接续基础模型；
- WavLM 转换模型逐帧输出（去掉了 FastSpeech2 的长度调节器），因此离散 CTC 目标无法实现论文所说的时间变形；
- 评测集 libri_dev、libri_test 由 OpenSLR 的 LibriSpeech 按官方清单重建。

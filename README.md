# 第A082组：HLTCOE JHU VoicePrivacy 2024 说话人匿名化复现

## 1. 小组与项目

| 信息 | 本组填写 |
| --- | --- |
| 组号 | 第 A082 组 |
| 选题名称 | A082：HLTCOE JHU Submission to the Voice Privacy Challenge 2024（说话人匿名化） |
| 成员 | 刘心铭（GitHub `edsfod`）、邱子骞（GitHub `lightburst404`），分工见下表 |
| 方法 ID | `group_a082_voice_anonymization` |
| 本组仓库与提交分支 | [edsfod/algorithm_research_frontend](https://github.com/edsfod/algorithm_research_frontend)，分支 [`group_a082_submission`](https://github.com/edsfod/algorithm_research_frontend/tree/group_a082_submission) |
| 在线算法服务（华为云） | 演示页面 http://119.3.187.247:8000 ，平台接口 `http://119.3.187.247:8000/process` |
| 模型权重与演示资产 | 百度网盘 https://pan.baidu.com/s/1haKLqx9Ct6gUvpfraoVTEQ?pwd=x9du （提取码 x9du） |

| 成员 | 分工 |
| --- | --- |
| 刘心铭 | 选题与复现规划；全部系统与评测流程的实现（kNN-VC 系列、级联、混合、WavLM 转换模型、评测封装）；语料与评测环境搭建；表 1 中 ID 0–5、1M、三个混合与图 4 扫描的匿名化与评测；加分实验（系统分类器、分离攻击、改进 A/B、有损转码）；ID 5、6 训练；网页后端与前端初版；审核 ID 7、8 首轮训练并修正重训、评测；说话人信息探针与单元错误率检查；课程平台接入 |
| 邱子骞 | 交接后的环境恢复；ID 6 的匿名化与评测；ID 7、8 的首轮训练、匿名化与评测（作为修正重训的对照）；修复评测封装的两处缺陷；结果表、网页数据与前端改进；部署脚本与华为云部署；情绪定向坍缩等问题分析 |

说话人匿名化是一种针对声纹识别的**主动防御**：在语音公开或上传之前，把其中"是谁在说"的信息去掉，同时尽量保留"说了什么"和情感，使攻击者的说话人验证系统无法把匿名化后的语音和原说话人对上。本方法的输入是一段英语语音，输出是换成其他说话人音色、内容不变的语音。目标效果按 VoicePrivacy 2024 的协议衡量：攻击者的等错误率 EER 越高越好，语音识别词错误率 WER 越低越好，情感识别的非加权平均召回率 UAR 越高越好。

## 2. 论文与代码来源

- **论文：** Li Xinyuan et al., *HLTCOE JHU Submission to the Voice Privacy Challenge 2024*, Proc. Symposium on Security and Privacy in Speech Communication (SPSC) 2024, pp. 61–66。[doi:10.21437/SPSC.2024-11](https://doi.org/10.21437/SPSC.2024-11)，[arXiv:2409.08913](https://arxiv.org/abs/2409.08913)。论文没有公开代码。
- **使用的开源项目（只加载预训练权重，或原样调用）：**

  | 组件 | 来源与版本 | 许可 |
  | --- | --- | --- |
  | WavLM-Large、预匹配 HiFi-GAN 声码器 | [bshall/knn-vc](https://github.com/bshall/knn-vc) release v0.1；网络定义原样放在 [`third_party/knnvc/`](algorithms/group_a082_voice_anonymization/third_party/knnvc/) | MIT |
  | Whisper medium.en | [openai/whisper](https://github.com/openai/whisper) 提交 `86098128` | MIT |
  | Piper `en_US-libritts_r-medium` | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | MIT；训练数据 LibriTTS-R 为 CC BY 4.0 |
  | ECAPA-TDNN 说话人模型（只用于网页上的相似度读数） | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Apache-2.0 |
  | 评测模型与脚本 | [VoicePrivacy Challenge 2024 框架](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024)，由安装脚本下载，不随本仓库分发 | GPL-3.0 |

- **本组算法代码：** [`algorithms/group_a082_voice_anonymization/`](algorithms/group_a082_voice_anonymization/)，说明见其中的 [README](algorithms/group_a082_voice_anonymization/README.md)。
- **数据许可：** VoxCeleb 与 IEMOCAP 的音频按许可不随本项目分发；网页演示的目标说话人和本仓库的样例都来自 LibriSpeech test-clean（CC BY 4.0）。

## 3. 本组实现内容

论文的方法全部由本组自己实现（论文没有公开代码）；公开组件只用其预训练权重，见第 2 节。

| 功能 | 完成状态 | 实现位置与验证方式 |
| --- | --- | --- |
| kNN-VC（表 1 ID 1）、多目标池化（1M）、长度扰动（2）、特征加噪（3）、目标说话人池 | 已完成 | [`a082/knnvc/`](algorithms/group_a082_voice_anonymization/a082/knnvc/)；与 kNN-VC 官方实现逐数比对，输出最大差 1.2×10⁻⁷ |
| Whisper → Piper 级联（ID 4） | 已完成 | [`a082/cascade/`](algorithms/group_a082_voice_anonymization/a082/cascade/) |
| 级联与 kNN-VC 的随机混合（ID 1+4，p = 0.2 / 0.325 / 0.4 与图 4 的 0.1–0.5 扫描） | 已完成 | [`a082/admix.py`](algorithms/group_a082_voice_anonymization/a082/admix.py) |
| WavLM 转换模型（ID 5–8）：网络、重建损失、说话人对抗、k-means 单元上的 CTC，训练与推断 | 已完成 | [`a082/wavlm_conv/`](algorithms/group_a082_voice_anonymization/a082/wavlm_conv/)；自己训练，权重在网盘 |
| VoicePrivacy 2024 评测编排、结果汇总、按容差与论文对照 | 已完成 | [`a082/eval/`](algorithms/group_a082_voice_anonymization/a082/eval/)；原始语音的 EER、WER、UAR 与官方数字逐位相同 |
| 表 1 全部 12 行的复现 | 已完成：5 行复现、7 行部分复现 | 结果见第 7 节 |
| 加分实验：系统分类器、分离攻击、两个改进方案、有损转码、说话人信息探针 | 已完成 | [`a082/bonus/`](algorithms/group_a082_voice_anonymization/a082/bonus/) |
| 自有网页（演示、结果、方法三页）与 CPU 部署 | 已完成，部署在华为云 | [`webapp/`](algorithms/group_a082_voice_anonymization/webapp/)、[`deploy/`](algorithms/group_a082_voice_anonymization/deploy/) |
| **课程平台适配**：`POST /process` 接口（multipart 音频 + `parameters` JSON，返回 `artifacts`，错误返回 `{"error"}`，允许跨域）；方法配置与样例 | 已完成 | 接口在 [`webapp/server.py`](algorithms/group_a082_voice_anonymization/webapp/server.py)，测试 [`tests/test_server.py`](algorithms/group_a082_voice_anonymization/tests/test_server.py)；配置 [`public/methods/group_a082_voice_anonymization.json`](public/methods/group_a082_voice_anonymization.json)；样例 [`public/samples/group_a082_voice_anonymization/`](public/samples/group_a082_voice_anonymization/) |

与论文不同或论文没有写明的地方，见算法 README 的"与论文不同或论文没写的地方"一节，其中影响数字的有五处：目标说话人池用 VoxCeleb1 dev + test；系统 3 的噪声位置和幅度用 dev 集标定；WavLM 转换模型的训练步数和损失权重（基础模型 100k 步，系统 6–8 从它继续 50k 步）；WavLM 转换模型逐帧输出（去掉了长度调节器）；评测集由 OpenSLR 的 LibriSpeech 按官方清单重建。

## 4. 输入、输出与参数

| 输入字段 | 素材类型与格式 | 必填 | 限制与说明 |
| --- | --- | --- | --- |
| `audio` | 音频，WAV 或 FLAC | 是 | 单个文件不超过 20 MB；时长 0.5–30 秒；多声道取平均，任意采样率重采样到 16 kHz。模型在英语朗读语音（LibriSpeech）上训练与评测 |

| 参数字段 | 类型 | 默认值与范围 | 作用 |
| --- | --- | --- | --- |
| `system` | 下拉选择 | `knn1`；可选 `knn1`、`knn1m`、`knn2`、`knn3`、`tts`、`mix`、`wc5`、`wc6`、`wc7`、`wc8` | 匿名化系统，对应论文表 1 的 ID 1、1M、2、3、4、1+4、5、6、7、8 |
| `seed` | 文本，可选 | 留空则随机生成 | 相当于句子编号：相同种子得到相同的随机选择（目标说话人、长度因子、噪声、合成音色、混合抽签）；系统 5–8 不受影响 |
| `mix_p` | 数值，可选 | 0.4；0–1，步长 0.05 | 只对 `mix` 有效：这句话以概率 p 交给级联，否则交给 kNN-VC |

| 输出 | 文件类型 | 含义 |
| --- | --- | --- |
| 匿名化语音 | WAV（16 kHz，16 bit） | 匿名化结果 |
| 原始语音的对数梅尔谱 | PNG | 与下一张使用同一色标，便于对比 |
| 匿名化语音的对数梅尔谱 | PNG | 同上 |
| 处理记录 | JSON | 实际使用的系统（`mix` 会写出抽到哪一个）、目标说话人、种子、输入输出时长、用时，以及 ECAPA-TDNN 给出的匿名化前后说话人余弦相似度（越接近 0 越不像原说话人） |

## 5. 环境与安装

- **已验证的环境：**
  - 部署（推理）：华为云 ECS，2 vCPU / 8 GB 内存（另加 2 GB swap），Ubuntu 22.04，Python 3.11，PyTorch 2.0.1 CPU 版；
  - 训练与评测：Linux，单卡 RTX 3090 级 GPU（WavLM 转换模型训练需 24 GB 显存），CUDA 11.8，PyTorch 2.0.1；
  - 前端：Node.js 22 以上；本地开发在 Windows 11 上验证。
- **依赖：** Python 包见 [`pyproject.toml`](algorithms/group_a082_voice_anonymization/pyproject.toml) 与算法 README 的"安装"一节；前端依赖见根目录 `package.json`。
- **权重与资产：** 从百度网盘下载资产包（约 2.6 GB，含 `SHA256SUMS`）。其中有 WavLM 转换模型 `wc/wc5.pt`–`wc8.pt`、`torch/hub/checkpoints/`（WavLM-Large、预匹配 HiFi-GAN）、`piper/`、`pool_demo/`（演示用 6 个目标说话人）、`samples/`。Whisper medium.en 与 ECAPA 在第一次使用时自动下载。

在 Linux 上安装算法服务（CPU 即可）：

```bash
cd algorithms/group_a082_voice_anonymization
python3.11 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.0.1 torchaudio==2.0.2
pip install -e . "numpy<2" soundfile pyyaml tqdm matplotlib fastapi uvicorn python-multipart pytest \
    speechbrain==0.5.16 "huggingface_hub==0.25.2" piper-tts==1.8.0 \
    "openai-whisper @ git+https://github.com/openai/whisper@86098128c0b4f24f0e2aa2994de830614b474227"
pytest -q tests          # 不需要权重，约 1 分钟
```

## 6. 启动与运行

**1. 启动算法服务。** 三种方式任选一种：

- **直接用已部署的服务（默认）：** 方法配置里的接口已经是 `http://119.3.187.247:8000/process`，不用做任何事。
- **本机启动真实模型：** 网盘资产解压到 `<资产目录>` 并校验（`cd <资产目录> && sha256sum -c SHA256SUMS`），然后在 `algorithms/group_a082_voice_anonymization/` 下把 `webapp/config.example.yaml` 复制为 `webapp/config.yaml`，把 `pool`、`piper_model`、`wavlm_conv` 的路径改成 `<资产目录>` 下的对应文件（只有 CPU 时可把 `whisper_model` 改成 `base.en`，快得多），再启动：

  ```bash
  export TORCH_HOME=<资产目录>/torch
  python webapp/server.py --config webapp/config.yaml --host 127.0.0.1 --port 8000
  ```

  接口地址为 `http://127.0.0.1:8000/process`。
- **不下载权重、只走通流程：** `python tests/fake_service.py --port 8765`，接口地址 `http://127.0.0.1:8765/process`。这个替身服务不加载模型，"匿名化"只是把输入放慢到 0.8 倍。

用本机服务时，把 [`public/methods/group_a082_voice_anonymization.json`](public/methods/group_a082_voice_anonymization.json) 里 `adapter.endpoint` 改成上面的地址；或者不改文件，在平台的"接入算法"页面导入这份配置、改接口地址后"登记到本机预览"。

**2. 启动前端。** 在仓库根目录：

```bash
npm install
npm run dev
```

浏览器打开终端打印的地址（默认 `http://127.0.0.1:5173`），在"方法集合"里选择分类"主动防御"或搜索"说话人匿名化"，进入本方法。

**3. 第一次运行。** 在方法详情里选第一个样例"系统 1 kNN-VC"，点"用样例试一试"（载入样例音频和参数 `system = knn1`、`seed = a082`），再点"开始处理"。约 10–20 秒后得到 4 个结果：匿名化语音、两张梅尔谱、处理记录。用同一种子得到的匿名化语音应与样例输出相同，处理记录中目标说话人为 `1089`。之后可以上传自己的英语录音，换成系统 3（隐私更强）或系统 8 对比。

**4. 结果与清理。** 结果在页面上预览，用下载按钮保存（平台另外提供本次参数记录的下载）。服务器只在内存中保留最近 32 个结果，不写磁盘，无需清理；本机服务用 Ctrl+C 停止即可。

算法配置文件：[`public/methods/group_a082_voice_anonymization.json`](public/methods/group_a082_voice_anonymization.json)。

**其他入口：**

- 本组自己的网页演示：http://119.3.187.247:8000 （同一服务，三页：在线演示、复现结果、方法与系统）。
- 完整复现论文表 1（需要 GPU、VPC 2024 框架与数据集）：[`docs/REPRODUCE.md`](algorithms/group_a082_voice_anonymization/docs/REPRODUCE.md)。

## 7. 样例与实际结果

**平台样例。** 输入是 LibriSpeech test-clean 的一句（说话人 1580，9 秒，不在演示目标池中）。输出都由部署的服务对这段输入实际生成；"相似度"是 ECAPA-TDNN 的匿名化前后余弦相似度。

| 样例 | 输入链接 | 输出链接 | 参数 | 观察重点 |
| --- | --- | --- | --- | --- |
| 系统 1 kNN-VC | [input](public/samples/group_a082_voice_anonymization/input_1580-141084-0004.wav) | [output_knn1](public/samples/group_a082_voice_anonymization/output_knn1.wav) | `system=knn1, seed=a082` | 目标说话人 1089；内容与语调保留，音色换成目标；相似度 0.169 |
| 系统 3 + 长度扰动 + 特征加噪 | 同上 | [output_knn3](public/samples/group_a082_voice_anonymization/output_knn3.wav) | `system=knn3, seed=a082` | 语速拉伸为 0.867 倍；隐私更强（相似度 0.032），可懂度下降 |
| 系统 4 级联 | 同上 | [output_tts](public/samples/group_a082_voice_anonymization/output_tts.wav) | `system=tts, seed=a082` | Whisper 的转写与参考文本一字不差，再由 Piper 第 772 号音色读出（9.2 秒）；说话人信息几乎全部去掉（相似度 0.108），语调和情感也随之丢失 |
| 系统 6 WavLM 转换 + 对抗 | 同上 | [output_wc6](public/samples/group_a082_voice_anonymization/output_wc6.wav) | `system=wc6` | 转到所有输入共用的目标说话人 VoxCeleb id10986；相似度 0.151；系统 5–8 不受种子影响 |
| 系统 8 WavLM 转换 + 对抗与离散目标 | 同上 | [output_wc8](public/samples/group_a082_voice_anonymization/output_wc8.wav) | `system=wc8` | 同一目标说话人；相似度 0.136；没有出现论文中离散目标带来的可懂度下降（原因见下文主要结论） |

**论文表 1 的复现结果。** 按 VoicePrivacy 2024 官方协议在 LibriSpeech dev/test 与 IEMOCAP 上评测，四个子集的平均值（%），括号里是论文的数字。判定按逐行设定的容差（例如系统 1、2 与混合：UAR ±2、WER ±0.3；系统 6–8 用定性判据，见 [`a082/eval/targets.py`](algorithms/group_a082_voice_anonymization/a082/eval/targets.py)）：5 行复现，7 行部分复现。

| 表 1 | 系统 | EER ↑ | UAR ↑ | WER ↓ | 判定 |
| --- | --- | --- | --- | --- | --- |
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

主要结论：原始语音三项指标与官方数字逐位相同，说明评测链路正确；隐私—效用的权衡、各方法的隐私排序和混合的超线性收益都复现了（$p = 0.4$ 时 EER 40.53，比两个系统按比例的线性插值 26.74 高 13.8）；kNN-VC 系列 EER 偏高约 4.3；系统 7、8 没有出现论文中离散目标带来的可懂度下降，原因是我们的转换模型逐帧输出、保留逐帧重建损失，CTC 项几乎不起作用。分析与加分实验的完整结果见课程报告。

## 8. 验证与已知限制

- **实际执行过的测试：**
  - 算法：`pytest -q tests` 60 项通过（含 `/process` 接口的请求格式、结果、错误、跨域、20 MB 上传上限与并发取图测试，用不加载模型的替身服务）；
  - 部署：同一段语音分别经线上服务和本地（同一份权重）推理，波形相关系数 0.99998（系统 7）与 1.0000（系统 8）；线上压力测试：10 个系统 × 两段语音（9 秒、17.7 秒）× 两个种子共 40 次，29.5 秒语音跑 5 个系统，再加 3 路并发请求和 31 秒超长输入（按预期返回错误），全部按预期完成，服务进程没有重启；
  - 前端：`npm test` 6 项通过，`npm run build` 成功；用平台自带的 `runAlgorithm`（与浏览器执行的是同一段代码）调用线上接口，返回的 4 个结果都能取回，传入不存在的系统时页面收到服务端的错误信息
- **接口状态：** 华为云服务 `http://119.3.187.247:8000/process` 长期开机。若实例因故不可用，按第 6 节在本机启动服务。
- **页面协议：** 接口是 HTTP。从 `npm run dev` 的本地页面（`http://127.0.0.1:5173`）调用没有问题；若平台部署在 HTTPS 页面上，浏览器会拦截对 HTTP 接口的请求，此时需要在本机启动服务，或为服务器配置 HTTPS。
- **耗时与资源：** 服务器是 2 vCPU 的 CPU 机器。9 秒的语音，kNN-VC 约 6–8 秒，WavLM 转换约 8–10 秒，级联约 30 秒（与论文相同，用 Whisper medium.en 转写）；接近上限的 29.5 秒语音，kNN-VC 与 WavLM 转换约 32–34 秒，级联约 64 秒。服务重启后第一次调用要加载模型，kNN-VC 与 WavLM 转换约 30 秒，级联约 60–75 秒。所有模型都加载后，进程内存峰值在 17.7 秒的输入下约 6.7 GiB，29.5 秒的输入下约 7.0 GiB（实测），所以服务器配了 8 GB 内存，并把 `vm.swappiness` 设为 10，让内存峰值时能用上 2 GB swap，而不是让内核直接杀掉进程（有的云镜像默认设为 0，安装脚本会提示）。安装脚本默认把级联的转写模型装成 base.en，这样 4 GB 的机器也放得下；安装时设 `WHISPER_MODEL=medium.en` 就用论文的模型，本服务器就是这样装的。转换模型在内存中只保留最近用过的一个，切换系统 5–8 时重新加载约 1–2 秒。同一时间只处理一个请求，其他请求排队等待。
- **取消：** 取消只中断网页请求；服务器上已经开始的处理会做完，结果留在内存中，随后被新结果挤出（最多保留 32 个）。
- **适用范围：** 英语朗读语音效果最好；噪声大、非英语或带音乐的录音未经验证。级联系统依赖 Whisper 的转写，转写错误会直接变成内容错误。
- **与表 1 的关系：** 网页演示的目标说话人是 LibriSpeech test-clean 的 6 个人，不是论文评测用的 VoxCeleb 目标池；单句的相似度读数用于直观比较，不能代替表 1 的 EER。

## 9. 附加文档

- [算法代码说明（安装、运行、目录结构、自己实现与现成组件、与论文的差异）](algorithms/group_a082_voice_anonymization/README.md)
- [完整复现步骤（GPU 服务器上从零复现表 1）](algorithms/group_a082_voice_anonymization/docs/REPRODUCE.md)
- [CPU 服务器部署脚本](algorithms/group_a082_voice_anonymization/deploy/install_cpu.sh)
- [kNN-VC 网络定义的来源与许可](algorithms/group_a082_voice_anonymization/third_party/knnvc/README.md)
- [方法配置](public/methods/group_a082_voice_anonymization.json)
- [平台样例](public/samples/group_a082_voice_anonymization/)

平台本身的说明保留在 [`docs/`](docs/) 目录：[使用说明](docs/使用说明.md)、[算法接入规范](docs/算法接入规范.md)、[算法提交指南](docs/算法提交指南.md)。

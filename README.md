# 第XX组：语音匿名化（情感增强特征融合）

## 1. 小组与项目

| 信息 | 本组填写 |
| --- | --- |
| 组号 | 第 XX 组（验收当天分配） |
| 选题名称 | 语音匿名化：融合情感增强特征的 STT+TTS 方法复现 |
| 成员 | 向泽涛（GitHub: comet7k7k）——单人组，负责全部实现 |
| 方法 ID | `voice_anonymization` |
| 本组仓库与提交分支 | [comet7k7k/algorithm_research_frontend](https://github.com/comet7k7k/algorithm_research_frontend) · `voice_anon_submission` |

本方法解决语音数据中的说话人身份泄露问题：输入一段任意说话人的语音，输出一段音色固定为伪说话人的匿名语音，使听者无法识别原说话人身份，同时保留文本内容。

## 2. 论文与代码来源

- 论文：Lee, Park, You. *Voice Anonymization Using Emotion-Enriched Feature Integration with STT and TTS Models.* Proc. SPSC, 2024.
- 算法后端代码：[comet7k7k/voice-anon](https://github.com/comet7k7k/voice-anon)
- 使用的开源组件：
  - Whisper small.en（OpenAI，faster-whisper 封装）
  - VITS LJ Speech（Coqui TTS）
  - ECAPA-TDNN（speechbrain，说话人验证）
  - wav2vec2-base-emotion（SUPERB SER，情感识别）
  - RAVDESS 数据集（情感原型计算）

## 3. 本组实现内容

| 功能 | 完成状态 | 实现位置与验证方式 |
| --- | --- | --- |
| Whisper 文本提取 | 已完成 | `app/whisper_model.py`，WER=3.12% |
| F0 提取与情感原型注意力 | 已完成 | `app/emotion.py`，RAVDESS 真实 F0 质心 [222, 362, 337, 409] Hz |
| FiLM 调制层 | 已完成 | `app/film.py`（论文公式 6-7） |
| VITS 重合成匿名语音 | 已完成 | `app/vits.py`，固定伪说话人音色 |
| WER / EER / UAR 评测闭环 | 已完成 | `eval/`，EER=0.00%，UAR 53.12%→25.00% |
| GPU 加速（RTX 4060） | 已完成 | Whisper 3.3×、ECAPA 8.0×、SER 6.8× |
| 华为云 V100 公网部署 | 已完成 | http://120.46.83.92/（按需开机） |
| 平台标准 /process 接口 | 已完成 | `app/main.py`，支持跨域 |

## 4. 输入、输出与参数

| 输入字段 | 素材类型与格式 | 必填 | 限制与说明 |
| --- | --- | --- | --- |
| `file` | 音频 WAV | 是 | 不超过 20MB，建议 5–15 秒英文语音 |

本方法无需额外参数。

| 输出 | 文件类型 | 含义 |
| --- | --- | --- |
| 匿名后音频 | WAV | VITS 重合成的固定伪说话人音色语音 |
| 处理报告 | JSON | 转录文本、F0 均值、情感原型来源、注意力权重形状 |

## 5. 环境与安装

后端算法服务（Python）：

- OS：Ubuntu 22.04 / Windows 11
- Python：3.10
- GPU：NVIDIA GPU + CUDA 12.2（V100 或 RTX 4060）
- 依赖：`requirements.txt`（含 torch 2.5.1+cu121、faster-whisper、coqui-tts、speechbrain、fastapi）

```bash
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

模型权重不进 Git，从 [百度网盘](https://pan.baidu.com/s/1rI8gQftydnkzUqLcVg0IPg?pwd=wqcx)（提取码 wqcx）下载，放到对应目录。

## 6. 启动与运行

1. 启动后端服务：
```bash
cd voice-anon
export PUBLIC_BASE=http://120.46.83.92  # 公网地址
python -m uvicorn app.main:app --host 0.0.0.0 --port 80
```
2. 前端平台：
```bash
npm install
npm run dev
```
3. 在"方法集合"中选择"语音匿名化"，上传一段 WAV，点击开始处理。
4. 结果区可试听匿名后音频，下载处理报告 JSON。

方法配置文件：`public/methods/voice_anonymization.json`

## 7. 样例与实际结果

| 样例 | 输入链接 | 输出链接 | 参数 | 观察重点 |
| --- | --- | --- | --- | --- |
| RAVDESS 中性语音 | [input.wav](public/samples/voice_anonymization/input.wav) | [output.wav](public/samples/voice_anonymization/output.wav) | 无 | 原始男声 → 匿名后统一女声音色，说话人身份抹除 |

评测指标（RAVDESS 32 段）：WER=3.12%，EER=0.00%，UAR=25.00%。

## 8. 验证与已知限制

- 已验证：Whisper 转录正确、VITS 合成正常、EER/UAR/WER 三项指标闭环可复跑、华为云公网可访问。
- 接口状态：华为云实例按需开机，验收当天在线；其余时间关机保留。
- 已知限制：
  1. VITS 无显式 F0 输入口，情感保真度有限（消融 UAR 差异 0.00pp）；
  2. 仅支持英文语音；
  3. 测试集仅 RAVDESS 32 段，统计置信度有限。

## 9. 附加文档

- [算法后端代码仓库](https://github.com/comet7k7k/voice-anon)
- [模型权重百度网盘](https://pan.baidu.com/s/1rI8gQftydnkzUqLcVg0IPg?pwd=wqcx)（提取码 wqcx）

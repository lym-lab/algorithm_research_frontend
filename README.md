# 第N057组：LRAW 音频水印复现

## 1. 小组与项目

| 信息 | 本组填写 |
| --- | --- |
| 组号 | 第N057组 |
| 选题名称 | LRAW 音频水印复现：面向重录与混合攻击的 LightGBM 盲恢复 |
| 成员 | 汤思成（GitHub: `chimuhuakai`），1 人组，全部模块 |
| 方法 ID | `group_n057_lraw_watermark` |
| 本组仓库与提交分支 | `https://github.com/chimuhuakai/algorithm_research_frontend/tree/group_n057_submission` |

本组复现论文的发送端 DWT-GBT-SVD 水印嵌入、LightGBM 逐段 0/1 提取、m 序列滑窗重建，并补齐真实接收方口径的盲恢复。平台侧提供一条同步 `POST /process` 接口：上传 12 秒音频后，返回含水印音频、攻击后音频、恢复结果摘要、恢复报告 JSON 和波形对比图。

## 2. 论文与代码来源

- 论文：*LightGBM-Based Audio Watermarking Robust to Recapturing and Hybrid Attacks*
- 作者开源 baseline：[IEEE TIFS 2025 论文](https://doi.org/10.1109/TIFS.2025.3559408)（DOI: 10.1109/TIFS.2025.3559408）
- 本组算法源码：`algorithms/group_n057_lraw_watermark/`
- 模型权重：`author_trained.pkl`，约 3.6 MB；百度网盘：https://pan.baidu.com/s/10fK98eCWkWq6c-WzWkVCtQ?pwd=5jnh（提取码：`5jnh`）；仓库源码放置到 `algorithms/group_n057_lraw_watermark/assets/author_trained.pkl`，服务器部署对应 `/opt/LRAW-reproduction/assets/author_trained.pkl`
- 第三方许可证：按作者开源协议保留；本组只做课程复现，不再分发作者数据集

## 3. 本组实现内容

| 功能 | 完成状态 | 实现位置与验证方式 |
| --- | --- | --- |
| DWT-GBT-SVD 嵌入、SNR 评估 | 已完成 | `lraw/embed.py`、`lraw/transform.py`；`scripts/` 与平台 `/process` |
| LightGBM 逐段 0/1 提取 | 已完成 | `lraw/extract.py`、`assets/author_trained.pkl`；10% 留出 F1=0.9970 |
| m 序列滑窗重建（论文 Algorithm 1） | 已完成 | `lraw/recover.py`；oracle 口径只作上界 |
| 真实接收方盲恢复（只持 K/L/taps） | 已完成 | `lraw/recover.py::blind_recover`；干净与 MP3/TSM/LSC 实测锁定 |
| 31 档攻击矩阵 × 20 宿主 | 已完成 | `docs/attack_matrix_results_L4.csv`；报告与 PPT 引用同一口径 |
| WebUI 三阶段演示 | 已完成 | `algorithms/group_n057_lraw_watermark/webui/app.py`；本机可视化，不作为平台接口 |
| 平台 `/process` 同步接口 | 已完成 | `algorithms/group_n057_lraw_watermark/platform_service.py`；返回 4–5 个 `artifacts` |
| 华为云公网部署 | 已完成 | `http://121.36.39.138:8000/health`；Ubuntu 24.04 + Gunicorn + systemd |

## 4. 输入、输出与参数

| 输入字段 | 素材类型与格式 | 必填 | 限制与说明 |
| --- | --- | --- | --- |
| `input_file` | `audio/wav`、`audio/mpeg`、`audio/flac` | 是 | 3 秒以上；平台配置上限 20 MB；内部统一到 44.1 kHz 单声道 |

| 参数字段 | 类型 | 默认值与范围 | 作用 |
| --- | --- | --- | --- |
| `attack_preset` | select | `mp3c_64`；33 档 | 31 档主矩阵 + 干净信道 + MP3C 128 kbps；覆盖 8 种 CSP、TSM/PSM 各 4 档、LSC 5 档、数字重录 7 档、3 种混合攻击 |
| `repeat` | select | `1`；1/3/5 | 同一比特连续写入 k 段；k 越大越稳、容量越低 |
| `message` | text | `LRAW2026`；≤40 字符 | 仅 ASCII；留空写随机比特 |

| 输出 | 文件类型 | 含义 |
| --- | --- | --- |
| 含水印音频 | WAV | 发送端嵌入结果，附 SNR |
| 攻击后音频 | WAV | 所选信道攻击后的实际文件 |
| 恢复结果摘要 | PNG | LOCKED/UNLOCKED、盲 BER、原始 BER、SNR、恢复文本 |
| 恢复报告 | JSON | `locked`、`ber`、`agreement`、`margin`、`phase_d`、`n_obs`、`snr_embed_db` 等 |
| 波形对比 | PNG | 原始 / 含水印 / 攻击后三段波形 |

## 5. 环境与安装

- 已验证：Ubuntu 24.04、Python 3.12、2 vCPU / 2 GiB、CPU 推理
- 依赖：`requirements.txt` + `gunicorn` + `matplotlib`
- 系统包：`python3-venv`、`python3-pip`、`libsndfile1`、`libgomp1`

```bash
cd /opt/LRAW-reproduction
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn matplotlib
```

模型权重放到 `/opt/LRAW-reproduction/assets/author_trained.pkl`；也可用 `LRAW_MODEL_PATH` 指向其他位置。

## 6. 启动与运行

1. 启动算法服务：

```bash
cd /opt/LRAW-reproduction
source .venv/bin/activate
PUBLIC_BASE_URL=http://121.36.39.138:8000 \
  gunicorn --worker-class gthread -w 1 --threads 4 -b 0.0.0.0:8000 platform_service:app
```

1. 健康检查：

```bash
curl http://121.36.39.138:8000/health
```

1. 启动前端：

```bash
git clone https://github.com/chimuhuakai/algorithm_research_frontend.git
cd algorithm_research_frontend
git switch group_n057_submission
npm install
npm run dev
```

1. 浏览器打开 `http://localhost:5173`，在方法列表选择“LRAW 音频水印嵌入与盲恢复”，上传 `public/samples/group_n057_lraw_watermark/music2.wav`，参数 `mp3c_64`、`repeat=1`、`message=LRAW2026`，点击运行。
2. 结果在页面直接预览；文件保存在算法服务器 `output/platform/<task_id>/`，默认保留 24 小时。

方法配置：`public/methods/group_n057_lraw_watermark.json`。

## 7. 样例与实际结果

| 样例 | 输入链接 | 输出链接 | 参数 | 观察重点 |
| --- | --- | --- | --- | --- |
| 音乐样本 2 + MP3C 64 kbps | `public/samples/group_n057_lraw_watermark/music2.wav` | `public/samples/group_n057_lraw_watermark/music2_mp3c64_attacked.wav` | `attack_preset=mp3c_64`, `repeat=1`, `message=LRAW2026` | `locked=true`，盲 BER=0.00%，原始提取 BER=6.20%，SNR=26.50 dB |
| 语音 + TSM 0.9× | `public/samples/group_n057_lraw_watermark/speech_01.wav` | `public/samples/group_n057_lraw_watermark/speech_01_tsm09_attacked.wav` | `attack_preset=tsm_0.9`, `repeat=1`, `message=LRAW2026` | `locked=true`，盲 BER=0.00%，原始提取 BER=2.27%，SNR=27.51 dB |

主矩阵：20 个宿主 × 31 档，共 620 次；全矩阵均值 12.90%，剔除 PSM 相关 5 档后 5.93%。平台示例只展示 2 个档位，方法页下拉框可选 33 个档位。

## 8. 验证与已知限制

- 华为云已实测：`/health` 返回 200；`/process` 的 MP3C 64 档返回 5 个 artifacts（含恢复结果摘要和波形对比）；`report.json` 为 `locked=true`、盲 BER=0；`OPTIONS /process` 返回 204 且 CORS 头正常。本机 Flask test client 也验证了 `tsm_0.9`、`lsc_0.5` 等档位。
- 公开接口：`http://121.36.39.138:8000`，验收期间保持运行；如更换 IP 需同步修改方法配置的 `endpoint`。
- 限制 1：PSM 改变音高，s3 的条件分布收敛，d′ 从 0.77 降到 0.07，盲 BER 约 50%，属方法层边界，不应当作调参失败。
- 限制 2：重录是数字信道模型，不是 JBL GO ESSENTIAL + iPhone 12 的物理实测；真实房间混响与非线性的影响未覆盖。
- 限制 3：裁剪采用“保留前缀”语义；任意起点裁剪需要额外时间轴同步，不在当前主表内。
- 限制 4：论文未公开 VAD 阈值，攻击前后各做一次 VAD 仍难与发送端掩码严格一致；删除型 VAD 直接替换后，干净回环 BER 也会接近 50%。
- 限制 5：平台同步接口按单请求处理，12 秒音频约 10–12 秒；不支持异步轮询；并发高时需加 worker 或队列。
- 限制 6：当前云端接口是 HTTP。前端用本地 `http://localhost:5173` 打开时正常；如果前端页面是 HTTPS，浏览器会拦截 HTTP 接口，需要把接口也升级到 HTTPS。
- 取消行为：浏览器取消等待只会断开连接，服务端任务可能继续跑完；结果目录 24 小时后清理。

## 9. 附加文档

- [项目报告](docs/LRAW音频水印复现项目报告_汤思成.docx)
- [答辩 PPT](docs/实验汇报.pptx)
- [攻击矩阵 CSV](docs/attack_matrix_results_L4.csv)
- [算法代码说明](algorithms/group_n057_lraw_watermark/README.md)

没有放入 Git 的内容：模型权重（百度网盘链接见第 2 节）、作者数据集 `dataset.xlsx`、`output/platform/` 运行产物、`.venv/`、服务凭据。

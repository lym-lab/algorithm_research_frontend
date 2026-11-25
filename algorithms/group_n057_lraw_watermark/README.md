# LRAW 算法源码说明

本目录是第N057组“LRAW 音频水印复现”的算法源码。根目录 `README.md` 负责课程提交说明和结果入口；本文负责**代码导读**：讲清楚一段音频从输入到输出经过了哪些模块、每个阶段在做什么，以及平台参数如何映射到算法实现。

阅读顺序建议：

```text
config.py -> transform.py -> embed.py -> extract.py -> recover.py
          -> attacks.py -> platform_service.py -> pipeline.py
```

---

## 1. 目录与模块分工

```text
algorithms/group_n057_lraw_watermark/
├── lraw/
│   ├── config.py        # 全局参数、A/B、m 序列阶数 L 和本原多项式表
│   ├── transform.py     # DWT-GBT-SVD 特征提取与 s3 修改后的重建
│   ├── embed.py         # 发送端水印嵌入
│   ├── extract.py       # 接收端逐段 0/1 提取
│   ├── recover.py       # 盲恢复、oracle 恢复、长度对齐
│   ├── msequence.py     # m 序列生成、校验、密钥和窗口恢复
│   ├── attacks.py       # 31 档攻击与混合攻击
│   ├── pipeline.py      # 实验用端到端流水线
│   ├── model.py         # LightGBM 加载与训练
│   ├── metrics.py       # SNR、BER 等指标
│   ├── audio_io.py      # 音频读写
│   └── vad.py           # 可选能量 VAD
├── platform_service.py  # 平台 HTTP 服务：/process、/health、/files
├── webui/app.py         # 本机 WebUI，不作为平台接口
├── scripts/             # 训练、演示、批量攻击评估脚本
├── viz/                 # 流程图、波形图、鲁棒性图
├── assets/              # 模型元数据；author_trained.pkl 不入库
├── requirements.txt
└── README.md
```

---

## 2. 端到端调用链

### 2.1 平台实际调用链

平台页面上传音频后，实际入口是 `platform_service.py::process`：

```text
POST /process
  -> read_upload()                 读入并统一音频格式
  -> embed()                       分段、DWT-GBT-SVD、修改 s3、逆变换
  -> apply_preset()                按 attack_preset 施加攻击
  -> extract_bits()                LightGBM 逐段预测 0/1
  -> blind_recover()               只使用 K/L/taps 做 m 序列相位搜索
  -> write_audio / report.json     写出结果文件
  -> artifacts JSON                 返回 5 个结果入口
```

平台路径**不读取原始水印真值 W**。它把提取出的含错序列 `W~`、密钥 `K`、m 序列阶数 `L` 和抽头 `taps` 交给 `blind_recover`，这才是实际接收方口径。

### 2.2 实验脚本调用链

`lraw/pipeline.py::run_pipeline` 是给脚本和本地实验用的简化流水线：

```text
embed -> attack -> extract_bits -> recover_watermark
```

注意：`pipeline.py` 里的 `recover_watermark` 是 **oracle 口径**，它为了定位锚点会读取真值 `W`。它可以作为理论上界和调试工具，但不能当作真实接收方的鲁棒性结果。平台和论文主结果使用 `blind_recover`。

---

## 3. 发送端：`embed.py`

入口函数：

```python
emb = embed(audio, watermark_bits, cfg, repeat=1)
```

主要步骤如下。

### 3.1 分段

默认每段 `seg_len = 400` 个样本：

```text
音频 -> 固定网格切段 -> 段矩阵 (n_seg, 400)
```

默认 `cfg.use_vad = False`，即从样本 0 开始等长分段。这样发送端和接收端使用完全相同的段边界，不依赖 VAD 阈值。VAD 只作为可选预处理保留。

### 3.2 DWT-GBT-SVD 特征

每个 400 样本段在 `transform.py::extract_features` 中处理：

1. 对段做两层 DWT，默认 `db4`，`periodization` 模式；
2. 取第二层近似系数 `cA2`，长度为 `400 / 4 = 100`；
3. 构造路径图拉普拉斯特征基 `E`，做图变换：

   ```text
   spar = appr @ E.T
   ```

4. 把 100 个图频域系数排成 `5 x 20` 矩阵 `Y`；
5. 对 `Y` 做薄 SVD：

   ```text
   Y = U * diag(S) * V^T
   S = [s1, s2, s3, s4, s5]   # 已按降序排列
   ```

6. 用最大奇异值归一化，得到 LightGBM 的 5 维输入特征。

路径图特征基只与段长有关，代码用 `lru_cache` 缓存 `100 x 100` 的特征基，避免长音频逐段重复分解。

### 3.3 修改第三个奇异值

嵌入只改 `s3`，保持 `s4 < s3~ < s2` 的奇异值顺序。参数 `a=5`、`b=2`：

```text
w_i = 0：s3~ = (s3 + a * s4) / (a + 1)
w_i = 1：s3~ = (s2 + b * s3) / (b + 1)
```

直观理解：

- 写入 0 时把 `s3` 向 `s4` 拉，第三个奇异值偏小；
- 写入 1 时把 `s3` 向 `s2` 拉，第三个奇异值偏大。

随后 `transform.py::reconstruct_with_modified_s3` 只替换 `S[2]`，重新合成 `Y`，再做 IGBT、IDWT，保留原始细节系数。

### 3.4 水印比特、重复度与密钥

- `watermark_bits` 是逻辑水印比特 `W`；
- `repeat=k` 表示同一个比特连续写入 `k` 段，逻辑容量变成 `n_seg // k`；
- m 序列阶数：

  ```text
  L = ceil(log2(n + 1))
  ```

  其中 `n` 是逻辑比特数。`config.py` 维护 `L=5..20` 的本原多项式表，长音频超过 5 分钟时会用到 `L=17` 以上；
- 密钥：

  ```text
  K = W xor M
  ```

  发送端保存 `K`，接收端恢复 `M` 后再计算 `W = K xor M`。

### 3.5 嵌入保真度

`embed` 返回 `snr_voiced`：

```text
SNR = 10 * log10(原始有声区能量 / 水印扰动能量)
```

这个值只衡量嵌入造成的失真，不包含后续 MP3、变速、重录等攻击的影响。

---

## 4. 接收端：`extract.py`

入口函数：

```python
ext = extract_bits(audio, model, cfg, expected_len=..., repeat=...)
```

### 4.1 时长归一

如果攻击只造成轻微时长变化，接收端按原始样本数重采样回原长度，再按 400 样本切段：

- 普通攻击 `tol=0.03`；
- TSM、重录等可能产生明显时长变化的攻击，平台传 `tol=0.25`；
- 裁剪类 LSC 不能简单按比例归一，否则会把删掉的片段当成变速。

### 4.2 逐段分类

每一段仍然做一次 `extract_features`，得到 5 个归一化奇异值：

```text
400 样本段 -> cA2(100) -> GBT -> Y(5x20) -> SVD -> [s1..s5]/s1
```

将这 5 维特征送入 LightGBM，输出该段“嵌入 1”的概率。

### 4.3 重复度融合

当 `repeat=k` 时，每 `k` 段的概率取平均，再按 0.5 判决：

```text
p_i = mean(p_seg[i*k : i*k+k])
w_i = 1 if p_i >= 0.5 else 0
```

这里是在概率域平均，而不是先硬判决再做多数投票，目的是保留弱证据和强证据之间的置信度差异。

---

## 5. 盲恢复：`recover.py`

### 5.1 真实接收方口径

`blind_recover(W_tilde, key, L, taps)` 只使用：

- 含错提取序列 `W~`；
- 密钥 `K`；
- m 序列阶数 `L`；
- 本原多项式抽头 `taps`。

不读取原始 `W`。

计算过程：

```text
M_obs = K xor W~
```

m 序列是周期为 `2^L - 1` 的循环序列，整条序列由相位 `d` 决定。代码枚举所有候选相位，选择与 `M_obs` 一致率最高的相位：

```text
score(d) = mean_i [ c_i * (-1)^(K_i xor M[(i+d) mod P]) ]
```

其中 `c_i = 2p_i - 1`，`p_i` 是 LightGBM 的软概率。这个式子被改写为带符号权重与 m 序列的圆相关，用 FFT 一次计算全部相位，替代原来的 `O(n*m)` Python 双重循环。

### 5.2 锁定条件

代码不是简单固定阈值，而是按实际观测长度 `m` 自适应：

```text
threshold = 0.5 + 5.5 * (0.5 / sqrt(m))
locked = m >= 20 and best >= threshold and margin > 0.03
```

`margin` 是最优相位与次优相位的差距。这样既控制“在多个相位里挑最大值”的虚警，也能在裁剪后只剩短片段时按实际证据量判断。

### 5.3 oracle 与 blind 的区别

- `recover_watermark()`：借真值 `W` 搜索长度为 `L` 的完全一致锚点，命中后理论上 BER=0；属于 oracle 上界；
- `blind_recover()`：不借 `W`，直接搜索 m 序列相位；平台 `report.json` 中的 `blind` 字段来自它。

报告中的 `raw_ber` 是提取序列 `W~` 的误码率，`blind.ber` 是盲恢复后的误码率；判断实际是否成功应以 `locked` 为准。

---

## 6. 攻击实现：`attacks.py`

`platform_service.py` 的 `PRESETS` 把平台下拉框的 33 个档位映射到攻击函数：

- 8 种 CSP：GNA、MP3C、AM、RS、LPF、MF、RQ、CP；
- TSM 四档：0.8/0.9/1.1/1.2；
- PSM 四档：0.8/0.9/1.1/1.2；
- LSC 五档：保留前缀 20%/50%/70%/80%/90%；
- 数字重录七档：0.5/1/2/3/5/10/13 m；
- 干净信道 `none`；
- MP3C 128 kbps；
- 3 种混合攻击：重录 1 m + ECSP/TSM/PSM。

几个实现细节：

- `_codec_delay` 用多窗口估计 MP3 编解码延迟，避免弱起音乐开头相关性不足导致整首歌错位；
- `_resample_scale` 负责 TSM 的时间轴缩放；
- `recapture` 是数字信道模型（距离衰减、扬声器低通、环境噪声、速率失配、房间混响），不是物理扬声器 + 手机实测；
- `hybrid_attack` 按顺序串联两个攻击。

---

## 7. 平台接口：`platform_service.py`

### 7.1 请求

```http
POST /process
Content-Type: multipart/form-data
```

字段：

| 字段 | 说明 |
|---|---|
| `input_file` | 音频文件，支持 WAV/MP3/FLAC |
| `method_id` | `group_n057_lraw_watermark` |
| `parameters` | JSON 字符串，不是 `params` |

`parameters` 示例：

```json
{"attack_preset":"mp3c_64","repeat":"1","message":"LRAW2026"}
```

平台参数含义：

| 参数 | 作用 |
|---|---|
| `attack_preset` | 33 个攻击档位之一 |
| `repeat` | 同一比特连续写入的段数，1/3/5 |
| `message` | ASCII 水印文本，最多 40 字符；留空则写随机比特 |

### 7.2 返回

正常返回 5 个 artifacts：

```text
watermarked.wav   含水印音频
attacked.wav      攻击后音频
summary.png       恢复结果摘要
report.json       恢复报告
waveform.png      三段波形对比
```

`summary.png` 和 `waveform.png` 的绘制依赖 matplotlib；如果绘图库缺失，代码会捕获绘图异常，仍保留音频和 `report.json`，此时 artifacts 可能是 3 个；在完整依赖环境下应为 5 个。

`report.json` 的关键字段：

- `attack_preset`、`repeat`、`n_bits`；
- `snr_embed_db`；
- `raw_ber`；
- `blind.locked`、`blind.ber`、`blind.agreement`、`blind.margin`、`blind.phase_d`；
- `recovered_text`、`text_match`；
- `elapsed_s`。

### 7.3 其他接口

```text
GET /health                 健康检查
GET /files/<task_id>/<name> 下载结果文件
OPTIONS /process            CORS 预检
```

---

## 8. 参数与源码映射

| 源码位置 | 作用 |
|---|---|
| `config.py::Config` | 采样率、段长、DWT、SVD、VAD、LightGBM 超参数 |
| `config.py::PRIMITIVE_POLYNOMIAL` | `L=5..20` 的 m 序列本原多项式 |
| `transform.py::extract_features` | DWT-GBT-SVD 特征提取 |
| `transform.py::reconstruct_with_modified_s3` | 只修改 s3 后的逆变换 |
| `embed.py::embed` | 分段、量化、密钥、重复度、SNR |
| `extract.py::extract_bits` | 时长归一、LightGBM、重复度融合 |
| `recover.py::blind_recover` | 真实接收方盲恢复 |
| `recover.py::recover_watermark` | oracle 锚点恢复，仅作调试/上界 |
| `attacks.py::apply_attack` | 单攻击入口 |
| `attacks.py::hybrid_attack` | 混合攻击入口 |
| `platform_service.py::process` | 平台 HTTP 编排与报告生成 |
| `pipeline.py::run_pipeline` | 实验脚本流水线 |

---

## 9. 运行方式

### 9.1 Windows 本地运行

Windows 不能直接运行 Gunicorn；本地开发使用 Flask 自带服务器：

```powershell
cd algorithms\group_n057_lraw_watermark
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install matplotlib

# 模型权重不入库，按根 README 的百度网盘链接下载后放到：
# assets\author_trained.pkl

.\.venv\Scripts\python.exe platform_service.py
```

默认监听 `http://127.0.0.1:8000`。本地不需要设置 `PUBLIC_BASE_URL`；代码会自动使用请求的 `url_root`。如需显式指定：

```powershell
$env:PUBLIC_BASE_URL = "http://127.0.0.1:8000"
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

如果本地前端页面要调用本地服务，需要把 `public/methods/group_n057_lraw_watermark.json` 的 endpoint 临时改成 `http://127.0.0.1:8000/process`；这只是本机调试，不要提交这个地址。

### 9.2 Linux 服务器运行

服务器使用 Gunicorn。长音频请求需要显式增加 worker timeout：

```bash
cd /opt/LRAW-reproduction
source .venv/bin/activate
PUBLIC_BASE_URL=http://<公网IP>:8000 \
  gunicorn --worker-class gthread -w 1 --threads 4 --timeout 600 \
  -b 0.0.0.0:8000 platform_service:app
```

`--timeout 600` 不能省略：4:15 的混合重录 + ECSP 在本机实测约 28 秒，默认 30 秒 worker timeout 余量太小。

### 9.3 前端

在仓库根目录：

```bash
npm install
npm run dev
```

前端默认读取 `public/methods/group_n057_lraw_watermark.json`；平台部署时 endpoint 应保持为公网算法服务地址。

---

## 10. 验证与已知边界

### 10.1 已完成的验证

- 12 秒 `music2.wav` + `mp3c_64`：5 个 artifacts，`locked=true`，盲 BER=0，`raw_ber=0.06198`，`snr_embed_db=26.5`；
- 4:15 MP3 + `mp3c_64`：`n_bits=28174`，`raw_ber=0.081139`，`locked=true`，盲 BER=0，文本匹配；
- 12.7 分钟音频：验证 `L=17` 分支，干净链路盲 BER=0；
- Python 包 `compileall` 通过；
- 仓库根目录 `npm test` 6/6 通过，`npm run build` 成功。

### 10.2 边界

- PSM 改变音高，条件分布收敛，BER 约 50%，属于方法层边界；
- 数字重录不是物理扬声器 + 手机实测；
- LSC 使用保留前缀语义，任意起点裁剪还要额外时间轴同步；
- 同步接口按单请求处理，极长音频和并发请求需要队列/更多 worker；
- 当前公网接口是 HTTP，HTTPS 页面调用会被 mixed content 拦截；
- m 序列表目前支持到 `L=20`，极端超长音频仍可能超出实际同步处理范围。

---

## 11. 模型权重与训练数据

- 推理权重：`assets/author_trained.pkl`，约 3.6 MB，**不在 Git 仓库**；按根 README 的百度网盘链接下载；
- 训练数据：`assets/dataset.xlsx`，**不在 Git 仓库**；只有 `scripts/train_model.py` 训练时需要，平台推理和 WebUI 不需要；
- 模型加载优先级：函数 `path` 参数 > 环境变量 `LRAW_MODEL_PATH` > `assets/author_trained.pkl`；
- 模型缺失时，`platform_service.py` 在 import 阶段就会报 `FileNotFoundError`，服务不会启动。

# 第A051组：基于 Noise-to-mask Ratio 损失的音频水印复现

## 1. 小组与项目

|信息|内容|
|---|---|
|组号|A051|
|成员与分工|李石屹，1320241091，GitHub：dthu7b8a-design；独立负责论文复现、数据与训练、实验验证、接口接入、部署和演示|
|方法 ID|`group_a051_nmr_watermark`|
|提交分支|[group_a051_submission](https://github.com/dthu7b8a-design/algorithm_research_frontend/tree/group_a051_submission)|
|选题|Noise-to-mask Ratio Loss for Deep Neural Network based Audio Watermarking|

项目为音频嵌入256或512位二进制消息，再从保存后的音频独立提取。提供 MSE256、NMR256、MSE512、NMR512 四个从头正式训练的模型，比较频谱均方误差与听觉掩蔽损失对音质的影响。

**当前验收状态（2026-11-01）：60候选训练和四模型选模通过检查；四模型RWC客观评价、本地及华为云HTTPS前端真实处理已完成。四模型公网嵌入、下载、重上传提取及整机重启后自动恢复均已实测通过。权重下载入口见下文；最终成果通过本组Fork分支提交。项目报告按本人要求暂不撰写。** 本组保留真实失败和复现差异，不声称得到与论文完全相同的数值。

## 2. 论文与代码来源

- 论文：Martin Moritz、Toni Olán、Tuomas Virtanen，2024，[Noise-to-mask Ratio Loss for Deep Neural Network based Audio Watermarking](https://arxiv.org/abs/2408.15553)。本组依据所选论文独立实现网络、损失、训练和推理，没有使用作者预训练权重。
- 本组源码：[algorithms/group_a051_nmr_watermark](algorithms/group_a051_nmr_watermark/README.md)，随本提交分支一起版本控制；[逐文件SHA-256清单](algorithms/group_a051_nmr_watermark/source-manifest.json)。
- 统一前端：[lym-lab/algorithm_research_frontend](https://github.com/lym-lab/algorithm_research_frontend)。共享组件及平台既有方法保留；本组新增配置、索引项、接口和真实样例。
- PEAQ 独立参考：[Kabal MATLAB镜像](https://github.com/sungminlee114/PEAQ_matlab/tree/8e122bde101eb476f6b4374725d192810583b840)，教育/研究用途，原参考源码未修改、不随本组源码重新打包。
- 听评工具：[webMUSHRA](https://github.com/audiolabs/webMUSHRA/tree/8c353f738aa88fa9dee0dc1aa7d18d2a06441cb5)，保留其许可证，按需独立获取。
- 数据：[FMA](https://github.com/mdeff/fma)、[RWC](https://zenodo.org/records/17177919)。原始音乐及完整数据集不上传本仓库；RWC遵循CC BY-NC 4.0。本组公开样例由程序生成，可自由分发。

## 3. 本组实现与验证

|功能|实际完成情况|证据|
|---|---|---|
|论文核心流程|44.1kHz单声道，STFT，U-Net嵌入，iSTFT后重新分析提取，MSE/NMR+BCE|[模型与损失源码](algorithms/group_a051_nmr_watermark/nmr_watermark/)、[复现规格](algorithms/group_a051_nmr_watermark/docs/reproduction.md)|
|正式训练|2消息长度×2损失×5α×3种子，60候选及12预热完整核验，四模型只按验证集规则选出|[最终检查点审计](algorithms/group_a051_nmr_watermark/outputs/final_training_audit.json)、[训练实际执行审计](algorithms/group_a051_nmr_watermark/outputs/paper_audit_20261021/training_runtime.json)|
|独立客观评价|四模型各100作品×30片段，3000段；BER/SNR/NMR全部保留|[实验数据目录](algorithms/group_a051_nmr_watermark/outputs/evaluation/)|
|PEAQ|MATLAB参考实现逐模型尝试3000段；各有相同30段结果未定义，ODG比较共同有效2970段，不填补分数|[共同样本与排除清单](algorithms/group_a051_nmr_watermark/outputs/final_analysis/peaq_cohort.json)、各模型逐行CSV和原始诊断|
|前端与HTTP|真实multipart上传、四模型选择、嵌入、浮点WAV下载、重新上传提取、可选已知消息BER、波形差值图|[方法配置](public/methods/group_a051_nmr_watermark.json)、[HTTP源码](algorithms/group_a051_nmr_watermark/nmr_watermark/service.py)、[四模型验证](public/samples/group_a051_nmr_watermark/validation.json)|
|听评准备|七段正式模型材料和匿名收集工具已在本地生成|[生成与收集工具](algorithms/group_a051_nmr_watermark/scripts/prepare_listening.py)；尚未招募听者，不提供虚构主观分数|
|华为云|四模型HTTPS实际往返、浏览器全流程及整机重启后恢复通过|[云端验收记录](docs/a051/cloud-validation.json)、[Dockerfile](Dockerfile.a051)、[Compose](compose.a051.yaml)、[HTTPS配置](Caddyfile.a051)|

## 4. 输入、输出与参数

|输入字段|类型|限制|
|---|---|---|
|`input_file`|单个WAV、FLAC或MP3音频|20 MiB；原音频最多180秒；解码后转44.1kHz单声道；提取允许额外一个尾部补零块|

|参数|默认值与范围|含义|
|---|---|---|
|`operation`|`embed`；或`extract`|嵌入或独立提取|
|`model`|`NMR256`；另可选`MSE256`、`MSE512`、`NMR512`|必须使用嵌入时相同的模型提取|
|`message`|空；填写时恰好256或512位0/1|嵌入留空会随机生成并在结果JSON记录；提取留空只输出逐块结果，填写已知消息则计算BER|

嵌入返回实际保存的FLOAT WAV、波形与差值PNG及结果JSON。JSON包含消息、原始/输出长度、保存后重新读取的BER等。提取返回逐块比特、概率及可选BER的JSON。末块补零保留在下载文件，原长度记录在JSON；裁掉尾部或改变起始位置可能改变提取结果。

接口：`POST /process`，`multipart/form-data`包含`input_file`、`method_id=group_a051_nmr_watermark`、JSON字符串`parameters`。成功返回`message`和1–3个`artifacts`；每项含`kind`、`url`、`label`、`filename`。参数错误、解码失败和超限有HTTP错误响应，文件与接口允许CORS。健康检查为`GET /health`。

## 5. 环境与安装

本地已验证：Windows、Python 3.10.8、PyTorch 2.6.0+cu124、RTX 4060 Laptop 8GB训练；推理服务使用CPU。前端Node.js 24.20.0、npm 11.19.0。依赖声明见[pyproject.toml](algorithms/group_a051_nmr_watermark/pyproject.toml)，训练环境版本记录见[Windows锁文件](algorithms/group_a051_nmr_watermark/requirements-windows.lock)。云端已验证Ubuntu 22.04、2核4GiB、CPU PyTorch 2.6.0+cpu、Docker 29.1.3、Compose 2.40.3；见[云端验收记录](docs/a051/cloud-validation.json)。

```bash
git clone https://github.com/dthu7b8a-design/algorithm_research_frontend.git
cd algorithm_research_frontend
git switch group_a051_submission
npm ci
cd algorithms/group_a051_nmr_watermark
python -m venv .venv
# Windows：.venv/Scripts/activate；Linux：source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

正式四模型已导出，总计约296 MiB，训练优化器不包含在推理权重中；权重不进入Git提交。公开下载：[MSE256](https://a051.119-8-40-242.sslip.io/weights/MSE256.pt)、[NMR256](https://a051.119-8-40-242.sslip.io/weights/NMR256.pt)、[MSE512](https://a051.119-8-40-242.sslip.io/weights/MSE512.pt)、[NMR512](https://a051.119-8-40-242.sslip.io/weights/NMR512.pt)、[SHA-256清单 selected.json](https://a051.119-8-40-242.sslip.io/weights/selected.json)。固定公网带宽1Mbps，全部权重理论传输时间约42分钟；支持HTTP Range。下载后将四个`.pt`及`selected.json`置于算法目录`weights/`，执行：

```bash
python -m nmr_watermark.verify_weights weights/selected.json
```

只有四个SHA-256全部匹配才启动服务。不要加载不可信来源的PyTorch检查点。

## 6. 启动与第一次处理

在算法目录启动CPU服务：

```bash
python -m nmr_watermark.service --port 7862
```

另开终端，在仓库根目录运行：

```bash
npm run dev
```

打开终端显示的前端地址，选择“A051：NMR感知损失音频水印”。提交配置默认调用已验证的公网HTTPS接口。本地运行服务时，先在仓库根目录执行 `node scripts/configure_a051.mjs http://127.0.0.1:7862/process`，再启动前端。点击“用样例试一试”，保持`embed`和`NMR256`，点击“开始处理”，试听并下载结果；再重新选择下载的`watermarked.wav`，切换`extract`，保留256位已知消息并运行，下载JSON检查BER。

实际服务结果存于算法目录`outputs/service/<随机任务编号>/`；原上传副本解码后删除，结果保留24小时，在后续请求时清理过期任务。取消等待仅取消浏览器请求，服务器会完成当前计算；一次只执行一项推理任务。180秒音频的耗时与CPU有关，本机三秒样例的浏览器实际嵌入约5.3秒；不据此保证云端时延。

在线演示：[华为云A051方法](https://a051.119-8-40-242.sslip.io/#/methods/group_a051_nmr_watermark)，[健康接口](https://a051.119-8-40-242.sslip.io/health)。实例位于中国香港，已付费至2026-12-30 23:59:59（北京时间），自动续费关闭。通过代金券抵扣，现金支付0元。

云端整合构建命令在仓库根目录执行，先将正式权重放在根目录`weights/`；Caddy会申请免费Let’s Encrypt证书，使用者须同意其订户协议；更换主机时同步修改域名及DNS：

```bash
node scripts/configure_a051.mjs https://a051.119-8-40-242.sslip.io/process
docker compose -f compose.a051.yaml up --build -d
docker compose -f compose.a051.yaml logs -f
```

镜像包含统一前端和CPU算法服务，后端只映射到127.0.0.1:7861，Caddy开放80/443并持久化证书。两个容器使用 `unless-stopped`；Docker已设开机启动，整机重启后实际自动恢复并重新验证四模型。前端HTTPS与接口HTTPS同源。

## 7. 真实结果与案例

|RWC独立测试|BER|平均SNR / dB|线性NMR均值转dB|PEAQ平均ODG|
|---|---:|---:|---:|---:|
|MSE256|0.0001158854|16.2904|21.6550|-2.0729|
|NMR256|0.0002773438|20.0385|-15.6306|-0.3267|
|MSE512|0.0026764323|15.1337|21.6166|-2.1326|
|NMR512|0.0022747396|16.0659|-13.7145|-0.4594|

BER/SNR/NMR每模型均为3000段。ODG为四模型共同有效2970段，越接近0表示参考模型预测的损伤越小。未定义30段不是0分也不是满分，详见排除清单及原始诊断。当前测试中NMR模型的ODG和NMR改善；256位的RWC BER并非更低，须同时看误码与音质。论文表1为另一组选择/验证口径，不能把本表当作作者原始数字。

- [逐项比较CSV](algorithms/group_a051_nmr_watermark/outputs/final_analysis/comparison.csv)
- [SNR箱线图](algorithms/group_a051_nmr_watermark/outputs/final_analysis/snr_values.png)
- [PEAQ箱线图](algorithms/group_a051_nmr_watermark/outputs/final_analysis/odg_values.png)，须线为5%/95%，不画离群点。
- [掩蔽与噪声案例图](algorithms/group_a051_nmr_watermark/outputs/case/mask_noise.png)、[案例采样位置与指标](algorithms/group_a051_nmr_watermark/outputs/case/case.json)。原始RWC音频不附在源码中。

|公开演示|输入|实际输出|复现参数|
|---|---|---|---|
|三秒生成和弦|[input.wav](public/samples/group_a051_nmr_watermark/input.wav)|[NMR256_output.wav](public/samples/group_a051_nmr_watermark/NMR256_output.wav)、[波形差值](public/samples/group_a051_nmr_watermark/NMR256_comparison.png)、[验证JSON](public/samples/group_a051_nmr_watermark/NMR256_result.json)|`embed`、`NMR256`、01交替256位|

样例由`scripts.validate_service`生成，真实四权重逐一推理，下载后重新上传提取，并非预录结果替代用户输入。该合成样例只验证流程，不替代RWC性能实验。

## 8. 验证与局限

已执行算法测试40项通过、平台测试7项通过、前端构建通过；60候选及12预热实际优化器步数和日程检查无错误。浏览器完成了样例填入、实际HTTP嵌入、结果预览、下载、重新上传和独立提取。可复查：

```bash
# 仓库根目录
npm test
npm run build
# 算法目录
python -m pytest -q
python -m scripts.validate_service
```

论文没有公开所有实现细节。本组披露单位化STFT及MSE尺度、191500 bps筛选容差、U映射边界、预热排除与共享、固定验证消息和图文冲突等假设或差异，详见[原文逐项审计](algorithms/group_a051_nmr_watermark/docs/audit-20261021.md)。不能声称逐点等同作者实现；若改变训练相关假设，必须独立新实验，不能沿用现有权重冒充。

不处理抗攻击、自动同步、纠错或水印存在性检测；重采样、压缩、裁剪和改变分块可能影响提取。没有真实主观听评分数；PEAQ不能代替人类评分。云端四模型HTTPS往返及重启后复测均已通过；浏览器NMR256实际下载后重新上传提取BER=0，三秒样例嵌入6.2秒、提取3.1秒。该耗时仅是本次案例，不保证长音频或并发时延。

## 9. 附加文档入口

- [9分钟汇报PPT（含已验收云端状态）](docs/a051/A051_9min_project_review_20261101.pptx)
- [本地浏览器真实下载与重上传提取验收](docs/a051/browser-validation.json)
- [云端四模型、HTTPS与整机重启后验收](docs/a051/cloud-validation.json)
- [云端浏览器嵌入截图](docs/a051/cloud-embedded.jpg)、[下载重传后独立提取截图](docs/a051/cloud-extracted.jpg)
- [云端逐模型请求与结果](algorithms/group_a051_nmr_watermark/outputs/cloud_validation_20261101/)、[整机重启后四模型复测](algorithms/group_a051_nmr_watermark/outputs/cloud_reboot_validation_20261101/)

- [算法目录与源文件](algorithms/group_a051_nmr_watermark/README.md)
- [复现规格与补充约定](algorithms/group_a051_nmr_watermark/docs/reproduction.md)
- [原文逐项审计](algorithms/group_a051_nmr_watermark/docs/audit-20261021.md)
- [课程项目验收清单](algorithms/group_a051_nmr_watermark/docs/completion-20261030.md)
- [完整实验统计口径](algorithms/group_a051_nmr_watermark/outputs/final_analysis/README.md)
- [部署与故障排查](algorithms/group_a051_nmr_watermark/docs/deployment-and-troubleshooting.md)
- [演示与真实听评准备](algorithms/group_a051_nmr_watermark/docs/presentation-and-listening.md)
- [平台算法提交指南](docs/算法提交指南.md)、[接口规范](docs/算法接入规范.md)、[自查清单](docs/接入自查清单.md)

项目报告和评分问卷不混入本轮代码提交；报告将在项目验收完成、本人明确要求后撰写。成果PR按指南保持打开，不自行合并或关闭。

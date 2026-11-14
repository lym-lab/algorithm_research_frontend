# 部署与故障排查

## 环境和权重

本机训练环境为Python3.10、PyTorch2.6.0 CUDA12.4；锁文件为requirements-windows.lock。优先先安装对应CUDA torch wheel，再安装项目依赖。CPU云端使用Dockerfile中的CPU wheel，不直接套Windows CUDA锁文件。完整训练需要已校验数据、CUDA及充足磁盘；仅推理不需要FMA、RWC、MATLAB。

正式训练完成后导出四权重至weights，selected.json记录相对路径与SHA256。上传前运行 `python -m nmr_watermark.verify_weights weights/selected.json`。不要把pilot权重重命名为正式权重。GitHub提交代码、配置及说明；音乐数据、缓存、个人听评记录、密码和大型权重不进入仓库。README的权重下载链接须在真实上传后填写。

## 云端执行顺序

1. 确认课程提供的实例类型、Linux架构、可用磁盘、CPU内存和网络规则。现有镜像面向x86 CPU；昇腾资源需要另行适配验证。
2. 安装课程允许的Docker环境，放入项目代码及四权重，校验SHA256。
3. 在课程前端Fork仓库根目录把权重放入 `weights/`，运行 `node scripts/configure_a051.mjs https://a051.119-8-40-242.sslip.io/process`，再运行 `docker compose -f compose.a051.yaml up --build -d`。更换主机时同步修改 `Caddyfile.a051` 的域名和解析。Caddy证书申请需同意Let’s Encrypt订户协议。
4. 检查 `/health` 健康端点，以及 `docker compose -f compose.a051.yaml ps` 和 `logs --tail 100`。后端只映射127.0.0.1:7861，公网由Caddy提供80/443。不要将本地训练控制页7863或训练目录开放到互联网；云端只需演示推理服务。
5. 从另一台设备实际完成上传、嵌入、下载、重新上传和提取，保存时间、URL、版本、权重哈希、请求结果及截图。
6. 确保Docker开机启动、容器为 `unless-stopped`，实际重启整台实例后再验证四模型。只有上述真实操作完成，才能称“华为云部署完成”。

2026-11-01已实际验收：[在线方法](https://a051.119-8-40-242.sslip.io/#/methods/group_a051_nmr_watermark)、[健康接口](https://a051.119-8-40-242.sslip.io/health)。香港2核4GiB、Ubuntu22.04、CPU torch2.6.0+cpu，四模型完整权重包SHA256一致，公网四模型真实往返以及整机重启后四模型复测均通过；浏览器NMR256嵌入6.2秒、独立提取3.1秒且BER=0。固定1Mbps，付费实例到2026-12-30 23:59:59北京时间，自动续费关闭，现金支付0元。证据见 `outputs/cloud_validation_20261101` 与 `outputs/cloud_reboot_validation_20261101`；账单、账号截图和证书私钥不进入公开Git。

首次镜像构建因CPU PyTorch索引缺少 `flit_core` 失败。实际修复为先更新pip、固定 `torch==2.6.0+cpu`，并给CPU索引添加官方PyPI作为额外来源。CloudShell大文件上传遇到WebSocket1006后改用16MiB分片，服务器合并后校验完整SHA256再解压。证书首次CAA查询超时，由同一Caddy任务自动重试后签发，未关闭TLS验证。

## 运行与恢复

双击“打开训练进度.cmd”启动本地进度页。关机前点停止并等进程退出；继续按钮恢复完整epoch断点。未完成的一轮重跑。不要同时在命令行手动启动第二条run_full流水线。核心源文件的冻结哈希不一致时会阻止正式训练；不要为了继续运行而盲目覆盖冻结记录。

课程接口使用 `python -m nmr_watermark.service --registry weights/selected.json --port 7862`；独立Gradio页面仍可用 `python -m nmr_watermark.cli ui --registry weights/selected.json`。消息只能是规定长度二进制串；嵌入留空生成随机消息，提取留空不计算BER。完整FLOAT WAV保留尾部补零块，原始长度在JSON中。对齐条件失效时BER会升高，不能据此断言水印从未存在。

## 常见问题

|现象|先检查|处理|
|---|---|---|
|网页打不开|电脑是否重启、7863服务是否存在|双击启动脚本；服务启动不等于训练启动|
|显示进程存活但指标不变|完整epoch是否结束、CPU/GPU和日志是否有活动|指标只在训练加验证结束写；不要仅凭状态时间判断卡死|
|CUDA OOM|其他GPU程序占用、当前阶段|记录错误并停止重复启动；不要在不同正式候选之间随意改变batch size|
|找不到权重|selected.json路径及四文件是否完整|重新下载、校验SHA256；不以调试权重替代|
|坏MP3警告|是否已被筛选记录排除、有无后续异常堆栈|查看清单和pipeline错误日志，警告不等同于整个训练失败|
|选择四模型失败|是否60候选完成、严格BER条件是否满足|如实报告，不能放宽条件或用测试集挑选|
|PEAQ缺失|MATLAB许可、路径、48k评价副本及CSV|修正环境后重跑评价；缺失ODG不得填零|
|关机后重复上一轮|last.pt保存轮数|这是epoch边界恢复的预期行为|
|云CPU推理慢|音频时长、CPU资源|先演示短音频，记录实际耗时，不改写实验结果|

## 最终分析

完整四模型RWC和MATLAB评价后，执行 `python -m scripts.analyze_final`。工具在输出前检查3000唯一索引、每曲30段、四模型相同清单、PEAQ对齐、有限指标及CSV与summary一致；失败时先排查数据，不能跳过检查。输出到outputs/final_analysis，不覆盖训练记录。

课程验收报告、9分钟汇报、贡献说明、真实部署截图、可运行GitHub地址和权重下载链接应逐项检查。报告中的原文数值、本次验证、本次RWC测试以及主观听评分开呈现。

# 复现步骤

本文说明如何在一台 Linux GPU 服务器上从零复现表 1。按工作量分三档：

| 档 | 内容 | GPU 时间（单卡 RTX 3090 级） |
|---|---|---|
| 一 | 安装、正确性检查（kNN-VC 与官方实现逐数一致，原始语音的 EER、WER 与 VPC 2024 官方数字一致）、各类任务测速 | 约 1–2 h（另有约 1 h 下载） |
| 二 | 用网盘里的权重，只做匿名化和评测，复现其中一个或几个系统 | 每个系统约 7–20 h |
| 三 | 全部系统，包括训练 WavLM 转换模型 | 约 200–300 h，可多卡并行 |

各步都可以中断后重跑：匿名化按句续跑，训练从最近的检查点续训，安装脚本跳过已完成的步骤。

## 0. 需要什么

- **系统：** Ubuntu 22.04，root 权限，NVIDIA 驱动支持 CUDA 11.8 及以上。
- **GPU：** 显存 12 GB 以上；训练 WavLM 转换模型（第 6 节）需要 24 GB。
- **磁盘：** 工作目录所在的盘至少 350 GB，读写不低于 200 MB/s（安装脚本会先测，低于这个速度就停下：攻击者说话人模型的训练要反复读取几十 GB 的音频，慢盘上一次要多花数小时）。
- **网络：** 能访问 Hugging Face（或镜像 `https://hf-mirror.com`）、PyTorch 与 conda 的软件源；GitHub 访问不畅时见第 1 节的 `GH_ASSETS`。
- **账号：** 一个已接受 [ProgramComputer/voxceleb](https://huggingface.co/datasets/ProgramComputer/voxceleb) 使用条款的 Hugging Face 账号（VoxCeleb1 用作 kNN-VC 的目标说话人池）。

下文约定 `W` 为工作目录（例如 `/root/a082`），`R=$W/vpc2024` 为 VPC 2024 评测框架的目录。除安装外，所有 Python 命令都在本仓库根目录下、激活评测框架的环境之后运行：

```bash
cd $R && source env.sh && cd -
```

## 1. 安装（约 1 h，大部分时间在下载）

```bash
huggingface-cli login                     # 粘贴只读 token
HF_ENDPOINT=https://hf-mirror.com bash scripts/cloud_setup.sh $W 2>&1 | tee setup.log
```

脚本依次：测磁盘速度；在后台并行下载 LibriSpeech train-clean-360（核对 OpenSLR 公布的 md5）与 VoxCeleb1；从官方仓库取 VPC 2024 评测框架（固定到提交 `0ef5506`）并建环境；安装本项目、Whisper 和 Piper；按官方清单重建 libri_dev / libri_test 并逐句核对时长；准备评测框架的预训练模型、kNN-VC 的 WavLM 与 HiFi-GAN 权重、Piper 音色；准备 IEMOCAP。最后打印 `setup done`。

与评测框架原安装脚本不同的两处（不影响结果，否则装不上或跑不了）：

- PyTorch 用同一版本 2.0.1 的 CUDA 11.8 构建：CUDA 11.7 构建在 RTX 40 系显卡上调用 cuFFT 会报 `CUFFT_INTERNAL_ERROR`；
- 框架的依赖只写了版本下限，直接安装会装上 2026 年的大版本（NumPy 2、pandas 3、transformers 5）而无法运行，脚本把它们固定在 2024 年的版本（见脚本中的 `PINS`）。

可选的环境变量（详见脚本开头）：服务器访问 GitHub 很慢时，可以在别处下载脚本列出的几个文件，放进一个目录，用 `GH_ASSETS=<目录>` 指定；服务器上有平台预设的代理而它很慢时，用 `PROXY=off`；用 `DEMO_DATA=skip` 跳过 LibriTTS-R（网页演示的目标池改用 LibriSpeech test-clean，它随评测数据下载，所以一般可以跳过）。

## 2. 正确性检查与测速（约 1–1.5 h）

```bash
bash scripts/first_checks.sh $W           # 汇总在 $W/logs/first_checks.txt
```

| 检查 | 通过的标准 |
|---|---|
| kNN-VC 与官方实现（bshall/knn-vc）对同一句话的输出 | 退出码 0：特征与波形的最大差都不超过 1e-3 |
| 原始语音（表 1 第 0 行）走官方评测 | EER 10.511 / 0.931 / 8.761 / 0.418（libri dev 女、男，test 女、男），WER 1.807 / 1.844（dev、test），每个数与官方相差不超过 0.05 |
| 测速 | kNN-VC、级联的实时倍数，WavLM 转换模型训练的每秒步数与显存峰值 |

第二项对不上，说明重建的评测数据或环境有问题，后面的结果都不可信，要先查清楚。

另可运行单元测试（几十秒，不需要 GPU）：`pytest -q tests`。

## 3. kNN-VC 目标池（约 1–2 h）

```bash
python -m a082.knnvc.pool scan-voxceleb $W/corpora/voxceleb1/wav $W/pool_vox1.json
python -m a082.knnvc.pool build $W/pool_vox1.json $W/cache/pool_vox1
```

每个说话人生成一个 `.pt`（WavLM 第 6 层特征）和一个 `.frames` 文件；不足 5 分钟有效语音的说话人在使用时被排除。多卡时每张卡运行一个 `--shard i/n`，全部完成后再开始匿名化。

## 4. kNN-VC 系统（表 1 第 1–3 行）与级联系统（第 4 行）

```bash
python scripts/calibrate_noise.py --vpc-root $R --pool $W/cache/pool_vox1    # 第 3 行加噪参数的标定（约 0.5 h）
for s in knn1 knn2 knn3 knn1m; do
  python -m a082.knnvc.anonymize --config configs/$s.yaml --pool $W/cache/pool_vox1 --vpc-root $R --sets all
  python -m a082.eval.vpc run --vpc-root $R --suffix _$s
done
python -m a082.cascade.anonymize --vpc-root $R --piper-model $W/cache/piper/en_US-libritts_r-medium.onnx \
       --transcripts $W/transcripts --sets all
python -m a082.eval.vpc run --vpc-root $R --suffix _tts
```

`calibrate_noise.py` 打印各组参数在 dev 集上的 WER，把最接近论文第 3 行 WER（11.993）的一组写进 `configs/knn3.yaml`。`knn1m` 是我们加的变体（多集合池化），论文没有对应数字。每个系统的匿名化约 4–10 h（级联 15–30 h），评测约 3–11 h，其中大部分是在匿名化后的 train-clean-360 上重新训练攻击者的说话人验证模型。

**多卡并行：** 匿名化加 `--shard i/n`，每卡一个进程，全部完成后再用 `--finalize-only` 运行一次；评测要同时跑几个时，每个加 `--workspace auto` 并用 `CUDA_VISIBLE_DEVICES` 各占一张卡（每次评测在独立的框架副本里运行，结束后把结果移回 `$R/exp`）。

## 5. 混合系统（表 1 的三个混合比例与图 4）

```bash
python -m a082.admix --vpc-root $R --p 0.2 0.325 0.4 0.1 0.15 0.25 0.3 0.35 0.5
for p in 0p2 0p325 0p4 0p1 0p15 0p25 0p3 0p35 0p5; do
  python -m a082.eval.vpc run --vpc-root $R --suffix _mix$p
done
```

混合系统直接链接第 1 行和第 4 行的输出，不重新匿名化；每个比例重训一次攻击者模型。

## 6. WavLM 转换模型（表 1 第 5–8 行）

**用网盘里的权重（第二档）：** 把 `wc5.pt`–`wc8.pt` 放进 `$W/wc/`，然后

```bash
for v in 5 6 7 8; do
  python -m a082.wavlm_conv.train infer --ckpt $W/wc/wc$v.pt --suffix _wc$v --vpc-root $R --sets all
  python -m a082.eval.vpc run --vpc-root $R --suffix _wc$v
done
```

**从头训练（第三档，约 25 h）：**

```bash
python -m a082.wavlm_conv.train prepare-target --manifest $W/pool_vox1.json --pool $W/cache/pool_vox1 --out $W/wc/target.pt
python -m a082.wavlm_conv.train pretrain-spk --vpc-root $R --out $W/wc/spk.pt
python -m a082.wavlm_conv.train train --config configs/wc5.yaml --vpc-root $R --target $W/wc/target.pt \
       --manifest $W/pool_vox1.json --out $W/wc/wc5 --amp
for v in 6 7 8; do
  python -m a082.wavlm_conv.train train --config configs/wc$v.yaml --vpc-root $R --target $W/wc/target.pt \
         --manifest $W/pool_vox1.json --spk $W/wc/spk.pt --init $W/wc/wc5/last.pt --out $W/wc/wc$v --amp
done
```

`--amp` 用 fp16 混合精度（说话人分类器保持 fp32），速度约为 fp32 的 2.4 倍；表 1 第 6–8 行的模型都是这样训练的。第 6–8 行的配置里有 `schedule_offset: 100000`，学习率调度接着第 5 行的第 100k 步走、不重新预热；不要去掉这一项，否则峰值学习率会高出 2–5 倍，结果与表 1 不同。训练每 5000 步保存一次 `last.pt`，中断后用同一条命令续训。训练完成后按上面的 `infer` 与评测命令运行（`--ckpt $W/wc/wc$v/last.pt`）。

## 7. 扩展实验

| 实验 | 命令 |
|---|---|
| 区分混合系统中两类输出的分类器 | `python -m a082.bonus.sysclf train --vpc-root $R`；`python -m a082.bonus.sysclf eval --vpc-root $R --sets eval --json $W/bonus/sysclf_eval.json` |
| 先按系统分离、再分别训练的攻击者 | `python -m a082.bonus.sysclf split --vpc-root $R --mix-suffix _mix0p4 --p 0.4`；`python -m a082.bonus.attack train --vpc-root $R --mix-suffix _mix0p4 --subset X`（X 依次取 `pred_knn`、`pred_tts`、`true_knn`、`true_tts`） |
| 按测试句来源拆分 EER | `python -m a082.bonus.attack analyse --vpc-root $R --mix-suffix _mix0p4 --labels work/bonus/labels/mix0p4 --models single=exp/asv_anon_mix0p4 knn=exp/asv_anon_mix0p4__pred_knn tts=exp/asv_anon_mix0p4__pred_tts --json $W/bonus/attack_mix0p4.json` |
| 改进 A：级联输出再经 kNN-VC | `python -m a082.knnvc.anonymize --config configs/ttsknn.yaml --source-suffix _tts --pool $W/cache/pool_vox1 --vpc-root $R --sets all`；`python -m a082.admix --vpc-root $R --tts-suffix _ttsknn --out-suffix _mixA0p4 --p 0.4`，然后评测并重复上面两行 |
| 改进 B：按说话人固定所用系统 | `python -m a082.admix --vpc-root $R --level spk --p 0.4`，评测 `_mix0p4spk`，分离与攻击时加 `--level spk` |
| 有损转码后的隐私与可用性 | `python -m a082.bonus.transcode make --vpc-root $R --suffix _knn1 --codec opus12k`；`python -m a082.bonus.transcode evaluate --vpc-root $R --suffix _knn1 --codec opus12k`；对 `_orig`、`_tts`、`_mix0p4` 同样运行 |

## 8. 汇总

```bash
python -m a082.eval.vpc report --vpc-root $R --out results/table1.md --json results/table1.json --fig4 results/fig4.png
```

`results/table1.md` 逐行列出我们的结果、论文的数字和判定；`results/fig4.png` 是混合比例与 EER、UAR 的关系图。网页演示读取 `results/table1.json`，其目标说话人池用 LibriSpeech test-clean 的 6 个说话人（1089、1188、121、1995、2961、4507）建（网盘里已有建好的 `pool_demo/`）：

```bash
python -m a082.knnvc.pool scan-tree $W/corpora/LibriSpeech/test-clean $W/pool_demo_all.json --ext .flac
python -c "import json; m=json.load(open('$W/pool_demo_all.json')); json.dump({k: m[k] for k in ('1089','1188','121','1995','2961','4507')}, open('$W/pool_demo.json','w'))"
python -m a082.knnvc.pool build $W/pool_demo.json $W/cache/pool_demo
```

## 数据许可

VoxCeleb 与 IEMOCAP 的原始音频和匿名化后的音频都不随本项目分发；网盘只有模型权重、网页演示用的目标池（LibriSpeech test-clean 的 6 个说话人，CC BY 4.0）和 LibriSpeech test-clean 的几句样例。

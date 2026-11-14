"""Course demonstration UI, backed by the same implementation as the CLI."""
import json
import os
import uuid
from pathlib import Path
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from .audio import read_audio, chunks
from .inference import embed_file, load_model, extract_array, parse_message
from .model import SAMPLE_RATE, SpectralTransform
from .psychoacoustics import NMRLoss


def build_app(registry):
    registry = Path(registry)
    models = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    output = Path(os.environ.get("NMR_OUTPUT_DIR", "outputs/ui"))
    output.mkdir(parents=True, exist_ok=True)

    def checkpoint(name):
        if name not in models:
            raise gr.Error("尚无可用模型，请先完成训练并生成模型清单。")
        path = Path(models[name]["checkpoint"])
        path = path if path.is_absolute() else registry.parent/path
        if not path.exists():
            raise gr.Error("模型权重不存在，请按 README 下载并校验。")
        return path

    def plot_pair(x, y):
        y = y[:len(x)]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), layout="constrained")
        stride = max(1, len(x)//15000)
        t = np.arange(0, len(x), stride)/SAMPLE_RATE
        axes[0].plot(t, x[::stride], label="Original", alpha=.7)
        axes[0].plot(t, y[::stride], label="Watermarked", alpha=.6)
        axes[0].legend()
        axes[0].set_ylabel("Amplitude")
        axes[1].plot(t, (y-x)[::stride])
        axes[1].set_ylabel("Difference")
        axes[2].specgram(y, NFFT=1024, Fs=SAMPLE_RATE, noverlap=512, cmap="magma")
        axes[2].set_ylim(0, 10000)
        axes[2].set_ylabel("Frequency (Hz)")
        axes[2].set_xlabel("Time (s)")
        return fig

    def embed(source, name, message):
        if not source:
            raise gr.Error("请上传音频")
        x = read_audio(source)
        if len(x)/SAMPLE_RATE > 180:
            raise gr.Error("演示界面每次最多处理180秒；更长音频请使用命令行。")
        path = output/f"{uuid.uuid4().hex}.wav"
        info = embed_file(checkpoint(name), source, path, message)
        fig = plot_pair(x, read_audio(path))
        return str(path), str(path), info["message"], info, fig, str(path.with_suffix(".json"))

    def extract(source, name, expected):
        if not source:
            raise gr.Error("请上传音频")
        model, config = load_model(checkpoint(name))
        audio = read_audio(source)
        if len(audio)/SAMPLE_RATE > 180:
            raise gr.Error("演示界面每次最多处理180秒")
        probability = extract_array(model, audio)
        decoded = (probability >= .5).astype(int)
        info = {"blocks": len(decoded), "bits_per_block": model.bits, "pilot": config.get("pilot", False),
                "note": "此输出不是水印存在性检测；需要与嵌入时相同的起点和块边界。"}
        if expected.strip():
            expected_bits = parse_message(expected, model.bits)
            info["ber"] = float(np.mean(decoded != expected_bits))
            info["per_block_ber"] = np.mean(decoded != expected_bits, axis=1).tolist()
        text = "\n".join(f"块 {i+1}: " + "".join(map(str, row)) for i, row in enumerate(decoded))
        path = output/f"{uuid.uuid4().hex}_bits.json"
        path.write_text(json.dumps({**info, "decoded": decoded.tolist(), "probabilities": probability.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
        return text, info, str(path)

    def compare(original, marked):
        if not original or not marked:
            raise gr.Error('请同时上传原音频与水印音频')
        x, y = read_audio(original), read_audio(marked)
        if max(len(x), len(y)) > 181*SAMPLE_RATE:
            raise gr.Error('对比最多处理180秒原音频及其尾部补零块')
        if len(y) < len(x):
            raise gr.Error('水印音频短于原音频，无法按原始位置对齐比较')
        xb, _ = chunks(x)
        yb, _ = chunks(y)
        if xb.shape != yb.shape:
            raise gr.Error('音频块数不一致，请提供未裁剪的对应文件')
        stft, nmr = SpectralTransform(), NMRLoss()
        values = []
        with torch.no_grad():
            for i in range(0, len(xb), 8):
                values.extend(nmr(stft(torch.from_numpy(yb[i:i+8])), stft(torch.from_numpy(xb[i:i+8]))).tolist())
        snr = 10*np.log10(max(float(np.sum(x*x)), 1e-20)/max(float(np.sum((x-y[:len(x)])**2)), 1e-20))
        return {'snr_db': float(snr), 'nmr_db': float(10*np.log10(max(np.mean(values),1e-30))),
                'peaq': '仅在本机 MATLAB 批量评价中提供', 'alignment': '假定起始采样点一致；未进行自动对齐'}, plot_pair(x,y)

    with gr.Blocks(title="NMR 音频水印论文复现") as app:
        gr.Markdown("# NMR 音频水印\nIS2 2024 论文复现：比较 MSE 与听觉掩蔽损失对水印音质的影响。")
        if not models:
            gr.Markdown("**状态：正式模型尚未训练完成，当前不可执行推理。**")
        elif "pilot" in str(registry).lower() or any("PILOT" in name for name in models):
            gr.Markdown("**当前为小样本调试模型，不代表论文完整实验结果。**")
        model_choice = gr.Dropdown(list(models), value=next(iter(models), None), label="模型")
        with gr.Tab("嵌入水印"):
            source = gr.File(label="上传音频", file_types=[".wav", ".flac", ".mp3"])
            message = gr.Textbox(label="二进制水印（留空随机生成）", lines=3)
            button = gr.Button("嵌入并验证下载文件", variant="primary")
            preview = gr.Audio(label="水印音频试听", type="filepath")
            download = gr.File(label="下载完整浮点 WAV")
            actual_message = gr.Textbox(label="本次水印，请保存用于验证", lines=3)
            summary = gr.JSON(label="保存后重新提取的结果")
            figure = gr.Plot(label="音频对比")
            metadata = gr.File(label="下载水印信息与原始长度")
            button.click(embed, [source, model_choice, message], [preview, download, actual_message, summary, figure, metadata], api_name="embed")
            gr.Markdown("音频转为44.1 kHz单声道。下载文件保留最后一个补零块，原始长度记录在JSON中。")
        with gr.Tab("独立提取"):
            marked = gr.File(label="重新上传水印音频", file_types=[".wav", ".flac", ".mp3"])
            expected = gr.Textbox(label="已知原始水印（可选，用于计算误码率）", lines=3)
            decode = gr.Button("提取水印")
            bits = gr.Textbox(label="逐块提取结果", lines=8)
            stats = gr.JSON(label="验证信息")
            bitfile = gr.File(label="下载详细结果")
            decode.click(extract, [marked, model_choice, expected], [bits, stats, bitfile], api_name="extract")
        with gr.Tab("实验说明"):
            gr.Markdown("比较模型：MSE256、NMR256、MSE512、NMR512。\n\n音质评价需要原音频；误码率需要已知水印。未经真实听评不生成主观评分。\n\n论文未处理不同步和抗攻击问题，裁剪、转码可能影响提取。")
            gr.JSON(models, label="当前模型清单及验证集指标")
        with gr.Tab('原音频对比'):
            reference = gr.File(label='原音频', file_types=['.wav','.flac','.mp3'])
            comparison = gr.File(label='对应水印音频', file_types=['.wav','.flac','.mp3'])
            compare_button = gr.Button('计算对齐音频的音质指标')
            quality = gr.JSON(label='客观指标')
            comparison_plot = gr.Plot(label='音频对比')
            compare_button.click(compare, [reference, comparison], [quality, comparison_plot], api_name='compare')
    return app

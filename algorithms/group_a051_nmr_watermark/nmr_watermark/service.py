"""HTTP adapter for the course algorithm platform, using the final four models."""
import argparse
import json
import shutil
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import read_audio, save_audio
from .inference import embed_array, extract_array, load_model, parse_message
from .model import SAMPLE_RATE, SEGMENT_SAMPLES
from .verify_weights import verify

METHOD_ID = 'group_a051_nmr_watermark'
MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 180
RETENTION_SECONDS = 86400


def create_app(registry='weights/selected.json', output='outputs/service', frontend=None):
    registry = Path(registry)
    verify(registry)
    entries = json.loads(registry.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    app = FastAPI(title='A051 NMR Audio Watermark', version='1.0.0')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET', 'POST'],
                       allow_headers=['*'], expose_headers=['Content-Disposition'])

    @lru_cache(maxsize=4)
    def model_for(name):
        return load_model(registry.parent / entries[name]['checkpoint'])[0]

    @app.get('/health')
    def health():
        return {'status': 'ok', 'models': list(entries), 'retention_seconds': RETENTION_SECONDS}

    @app.post('/process')
    def process(input_file: UploadFile = File(...), method_id: str = Form(...),
                parameters: str = Form('{}')):
        folder = None
        try:
            if method_id != METHOD_ID:
                raise ValueError('未知方法 ID')
            params = json.loads(parameters)
            if not isinstance(params, dict) or set(params) - {'operation', 'model', 'message'}:
                raise ValueError('参数必须是仅含 operation、model、message 的 JSON 对象')
            operation = params.get('operation', 'embed')
            name = params.get('model', 'NMR256')
            text = params.get('message', '')
            if operation not in ('embed', 'extract') or name not in entries or not isinstance(text, str):
                raise ValueError('请选择嵌入或提取、四个正式模型之一，并提供文字格式的二进制消息')
            if text.strip():
                parse_message(text, int(name[-3:]))
            payload = input_file.file.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                return JSONResponse({'error': '文件超过20 MiB'}, status_code=413)
            with lock:
                now = time.time()
                for old in output.iterdir():
                    if old.is_dir() and len(old.name) == 32 and all(c in '0123456789abcdef' for c in old.name):
                        if now - old.stat().st_mtime > RETENTION_SECONDS:
                            shutil.rmtree(old)
                token = uuid.uuid4().hex
                folder = output / token
                folder.mkdir()
                source = folder / 'upload'
                source.write_bytes(payload)
                try:
                    info = sf.info(source)
                    if info.duration > MAX_SECONDS + (SEGMENT_SAMPLES / SAMPLE_RATE if operation == 'extract' else 0):
                        raise ValueError('原音频最多180秒；提取允许保留一个补零块')
                    audio = read_audio(source)
                except (sf.LibsndfileError, RuntimeError) as exc:
                    raise ValueError('无法解码音频，请上传有效的 WAV、FLAC 或 MP3') from exc
                finally:
                    source.unlink(missing_ok=True)
                model = model_for(name)
                artifacts = []
                result = {'model': name, 'operation': operation, 'sample_rate': SAMPLE_RATE,
                          'segment_samples': SEGMENT_SAMPLES, 'alignment_required': True,
                          'note': '提取需要相同的起点和分块边界；输出比特不代表检测到水印。'}
                if operation == 'embed':
                    message = parse_message(text, model.bits)
                    marked, length = embed_array(model, audio, message)
                    save_audio(folder / 'watermarked.wav', marked)
                    restored = read_audio(folder / 'watermarked.wav')
                    decoded = extract_array(model, restored) >= .5
                    result.update(message=''.join(map(str, message.astype(int))),
                                  original_samples=length, output_samples=len(marked),
                                  ber_after_save=float(np.mean(decoded != message)),
                                  per_block_ber=np.mean(decoded != message, axis=1).tolist(),
                                  wav_subtype='FLOAT', peak=float(np.abs(marked).max()))
                    artifacts.append({'kind': 'audio', 'filename': 'watermarked.wav', 'label': '水印音频'})
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 1, figsize=(10, 5), layout='constrained')
                    stride = max(1, len(audio) // 12000)
                    t = np.arange(0, len(audio), stride) / SAMPLE_RATE
                    axes[0].plot(t, audio[::stride], label='Original', alpha=.7)
                    axes[0].plot(t, restored[:len(audio):stride], label='Watermarked', alpha=.7)
                    axes[0].legend()
                    axes[0].set_ylabel('Amplitude')
                    axes[1].plot(t, (restored[:len(audio)] - audio)[::stride])
                    axes[1].set_ylabel('Difference')
                    axes[1].set_xlabel('Time (s)')
                    fig.savefig(folder / 'comparison.png', dpi=120)
                    plt.close(fig)
                    artifacts.append({'kind': 'image', 'filename': 'comparison.png', 'label': '波形与差值'})
                else:
                    probabilities = extract_array(model, audio)
                    decoded = probabilities >= .5
                    result.update(bits_per_block=model.bits, blocks=len(decoded),
                                  decoded=[''.join(map(str, row.astype(int))) for row in decoded],
                                  probabilities=probabilities.tolist())
                    if text.strip():
                        expected = parse_message(text, model.bits)
                        result.update(ber=float(np.mean(decoded != expected)),
                                      per_block_ber=np.mean(decoded != expected, axis=1).tolist())
                (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                artifacts.append({'kind': 'json', 'filename': 'result.json', 'label': '水印与验证信息'})
                for artifact in artifacts:
                    artifact['url'] = f'/files/{token}/{artifact["filename"]}'
                return {'message': '嵌入完成并已重读验证保存文件' if operation == 'embed' else '提取完成',
                        'artifacts': artifacts}
        except (ValueError, OSError) as exc:
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)
            return JSONResponse({'error': str(exc)}, status_code=400)
        finally:
            input_file.file.close()

    app.mount('/files', StaticFiles(directory=output), name='files')
    app.mount('/weights', StaticFiles(directory=registry.parent), name='weights')
    if frontend is not None:
        app.mount('/', StaticFiles(directory=frontend, html=True), name='frontend')
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', default='weights/selected.json')
    parser.add_argument('--output', default='outputs/service')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--frontend', help='Built course frontend directory')
    args = parser.parse_args()
    torch.set_num_threads(4)
    import uvicorn
    uvicorn.run(create_app(args.registry, args.output, args.frontend), host=args.host, port=args.port)


if __name__ == '__main__':
    main()

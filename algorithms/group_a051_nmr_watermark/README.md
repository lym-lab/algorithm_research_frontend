# A051 算法源码与实验入口

本目录提供本组独立复现的训练、推理、HTTP服务、测试与真实实验记录。详细运行步骤从仓库根目录 [README](../../README.md) 进入。

在本目录安装 `pip install -e .`，将四个最终模型及带 SHA-256 的 `selected.json` 放入 `weights/` 后，可执行：

```bash
python -m nmr_watermark.verify_weights weights/selected.json
python -m nmr_watermark.service --port 7862
python -m pytest -q
```

第三方 PEAQ、webMUSHRA 与音乐数据不随源码打包；版本、许可、下载和复现约定见 [复现规格](docs/reproduction.md)。公开演示使用本项目生成的合成和弦，正式音乐实验使用独立 RWC 3000 段。

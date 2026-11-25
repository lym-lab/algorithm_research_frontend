# LRAW 算法源码

- 方法 ID：`group_n057_lraw_watermark`
- 模型权重：`author_trained.pkl`，[百度网盘下载](https://pan.baidu.com/s/10fK98eCWkWq6c-WzWkVCtQ?pwd=5jnh)（提取码：`5jnh`），放到 `assets/author_trained.pkl`
- 训练数据：`dataset.xlsx` 不入库，来源见根 README
- 安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gunicorn matplotlib
```

- 启动：

```bash
PUBLIC_BASE_URL=http://121.36.39.138:8000 \
  gunicorn --worker-class gthread -w 1 --threads 4 -b 0.0.0.0:8000 platform_service:app
```

- 平台接口：`POST /process`，支持 33 个 `attack_preset`（31 档主矩阵 + 干净信道 + MP3C 128 kbps），返回 4–5 个 artifacts
- 健康检查：`GET /health`
- 说明：本目录只放源码，不放模型权重、作者数据集和运行产物；权重按课程要求放百度网盘。

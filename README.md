# PDF Image Table Cropper

一个可安装 CLI + SDK 工具：从 PDF 中检测并裁剪 `image / table / code / algorithm` 区域，输出 JPG 和 `metadata.json`。

不做 OCR，不提取正文文本。

## 1. 安装 (Install)

开发环境（推荐）：

```bash
pip install -e .
```

或普通本地安装：

```bash
pip install .
```

Python 建议 `>=3.10`。

## 2. 快速开始 (Quick Start)

CLI 方式（安装后）：

```bash
pdf-image-table-cropper \
  -i /path/to/paper.pdf \
  -o ./output
```

输出目录为：`./output/<pdf_stem>/`。

## 3. 常用命令 (Examples)

只导出表格，处理部分页码：

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --type table \
  --pages 1-5,8,10
```

默认模型（docling Heron，主检测）：

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --type both
```

启用 OpenDataLab 补充检测（默认关闭）：

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --enable-opendatalab
```

启用本地守护进程复用模型（避免每次命令冷启动）：

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --daemon-mode auto \
  --daemon-idle-seconds 300
```

提高清晰度（更慢）：

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --dpi 300
```

## 4. 参数速查 (CLI)

必填：

- `-i, --input-pdf`：输入 PDF 路径
- `-o, --output-dir`：输出根目录

核心可选参数：

- `--type {both,image,table,code,algorithm}`，默认 `both`
- `--pages all|1|1-3|1-3,7,10`，默认 `all`
- `--dpi`，默认 `200`
- `--imgsz`：OpenDataLab 补充模型输入尺寸，默认 `1280`
- `--conf`：OpenDataLab 补充模型置信度阈值，默认 `0.10`
- `--iou`：OpenDataLab 补充模型 IoU 阈值，默认 `0.45`
- `--device cpu|mps|cuda|cuda:0...`，默认自动选择
- `--no-merge`：关闭连通域合并（默认开启）
- `--heron-model MODEL_ID`：docling Heron 主检测模型（默认已启用）
- `--heron-conf`：Heron 主检测置信度阈值，默认 `0.5`
- `--enable-opendatalab`：启用 OpenDataLab YOLO 补充检测（默认关闭）
- `--metadata-file`，默认 `metadata.json`
- `--daemon-mode {off,auto,on}`：模型生命周期模式，默认 `off`
- `--daemon-socket`：本地 Unix Socket 路径
- `--daemon-idle-seconds`：守护进程空闲回收秒数，默认 `300`
- `--daemon-start-timeout`：守护进程启动等待秒数，默认 `12`
- `--daemon-run-timeout`：等待守护任务完成的超时秒数，`0` 表示不限时（默认 `0`）

模型下载相关：

- `--model-repo`：OpenDataLab 补充模型仓库（默认 `opendatalab/PDF-Extract-Kit-1.0`）
- `--model-file`：OpenDataLab 补充模型文件（默认 `models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt`）
- `--hf-cache-dir`
- `--hf-token`

## 5. 输出结构 (Output)

```text
output/
└── <pdf_stem>/
    ├── metadata.json
    ├── image/
    ├── table/
    ├── code/
    └── algorithm/
```

裁剪文件名：

```text
p{page}_{type}_{idx}_{x0}_{y0}_{x1}_{y1}[_merged].jpg
```

`metadata.json` 中每条 crop 记录包含：

- `content_type`
- `page_index`, `page_number`
- `page_size_pdf`
- `bbox_pdf`（PDF 坐标）
- `bbox_pixels`（像素坐标）
- `score`
- `image_path`
- `merged_from`（若发生合并）

## 6. 性能与调参 (Tuning)

- 漏检多：降低 `--conf`（如 `0.05~0.10`）
- 误检多：提高 `--conf`（如 `0.15~0.30`）
- 细节不够：提高 `--dpi`（通常到 `300`）
- 太慢或显存不足：降低 `--dpi` / `--imgsz`，或用 `--device cpu`
- 导出 `algorithm`：需要显式加 `--enable-opendatalab`

## 7. License（仓库代码 + 模型）

**仓库源码（source code）使用 MIT License。**

模型许可证：

- `docling-project/docling-layout-heron`：**Apache-2.0**
- `opendatalab/PDF-Extract-Kit-1.0`：**AGPL-3.0(非商用协议, 注意, 按需使用)**

## 8. 文档版本

- 中文（当前）：`README.md`
- English: `README.en.md`

## 9. SDK 用法 (SDK Usage)

```python
from pdf_cropper import CropJobConfig, crop_pdf, crop_pdf_simple

# 简单调用
result = crop_pdf_simple(
  input_pdf="paper.pdf",
  output_dir="./output",
  detect_type="table",
)

# 完整配置调用
config = CropJobConfig(
  input_pdf="paper.pdf",
  output_dir="./output",
  detect_type="both",
  pages="all",
  dpi=200,
  enable_opendatalab=False,
)
result = crop_pdf(config)
print(result["metadata_file"])
```

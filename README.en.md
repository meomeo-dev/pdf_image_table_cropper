# PDF Image Table Cropper

A pip-installable CLI + SDK tool that detects and crops
`image / table / code / algorithm` regions from PDFs,
then exports JPG files and `metadata.json`.

No OCR. No full-text extraction.

## 1. Install

Install from PyPI (recommended for production):

```bash
pip install pdf-image-table-cropper
```

Recommended for development:

```bash
pip install -e .
```

Or standard local install:

```bash
pip install .
```

Recommended Python version: `>=3.10`.

## 2. Quick Start

CLI mode after installation:

```bash
pdf-image-table-cropper \
  -i /path/to/paper.pdf \
  -o ./output
```

Output path: `./output/<pdf_stem>/`.

## 3. Common Examples

Export table regions only, with selected pages:

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --type table \
  --pages 1-5,8,10
```

Default model (docling Heron, primary detector):

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --type both
```

Enable OpenDataLab supplementary detection (off by default):

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --enable-opendatalab
```

Enable local daemon mode to reuse loaded models across commands:

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --daemon-mode auto \
  --daemon-idle-seconds 300
```

Increase rendering quality (slower):

```bash
pdf-image-table-cropper \
  -i paper.pdf \
  -o ./output \
  --dpi 300
```

## 4. CLI Reference

Required:

- `-i, --input-pdf`: input PDF path
- `-o, --output-dir, --storage-root`: output root directory (alias)

Core optional arguments:

- `--type {both,image,table,code,algorithm}` (default: `both`)
- `--pages all|1|1-3|1-3,7,10` (default: `all`)
- `--dpi` (default: `200`)
- `--imgsz`: OpenDataLab supplementary input size (default: `1280`)
- `--conf`: OpenDataLab supplementary confidence threshold (default: `0.10`)
- `--iou`: OpenDataLab supplementary IoU threshold (default: `0.45`)
- `--device cpu|mps|cuda|cuda:0...` (default: auto)
- `--no-merge`: disable connected-components merging (enabled by default)
- `--heron-model MODEL_ID`: primary docling Heron model (enabled by default)
- `--heron-conf`: Heron confidence threshold (default: `0.5`)
- `--enable-opendatalab`: enable OpenDataLab YOLO supplement (default: off)
- `--metadata-file` (default: `metadata.json`)
- `--daemon-mode {off,auto,on}` model lifecycle mode (default: `off`)
- `--daemon-socket` local Unix socket path
  (default: `/tmp/pdf_cropper_modeld_<uid>.sock`)
- `--daemon-idle-seconds` daemon idle timeout before auto-exit (default: `300`)
- `--daemon-start-timeout` daemon startup wait timeout in seconds (default: `12`)
- `--daemon-run-timeout`: timeout for waiting daemon job response;
  `0` means no timeout (default: `0`)

Model download arguments:

- `--model-repo`: OpenDataLab supplementary model repo
  (default: `opendatalab/PDF-Extract-Kit-1.0`)
- `--model-file`: OpenDataLab supplementary model file in snapshot
  (default: `models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt`)
- `--hf-cache-dir`
- `--hf-token`

## 5. Output Layout

```text
output/
└── <pdf_stem>/
    ├── metadata.json
    ├── image/
    ├── table/
    ├── code/
    └── algorithm/
```

Crop file pattern:

```text
p{page}_{type}_{idx}_{x0}_{y0}_{x1}_{y1}[_merged].jpg
```

Each crop item in `metadata.json` includes:

- `content_type`
- `page_index`, `page_number`
- `page_size_pdf`
- `bbox_pdf` (PDF coordinates)
- `bbox_pixels` (pixel coordinates)
- `score`
- `image_path`
- `merged_from` (if merged)

## 6. Tuning

- Too many misses: lower `--conf` (e.g. `0.05~0.10`)
- Too many false positives: raise `--conf` (e.g. `0.15~0.30`)
- Need finer detail: raise `--dpi` (commonly `300`)
- Too slow or OOM: lower `--dpi` / `--imgsz`, or use `--device cpu`
- Exporting `algorithm` requires `--enable-opendatalab`

## 7. License (Repository Code + Models)

**Repository source code uses MIT License.**

Model licenses:

- `docling-project/docling-layout-heron`: **Apache-2.0**
- `opendatalab/PDF-Extract-Kit-1.0`: **AGPL-3.0(非商用协议, 注意, 按需使用)**

## 8. Documentation Versions

- Chinese: `README.md`
- English (current): `README.en.md`

## 9. SDK Usage

```python
from pdf_cropper import CropJobConfig, crop_pdf, crop_pdf_simple

result = crop_pdf_simple(
  input_pdf="paper.pdf",
  output_dir="./output",
  detect_type="table",
)

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

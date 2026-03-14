from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .constants import (
    ALGORITHM_CLASS_ID,
    CAPTION_FAMILY,
    CODE_BODY_CLASS_ID,
    CODE_CAPTION_CLASS_ID,
    DEFAULT_OD_MODEL_RELATIVE_PATH,
    DEFAULT_OD_REPO,
    HERON_ADJACENT_IDS,
    HERON_CODE_ID,
    HERON_PICTURE_ID,
    HERON_TABLE_ID,
    IMAGE_BODY_CLASS_ID,
    IMAGE_CAPTION_CLASS_ID,
    TABLE_BODY_CLASS_ID,
    TABLE_CAPTION_CLASS_ID,
)
from .models import CropRecord


@dataclass
class _RawDet:
    """Page-level in-memory intermediate detection."""

    content_type: str
    pixel_box: list[int]
    score: float
    source_boxes: list[list[int]] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return ivalue


def parse_page_ranges(raw: str, page_count: int) -> list[int]:
    pages: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                raise ValueError(
                    f"invalid page range '{chunk}': start is greater than end"
                )
            pages.update(range(start, end + 1))
        else:
            pages.add(int(chunk))
    invalid = [p for p in pages if p < 1 or p > page_count]
    if invalid:
        raise ValueError(
            f"page selection out of bounds: {sorted(invalid)}; "
            f"valid range is 1..{page_count}"
        )
    return sorted(pages)


def ensure_model(
    model_repo: str,
    model_relative_path: str,
    cache_dir: str | None,
    token: str | None,
) -> str:
    from huggingface_hub import snapshot_download

    model_relative_path = model_relative_path.strip("/")
    snapshot_root = snapshot_download(
        repo_id=model_repo,
        allow_patterns=[model_relative_path, f"{model_relative_path}/*"],
        cache_dir=cache_dir,
        token=token,
    )
    model_path = os.path.join(snapshot_root, model_relative_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model file '{model_relative_path}' was not found inside the "
            "downloaded snapshot"
        )
    return model_path


def ensure_heron_model(
    model_name: str,
    cache_dir: str | None,
    token: str | None,
    device: str,
) -> tuple[Any, Any]:
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    print(f"Loading heron model: {model_name} ...", file=sys.stderr)
    processor = RTDetrImageProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=token,
    )
    model = (
        RTDetrV2ForObjectDetection.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=token,
        )
        .to(device)
        .eval()
    )
    return processor, model


@dataclass
class ModelRuntime:
    """Manage model lifecycle for one process (direct mode or daemon worker)."""

    device: str
    heron_processor: Any = None
    heron_model: Any = None
    od_detector: Any = None
    _heron_key: tuple[str, str | None, str | None, str] | None = None
    _od_key: tuple[str, str, str | None, str | None, str] | None = None

    def ensure_heron(
        self,
        model_name: str,
        cache_dir: str | None,
        token: str | None,
    ) -> tuple[Any, Any]:
        key = (model_name, cache_dir, token, self.device)
        if self._heron_key == key and self.heron_processor is not None:
            return self.heron_processor, self.heron_model
        self.heron_processor, self.heron_model = ensure_heron_model(
            model_name=model_name,
            cache_dir=cache_dir,
            token=token,
            device=self.device,
        )
        self._heron_key = key
        return self.heron_processor, self.heron_model

    def ensure_opendatalab(
        self,
        enabled: bool,
        model_repo: str = DEFAULT_OD_REPO,
        model_file: str = DEFAULT_OD_MODEL_RELATIVE_PATH,
        cache_dir: str | None = None,
        token: str | None = None,
    ) -> Any:
        if not enabled:
            self.od_detector = None
            self._od_key = None
            return None

        key = (model_repo, model_file, cache_dir, token, self.device)
        if self._od_key == key and self.od_detector is not None:
            return self.od_detector

        model_path = ensure_model(
            model_repo=model_repo,
            model_relative_path=model_file,
            cache_dir=cache_dir,
            token=token,
        )
        import torch
        from doclayout_yolo import YOLOv10
        from doclayout_yolo.nn.tasks import YOLOv10DetectionModel

        torch.serialization.add_safe_globals([YOLOv10DetectionModel])
        self.od_detector = YOLOv10(model_path).to(self.device)
        self._od_key = key
        return self.od_detector

    def cleanup(self) -> None:
        self.heron_processor = None
        self.heron_model = None
        self.od_detector = None
        self._heron_key = None
        self._od_key = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def _dedup_contained_raw(group: list[_RawDet]) -> list[_RawDet]:
    to_remove: set[int] = set()
    n = len(group)
    for i in range(n):
        if i in to_remove:
            continue
        ax0, ay0, ax1, ay1 = group[i].pixel_box
        area_a = (ax1 - ax0) * (ay1 - ay0)
        for j in range(n):
            if i == j or j in to_remove:
                continue
            bx0, by0, bx1, by1 = group[j].pixel_box
            area_b = (bx1 - bx0) * (by1 - by0)
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            if area_a >= area_b and inter / area_b >= 0.8:
                to_remove.add(j)
    return [d for i, d in enumerate(group) if i not in to_remove]


def _estimate_dilate_px_raw(group: list[_RawDet]) -> int:
    caption_types = set(CAPTION_FAMILY.keys())
    body_boxes = [d.pixel_box for d in group if d.content_type not in caption_types]
    caption_boxes = [d.pixel_box for d in group if d.content_type in caption_types]
    all_boxes = [d.pixel_box for d in group]

    gaps: list[int] = []
    for i, (x0a, y0a, x1a, y1a) in enumerate(all_boxes):
        for x0b, y0b, x1b, y1b in all_boxes[i + 1 :]:
            h_gap = max(0, max(x0a, x0b) - min(x1a, x1b))
            v_gap = max(0, max(y0a, y0b) - min(y1a, y1b))
            gap = max(h_gap, v_gap)
            if gap > 0:
                gaps.append(gap)
    if not gaps:
        return 4

    ref_boxes = body_boxes if body_boxes else all_boxes
    median_short = median([min(x1 - x0, y1 - y0) for x0, y0, x1, y1 in ref_boxes])
    cap = max(8, int(median_short * 0.10))
    result = max(4, min(min(gaps) + 4, cap))

    if caption_boxes:
        caption_floor = max(min(x1 - x0, y1 - y0) for x0, y0, x1, y1 in caption_boxes)
        result = max(result, caption_floor)

    return result


def _merge_cross_type_once(
    dets: list[_RawDet],
    img_w: int,
    img_h: int,
) -> tuple[list[_RawDet], bool]:
    import cv2
    import numpy as np

    if len(dets) <= 1:
        return dets, False

    dilate_px = _estimate_dilate_px_raw(dets)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for d in dets:
        x0, y0, x1, y1 = d.pixel_box
        mask[y0:y1, x0:x1] = 255

    k = dilate_px * 2 + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel)
    _, labels = cv2.connectedComponents(dilated, connectivity=8)

    comp_groups: dict[int, list[_RawDet]] = defaultdict(list)
    for d in dets:
        cx = (d.pixel_box[0] + d.pixel_box[2]) // 2
        cy = (d.pixel_box[1] + d.pixel_box[3]) // 2
        label = int(labels[cy, cx])
        comp_groups[label].append(d)

    merged_any = False
    out: list[_RawDet] = []
    for _, comp in sorted(comp_groups.items()):
        if len(comp) == 1:
            out.append(comp[0])
            continue

        merged_any = True
        ux0 = min(d.pixel_box[0] for d in comp)
        uy0 = min(d.pixel_box[1] for d in comp)
        ux1 = max(d.pixel_box[2] for d in comp)
        uy1 = max(d.pixel_box[3] for d in comp)

        area_by_type: dict[str, int] = defaultdict(int)
        source_boxes: list[list[int]] = []
        source_types: list[str] = []
        for d in comp:
            x0, y0, x1, y1 = d.pixel_box
            area_by_type[d.content_type] += (x1 - x0) * (y1 - y0)
            if d.source_boxes:
                source_boxes.extend(d.source_boxes)
            else:
                source_boxes.append(d.pixel_box)
            if d.source_types:
                source_types.extend(d.source_types)
            else:
                source_types.append(d.content_type)

        dominant_type = max(sorted(area_by_type.items()), key=lambda item: item[1])[0]
        out.append(
            _RawDet(
                content_type=dominant_type,
                pixel_box=[ux0, uy0, ux1, uy1],
                score=round(max(d.score for d in comp), 4),
                source_boxes=source_boxes,
                source_types=source_types,
            )
        )

    return out, merged_any


def _post_clean_isolated_caption_like(dets: list[_RawDet]) -> list[_RawDet]:
    caption_like = set(CAPTION_FAMILY.keys())
    cleaned: list[_RawDet] = []
    for d in dets:
        src_types = d.source_types or [d.content_type]
        has_body = any(t not in caption_like for t in src_types)
        if has_body:
            cleaned.append(d)
    return cleaned


def _merge_page_dets_by_cc(
    page_dets: list[_RawDet],
    img_w: int,
    img_h: int,
) -> list[_RawDet]:
    import cv2
    import numpy as np

    by_family: dict[str, list[_RawDet]] = defaultdict(list)
    for d in page_dets:
        family = CAPTION_FAMILY.get(d.content_type, d.content_type)
        by_family[family].append(d)

    result: list[_RawDet] = []
    for body_type, group in by_family.items():
        group = _dedup_contained_raw(group)
        if not group:
            continue
        if len(group) == 1:
            d = group[0]
            result.append(
                _RawDet(
                    content_type=body_type,
                    pixel_box=d.pixel_box,
                    score=d.score,
                    source_boxes=d.source_boxes,
                    source_types=d.source_types,
                )
            )
            continue

        dilate_px = _estimate_dilate_px_raw(group)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for d in group:
            x0, y0, x1, y1 = d.pixel_box
            mask[y0:y1, x0:x1] = 255

        k = dilate_px * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        dilated = cv2.dilate(mask, kernel)
        _, labels = cv2.connectedComponents(dilated, connectivity=8)

        comp_groups: dict[int, list[_RawDet]] = defaultdict(list)
        for d in group:
            cx = (d.pixel_box[0] + d.pixel_box[2]) // 2
            cy = (d.pixel_box[1] + d.pixel_box[3]) // 2
            label = int(labels[cy, cx])
            comp_groups[label].append(d)

        for _, comp in sorted(comp_groups.items()):
            if len(comp) == 1:
                d = comp[0]
                result.append(
                    _RawDet(
                        content_type=body_type,
                        pixel_box=d.pixel_box,
                        score=d.score,
                        source_boxes=d.source_boxes,
                        source_types=d.source_types,
                    )
                )
                continue
            ux0 = min(d.pixel_box[0] for d in comp)
            uy0 = min(d.pixel_box[1] for d in comp)
            ux1 = max(d.pixel_box[2] for d in comp)
            uy1 = max(d.pixel_box[3] for d in comp)
            result.append(
                _RawDet(
                    content_type=body_type,
                    pixel_box=[ux0, uy0, ux1, uy1],
                    score=round(max(d.score for d in comp), 4),
                    source_boxes=[d.pixel_box for d in comp],
                    source_types=[
                        t for d in comp for t in (d.source_types or [d.content_type])
                    ],
                )
            )

    if len(result) > 1:
        result = _dedup_contained_raw(result)

    for _ in range(8):
        if len(result) <= 1:
            break
        result, merged_any = _merge_cross_type_once(result, img_w, img_h)
        if len(result) > 1:
            result = _dedup_contained_raw(result)
        if not merged_any:
            break

    return _post_clean_isolated_caption_like(result)


def safe_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x0i, y0i, x1i, y1i = int(x0), int(y0), int(x1), int(y1)
    x0i = max(0, min(x0i, width))
    x1i = max(0, min(x1i, width))
    y0i = max(0, min(y0i, height))
    y1i = max(0, min(y1i, height))
    if x1i <= x0i or y1i <= y0i:
        return None
    return x0i, y0i, x1i, y1i


def _box_near_any(box: list[int], ref_boxes: list[list[int]], gap_px: int) -> bool:
    bx0, by0, bx1, by1 = box
    for rx0, ry0, rx1, ry1 in ref_boxes:
        h_gap = max(0, max(bx0, rx0) - min(bx1, rx1))
        v_gap = max(0, max(by0, ry0) - min(by1, ry1))
        if max(h_gap, v_gap) <= gap_px:
            return True
    return False


def _heron_page_layout_dets(
    page_img,
    heron_processor,
    heron_model,
    img_w: int,
    img_h: int,
    threshold: float,
    adj_gap_px: int,
    detect_type: str,
    include_visual_types: bool = True,
) -> list[_RawDet]:
    import torch

    inputs = heron_processor(images=[page_img], return_tensors="pt")
    dev = next(heron_model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = heron_model(**inputs)
    detections = heron_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[img_h, img_w]]),
        threshold=threshold,
    )[0]

    allow_ids = {HERON_PICTURE_ID, HERON_TABLE_ID, HERON_CODE_ID}.union(
        HERON_ADJACENT_IDS
    )
    raw: list[tuple[int, list[int], float]] = []
    for score, label_id, box in zip(
        detections["scores"],
        detections["labels"],
        detections["boxes"],
    ):
        cls_id = int(label_id.item())
        if cls_id not in allow_ids:
            continue
        raw_box = box.tolist()
        if len(raw_box) != 4:
            continue
        x0 = float(raw_box[0])
        y0 = float(raw_box[1])
        x1 = float(raw_box[2])
        y1 = float(raw_box[3])
        pixel_box = safe_box(x0, y0, x1, y1, img_w, img_h)
        if pixel_box is None:
            continue
        raw.append((cls_id, list(pixel_box), round(float(score.item()), 4)))

    code_boxes = [box for cls_id, box, _ in raw if cls_id == HERON_CODE_ID]
    dets: list[_RawDet] = []
    for cls_id, box, score in raw:
        if include_visual_types and cls_id == HERON_PICTURE_ID:
            if detect_type in ("both", "image"):
                dets.append(
                    _RawDet(
                        content_type="image",
                        pixel_box=box,
                        score=score,
                        source_types=["image"],
                    )
                )
        elif include_visual_types and cls_id == HERON_TABLE_ID:
            if detect_type in ("both", "table"):
                dets.append(
                    _RawDet(
                        content_type="table",
                        pixel_box=box,
                        score=score,
                        source_types=["table"],
                    )
                )
        elif cls_id == HERON_CODE_ID and detect_type in ("both", "code"):
            dets.append(
                _RawDet(
                    content_type="code",
                    pixel_box=box,
                    score=score,
                    source_types=["code"],
                )
            )
        elif detect_type in ("both", "code") and _box_near_any(
            box,
            code_boxes,
            gap_px=adj_gap_px,
        ):
            dets.append(
                _RawDet(
                    content_type="code_caption",
                    pixel_box=box,
                    score=score,
                    source_types=["code_caption"],
                )
            )

    return dets


def collect_crops(
    pdf_path: Path,
    output_dir: Path,
    pages_to_process: Iterable[int],
    detect_type: str,
    dpi: int,
    do_merge: bool = True,
    heron_processor=None,
    heron_model=None,
    heron_conf: float = 0.5,
    od_detector=None,
    od_imgsz: int = 1280,
    od_conf: float = 0.10,
    od_iou: float = 0.45,
) -> list[CropRecord]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    selected_class_ids = {
        IMAGE_BODY_CLASS_ID: "image",
        IMAGE_CAPTION_CLASS_ID: "image_caption",
        TABLE_BODY_CLASS_ID: "table",
        TABLE_CAPTION_CLASS_ID: "table_caption",
        CODE_BODY_CLASS_ID: "code",
        CODE_CAPTION_CLASS_ID: "code_caption",
        ALGORITHM_CLASS_ID: "algorithm",
    }
    if detect_type == "image":
        selected_class_ids = {
            IMAGE_BODY_CLASS_ID: "image",
            IMAGE_CAPTION_CLASS_ID: "image_caption",
        }
    elif detect_type == "table":
        selected_class_ids = {
            TABLE_BODY_CLASS_ID: "table",
            TABLE_CAPTION_CLASS_ID: "table_caption",
        }
    elif detect_type == "code":
        selected_class_ids = {
            CODE_BODY_CLASS_ID: "code",
            CODE_CAPTION_CLASS_ID: "code_caption",
        }
    elif detect_type == "algorithm":
        selected_class_ids = {ALGORITHM_CLASS_ID: "algorithm"}

    records: list[CropRecord] = []
    try:
        for page_number in pages_to_process:
            page_index = page_number - 1
            page = pdf[page_index]
            page_width, page_height = page.get_size()
            page_img = page.render(scale=scale).to_pil()
            img_w, img_h = page_img.size

            page_dets: list[_RawDet] = []
            if heron_processor is not None and heron_model is not None:
                adj_gap = max(30, min(80, img_h // 20))
                heron_visual = od_detector is None
                page_dets.extend(
                    _heron_page_layout_dets(
                        page_img,
                        heron_processor,
                        heron_model,
                        img_w,
                        img_h,
                        threshold=heron_conf,
                        adj_gap_px=adj_gap,
                        detect_type=detect_type,
                        include_visual_types=heron_visual,
                    )
                )

            if od_detector is not None:
                predictions = od_detector.predict(
                    page_img,
                    imgsz=od_imgsz,
                    conf=od_conf,
                    iou=od_iou,
                    verbose=False,
                )[0]
                if predictions.boxes is not None:
                    for xyxy, score, cls_id in zip(
                        predictions.boxes.xyxy.cpu(),
                        predictions.boxes.conf.cpu(),
                        predictions.boxes.cls.cpu(),
                    ):
                        category_id = int(cls_id.item())
                        if category_id not in selected_class_ids:
                            continue
                        raw_x0, raw_y0, raw_x1, raw_y1 = xyxy.tolist()
                        pixel_box = safe_box(
                            raw_x0,
                            raw_y0,
                            raw_x1,
                            raw_y1,
                            img_w,
                            img_h,
                        )
                        if pixel_box is None:
                            continue
                        label = selected_class_ids[category_id]
                        page_dets.append(
                            _RawDet(
                                content_type=label,
                                pixel_box=list(pixel_box),
                                score=round(float(score.item()), 4),
                                source_types=[label],
                            )
                        )

            if not page_dets:
                continue

            if do_merge and len(page_dets) > 1:
                page_dets = _merge_page_dets_by_cc(page_dets, img_w, img_h)
            else:
                page_dets = _post_clean_isolated_caption_like(page_dets)

            type_counter: dict[str, int] = defaultdict(int)
            for det in page_dets:
                x0, y0, x1, y1 = det.pixel_box
                content_type = det.content_type
                crop_dir = output_dir / content_type
                crop_dir.mkdir(parents=True, exist_ok=True)
                idx = type_counter[content_type]
                suffix = "_merged" if det.source_boxes else ""
                file_name = (
                    f"p{page_number:04d}_{content_type}_{idx:03d}"
                    f"_{x0}_{y0}_{x1}_{y1}{suffix}.jpg"
                )
                crop_path = crop_dir / file_name
                page_img.crop((x0, y0, x1, y1)).save(
                    crop_path,
                    format="JPEG",
                    quality=95,
                )
                type_counter[content_type] += 1

                records.append(
                    CropRecord(
                        content_type=content_type,
                        page_index=page_index,
                        page_number=page_number,
                        page_size_pdf=[
                            round(float(page_width), 2),
                            round(float(page_height), 2),
                        ],
                        bbox_pdf=[
                            round(x0 / scale, 2),
                            round(y0 / scale, 2),
                            round(x1 / scale, 2),
                            round(y1 / scale, 2),
                        ],
                        bbox_pixels=det.pixel_box,
                        score=det.score,
                        image_path=str(crop_path.resolve()),
                        merged_from=det.source_boxes,
                    )
                )
    finally:
        pdf.close()

    return records


def resolve_device(device_arg: str | None) -> str:
    if device_arg is not None:
        print(f"Using device: {device_arg}", file=sys.stderr)
        return device_arg

    import torch

    if torch.backends.mps.is_available():
        resolved = "mps"
    elif torch.cuda.is_available():
        resolved = "cuda"
    else:
        resolved = "cpu"
    print(f"Using device: {resolved} (auto-detected)", file=sys.stderr)
    return resolved


def run_single_job(payload: dict[str, Any], runtime: ModelRuntime) -> dict[str, Any]:
    input_pdf = Path(payload["input_pdf"]).expanduser().resolve()
    storage_root = Path(payload["storage_root"]).expanduser().resolve()
    output_dir = storage_root / input_pdf.stem

    if not input_pdf.exists() or not input_pdf.is_file():
        raise FileNotFoundError(
            f"input PDF does not exist or is not a file: {input_pdf}"
        )
    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"input file must have .pdf extension: {input_pdf}")
    output_dir.mkdir(parents=True, exist_ok=True)

    conf = float(payload["conf"])
    iou = float(payload["iou"])
    if not (0.0 <= conf <= 1.0):
        raise ValueError("--conf must be between 0 and 1")
    if not (0.0 <= iou <= 1.0):
        raise ValueError("--iou must be between 0 and 1")

    import pypdfium2 as pdfium

    pdf_probe = pdfium.PdfDocument(str(input_pdf))
    try:
        page_count = len(pdf_probe)
    finally:
        pdf_probe.close()

    pages_raw = payload["pages"]
    if pages_raw.lower() == "all":
        pages_to_process = list(range(1, page_count + 1))
    else:
        pages_to_process = parse_page_ranges(pages_raw, page_count)

    heron_processor, heron_model = runtime.ensure_heron(
        model_name=payload["heron_model"],
        cache_dir=payload["hf_cache_dir"],
        token=payload["hf_token"],
    )

    od_detector = runtime.ensure_opendatalab(
        enabled=bool(payload["enable_opendatalab"]),
        model_repo=payload["model_repo"],
        model_file=payload["model_file"],
        cache_dir=payload["hf_cache_dir"],
        token=payload["hf_token"],
    )
    if not payload["enable_opendatalab"] and payload["type"] == "algorithm":
        print(
            "Warning: --type algorithm needs OpenDataLab; " "use --enable-opendatalab.",
            file=sys.stderr,
        )

    print(f"Processing {len(pages_to_process)} page(s)...", file=sys.stderr)
    records = collect_crops(
        pdf_path=input_pdf,
        output_dir=output_dir,
        pages_to_process=pages_to_process,
        detect_type=payload["type"],
        dpi=int(payload["dpi"]),
        do_merge=not bool(payload["no_merge"]),
        heron_processor=heron_processor,
        heron_model=heron_model,
        heron_conf=float(payload["heron_conf"]),
        od_detector=od_detector,
        od_imgsz=int(payload["imgsz"]),
        od_conf=conf,
        od_iou=iou,
    )

    metadata_path = output_dir / payload["metadata_file"]
    metadata = {
        "input_pdf": str(input_pdf),
        "storage_root": str(storage_root),
        "output_path": str(output_dir),
        "page_count": page_count,
        "pages_processed": pages_to_process,
        "detect_type": payload["type"],
        "dpi": payload["dpi"],
        "model": {
            "primary": {"heron": payload["heron_model"]},
            "opendatalab": {
                "enabled": payload["enable_opendatalab"],
                "repo": payload["model_repo"],
                "file": payload["model_file"],
            },
        },
        "crops": [asdict(record) for record in records],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = {
        "metadata_file": str(metadata_path),
        "output_path": str(output_dir),
        "crop_count": len(records),
        "path": "direct",
    }
    print(f"Done. Exported {len(records)} crop(s).", file=sys.stderr)
    return result

# -*- coding: utf-8 -*-
"""
文档 OCR 适配器：版面 → 正文 det+rec / 表格 TableRecognitionPipelineV2。

模型目录见 ``ocr.load_ocr``。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document.ocr.labels import (
    DIRECT_REC_LABELS,
    FORMULA_LABELS,
    SKIP_RECOGNITION_LABELS,
    TABLE_LABEL,
)
from document.ocr.load_ocr import (
    DET_MODEL_NAME,
    FORMULA_MODEL_NAME,
    LAYOUT_MODEL_NAME,
    PIPELINE_VERSION,
    REC_MODEL_NAME,
    TABLE_CELL_WIRED_MODEL_NAME,
    TABLE_CELL_WIRELESS_MODEL_NAME,
    TABLE_CLS_MODEL_NAME,
    TABLE_MODEL_DIRS,
    TABLE_STRUCT_MODEL_NAME,
    ensure_table_deps,
    validate_model_dir,
)


@dataclass
class PipelineConfig:
    model_root: Path
    device: str
    threshold: float
    fast: bool
    rec_batch_size: int
    crop_threads: int
    enable_mkldnn: bool
    table_e2e: bool = False
    preprocess_mode: str = "auto"
    enable_formula: bool = True
    formula_model_name: str = FORMULA_MODEL_NAME
    max_attempts: int = 3
    rec_score_threshold: float = 0.85
    layout_score_threshold: float = 0.5
    low_confidence_page_ratio: float = 0.3
    pipeline_version: str = PIPELINE_VERSION

    def summary(self) -> dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "device": self.device,
            "threshold": self.threshold,
            "fast": self.fast,
            "preprocess_mode": self.preprocess_mode,
            "enable_formula": self.enable_formula,
            "formula_model_name": self.formula_model_name,
            "table_e2e": self.table_e2e,
            "max_attempts": self.max_attempts,
            "rec_score_threshold": self.rec_score_threshold,
            "layout_score_threshold": self.layout_score_threshold,
            "low_confidence_page_ratio": self.low_confidence_page_ratio,
        }


@dataclass
class _TextModels:
    layout: Any
    det: Any
    rec: Any


class OcrPipelineAdapter:
    """组合正文 OCR 与表格识别流水线。"""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._text: _TextModels | None = None
        self._table: Any | None = None
        self._formula: Any | None = None

    @property
    def text_models(self) -> _TextModels:
        if self._text is None:
            self._text = _create_text_models(self.cfg)
        return self._text

    def reset_text_models(self) -> None:
        self._text = None

    def _ensure_table_pipeline(self) -> Any:
        if self._table is None:
            ensure_table_deps()
            for model_dir in TABLE_MODEL_DIRS:
                missing = validate_model_dir(model_dir)
                if missing:
                    raise FileNotFoundError(
                        f"表格模型不完整 ({model_dir})，缺少: {', '.join(missing)}"
                    )
            print("  加载表格识别流水线…")
            self._table = _create_table_pipeline(self.cfg)
        return self._table

    def _ensure_formula_model(self) -> Any:
        if self._formula is None:
            FormulaRecognition = _import_formula_recognition()
            root = self.cfg.model_root
            model_dir = root / self.cfg.formula_model_name
            missing = validate_model_dir(model_dir)
            if missing:
                raise FileNotFoundError(
                    f"公式模型不完整 ({model_dir})，缺少: {', '.join(missing)}"
                )
            print("  加载公式识别模型…")
            self._formula = FormulaRecognition(
                model_name=self.cfg.formula_model_name,
                model_dir=str(model_dir),
                **_mkldnn_kw(self.cfg),
            )
        return self._formula

    def process_page(
        self,
        *,
        image_path: Path,
        page_index: int,
        layout_out: Path | None,
        crops_dir: Path | None,
        scratch_dir: Path,
    ) -> dict[str, Any]:
        return process_page(
            image_path=image_path,
            page_index=page_index,
            adapter=self,
            cfg=self.cfg,
            layout_out=layout_out,
            crops_dir=crops_dir,
            scratch_dir=scratch_dir,
        )


def _mkldnn_kw(cfg: PipelineConfig) -> dict[str, Any]:
    kw: dict[str, Any] = {"device": cfg.device}
    if cfg.enable_mkldnn and cfg.device.startswith("cpu"):
        kw["enable_mkldnn"] = True
    return kw


def _import_paddleocr():
    from paddleocr import LayoutDetection, TableRecognitionPipelineV2, TextDetection, TextRecognition

    return LayoutDetection, TextDetection, TextRecognition, TableRecognitionPipelineV2


def _import_formula_recognition():
    from paddleocr import FormulaRecognition

    return FormulaRecognition


def _is_det_inference_error(exc: BaseException) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "onednn" in msg or "convertpirattribute" in msg
    return False


def _create_text_models(cfg: PipelineConfig) -> _TextModels:
    LayoutDetection, TextDetection, TextRecognition, _ = _import_paddleocr()
    kw = _mkldnn_kw(cfg)
    root = cfg.model_root
    return _TextModels(
        layout=LayoutDetection(
            model_name=LAYOUT_MODEL_NAME,
            model_dir=str(root / LAYOUT_MODEL_NAME),
            **kw,
        ),
        det=TextDetection(
            model_name=DET_MODEL_NAME,
            model_dir=str(root / DET_MODEL_NAME),
            **kw,
        ),
        rec=TextRecognition(
            model_name=REC_MODEL_NAME,
            model_dir=str(root / REC_MODEL_NAME),
            **kw,
        ),
    )


def _create_table_pipeline(cfg: PipelineConfig) -> Any:
    _, _, _, TableRecognitionPipelineV2 = _import_paddleocr()
    root = cfg.model_root
    struct = root / TABLE_STRUCT_MODEL_NAME
    kw = _mkldnn_kw(cfg)
    return TableRecognitionPipelineV2(
        table_classification_model_name=TABLE_CLS_MODEL_NAME,
        table_classification_model_dir=str(root / TABLE_CLS_MODEL_NAME),
        wired_table_structure_recognition_model_name=TABLE_STRUCT_MODEL_NAME,
        wired_table_structure_recognition_model_dir=str(struct),
        wireless_table_structure_recognition_model_name=TABLE_STRUCT_MODEL_NAME,
        wireless_table_structure_recognition_model_dir=str(struct),
        wired_table_cells_detection_model_name=TABLE_CELL_WIRED_MODEL_NAME,
        wired_table_cells_detection_model_dir=str(root / TABLE_CELL_WIRED_MODEL_NAME),
        wireless_table_cells_detection_model_name=TABLE_CELL_WIRELESS_MODEL_NAME,
        wireless_table_cells_detection_model_dir=str(root / TABLE_CELL_WIRELESS_MODEL_NAME),
        text_detection_model_name=DET_MODEL_NAME,
        text_detection_model_dir=str(root / DET_MODEL_NAME),
        text_recognition_model_name=REC_MODEL_NAME,
        text_recognition_model_dir=str(root / REC_MODEL_NAME),
        text_recognition_batch_size=cfg.rec_batch_size,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        **kw,
    )


def _payload_from_result(res: Any) -> dict[str, Any]:
    payload: Any = res
    if hasattr(res, "json"):
        payload = res.json
    elif hasattr(res, "to_dict"):
        payload = res.to_dict()
    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]
    return payload if isinstance(payload, dict) else {}


def _boxes_from_layout_result(res: Any) -> list[dict[str, Any]]:
    boxes = _payload_from_result(res).get("boxes", [])
    return list(boxes) if boxes else []


def _dt_polys_from_det_result(res: Any) -> list[Any]:
    polys = _payload_from_result(res).get("dt_polys")
    return list(polys) if polys is not None else []


def _sort_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(box: dict[str, Any]) -> tuple:
        order = box.get("order")
        coord = box.get("coordinate") or [0, 0, 0, 0]
        return (1, coord[1], coord[0]) if order is None else (0, order)

    return sorted(boxes, key=sort_key)


def _poly_to_bbox(poly: Any) -> tuple[int, int, int, int]:
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _crop_from_image(im: Any, box: dict[str, Any]) -> Any | None:
    coord = box.get("coordinate")
    if not coord or len(coord) < 4:
        return None
    xmin, ymin, xmax, ymax = (int(round(c)) for c in coord[:4])
    if xmax <= xmin or ymax <= ymin:
        return None
    w, h = im.size
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(xmin + 1, min(xmax, w))
    ymax = max(ymin + 1, min(ymax, h))
    crop = im.crop((xmin, ymin, xmax, ymax))
    if crop.width < 2 or crop.height < 2:
        return None
    return crop


def _save_pil(path: Path, im: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def _recognize_paths_batch(
    rec_model: Any,
    paths: list[Path],
    *,
    batch_size: int,
) -> list[tuple[str, float]]:
    if not paths:
        return []
    results: list[tuple[str, float]] = []
    for start in range(0, len(paths), batch_size):
        chunk = [str(p) for p in paths[start : start + batch_size]]
        outputs = list(rec_model.predict(chunk, batch_size=len(chunk)))
        for out in outputs:
            payload = _payload_from_result(out)
            text = str(payload.get("rec_text") or payload.get("text") or "").strip()
            score = float(payload.get("rec_score") or payload.get("score") or 0.0)
            results.append((text, score))
    while len(results) < len(paths):
        results.append(("", 0.0))
    return results[: len(paths)]


def _recognize_pil_batch(
    rec_model: Any,
    images: list[Any],
    *,
    batch_size: int,
    scratch_dir: Path,
    prefix: str,
) -> list[tuple[str, float]]:
    paths: list[Path] = []
    for i, im in enumerate(images):
        p = scratch_dir / f"{prefix}_{i:04d}.png"
        _save_pil(p, im)
        paths.append(p)
    return _recognize_paths_batch(rec_model, paths, batch_size=batch_size)


def _heuristic_line_crops(region_im: Any, *, target_line_h: int = 36) -> list[Any]:
    """det 不可用时的水平条带切分，避免整图直接 rec。"""
    w, h = region_im.size
    if h <= target_line_h * 2:
        return [region_im]
    line_h = max(20, min(target_line_h, h // max(1, h // target_line_h)))
    lines: list[Any] = []
    y = 0
    while y < h:
        y2 = min(y + line_h, h)
        if y2 - y >= 10:
            lines.append(region_im.crop((0, y, w, y2)))
        y = y2
    return lines if lines else [region_im]


def _recognize_with_line_images(
    models: _TextModels,
    line_images: list[Any],
    *,
    cfg: PipelineConfig,
    scratch_dir: Path,
    region_tag: str,
    crops_dir: Path | None,
) -> tuple[str, float]:
    if not line_images:
        return "", 0.0
    if not cfg.fast and crops_dir:
        line_paths: list[Path] = []
        for li, lim in enumerate(line_images):
            lp = crops_dir / f"{region_tag}_line_{li:03d}.png"
            _save_pil(lp, lim)
            line_paths.append(lp)
        batch_results = _recognize_paths_batch(
            models.rec, line_paths, batch_size=cfg.rec_batch_size
        )
    else:
        batch_results = _recognize_pil_batch(
            models.rec,
            line_images,
            batch_size=cfg.rec_batch_size,
            scratch_dir=scratch_dir,
            prefix=f"{region_tag}_line",
        )
    parts = [t for t, _ in batch_results if t]
    if not parts:
        return "", 0.0
    scores = [s for t, s in batch_results if t]
    return "\n".join(parts), sum(scores) / len(scores)


def _line_crops_from_region(region_im: Any, polys: list[Any]) -> list[Any]:
    line_boxes = sorted((_poly_to_bbox(p) for p in polys), key=lambda b: (b[1], b[0]))
    w, h = region_im.size
    lines: list[Any] = []
    for xmin, ymin, xmax, ymax in line_boxes:
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(xmin + 1, min(xmax, w))
        ymax = max(ymin + 1, min(ymax, h))
        if xmax - xmin < 4 or ymax - ymin < 4:
            continue
        lines.append(region_im.crop((xmin, ymin, xmax, ymax)))
    return lines


def _recognize_region(
    models: _TextModels,
    region_im: Any,
    label: str,
    *,
    cfg: PipelineConfig,
    scratch_dir: Path,
    region_tag: str,
    crops_dir: Path | None,
) -> tuple[str, float]:
    if label in DIRECT_REC_LABELS:
        if cfg.fast:
            p = scratch_dir / f"{region_tag}_direct.png"
            _save_pil(p, region_im)
            return _recognize_paths_batch(models.rec, [p], batch_size=1)[0]
        p = crops_dir / f"{region_tag}.png" if crops_dir else scratch_dir / f"{region_tag}.png"
        _save_pil(p, region_im)
        return _recognize_paths_batch(models.rec, [p], batch_size=1)[0]

    if cfg.fast:
        region_path = scratch_dir / f"{region_tag}_region.png"
    else:
        region_path = (crops_dir or scratch_dir) / f"{region_tag}.png"
    _save_pil(region_path, region_im)

    try:
        det_outputs = list(models.det.predict(str(region_path), batch_size=1))
    except Exception as exc:
        if not _is_det_inference_error(exc):
            raise
        print(f"  区域 det 推理异常 ({region_tag})，使用行切分退化识别")
        line_images = _heuristic_line_crops(region_im)
        if len(line_images) > 1:
            return _recognize_with_line_images(
                models,
                line_images,
                cfg=cfg,
                scratch_dir=scratch_dir,
                region_tag=region_tag,
                crops_dir=crops_dir,
            )
        return _recognize_paths_batch(models.rec, [region_path], batch_size=1)[0]

    if not det_outputs:
        line_images = _heuristic_line_crops(region_im)
        if len(line_images) > 1:
            return _recognize_with_line_images(
                models, line_images, cfg=cfg, scratch_dir=scratch_dir,
                region_tag=region_tag, crops_dir=crops_dir,
            )
        return _recognize_paths_batch(models.rec, [region_path], batch_size=1)[0]

    polys = _dt_polys_from_det_result(det_outputs[0])
    if not polys:
        line_images = _heuristic_line_crops(region_im)
        if len(line_images) > 1:
            return _recognize_with_line_images(
                models, line_images, cfg=cfg, scratch_dir=scratch_dir,
                region_tag=region_tag, crops_dir=crops_dir,
            )
        return _recognize_paths_batch(models.rec, [region_path], batch_size=1)[0]

    line_images = _line_crops_from_region(region_im, polys)
    if not line_images:
        return "", 0.0

    return _recognize_with_line_images(
        models,
        line_images,
        cfg=cfg,
        scratch_dir=scratch_dir,
        region_tag=region_tag,
        crops_dir=crops_dir,
    )


def _extract_table_entries(table_res: Any) -> list[dict[str, Any]]:
    payload = _payload_from_result(table_res)
    entries: list[dict[str, Any]] = []

    if "table_res_list" in payload:
        for item in payload["table_res_list"] or []:
            if isinstance(item, dict):
                entries.append(item)
            else:
                entries.append(_payload_from_result(item))
        return entries

    if "pred_html" in payload:
        return [payload]

    return entries


def _recognize_formula(
    adapter: OcrPipelineAdapter,
    crop_path: Path,
    *,
    cfg: PipelineConfig,
) -> tuple[str, float]:
    if not cfg.enable_formula:
        return "", 0.0
    try:
        model = adapter._ensure_formula_model()
    except FileNotFoundError as exc:
        print(f"  公式识别跳过: {exc}")
        return "", 0.0

    try:
        outputs = list(model.predict(str(crop_path), batch_size=1))
    except ModuleNotFoundError as exc:
        print(f"  公式识别跳过（缺少依赖）: {exc}")
        return "", 0.0
    except ImportError as exc:
        print(f"  公式识别跳过（缺少依赖）: {exc}")
        return "", 0.0
    if not outputs:
        return "", 0.0
    payload = _payload_from_result(outputs[0])
    latex = str(payload.get("rec_formula") or payload.get("formula") or "").strip()
    score = float(payload.get("rec_score") or payload.get("score") or 0.0)
    if latex and score <= 0:
        score = 1.0
    return latex, score


def _recognize_table(
    adapter: OcrPipelineAdapter,
    crop_path: Path,
    *,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    pipe = adapter._ensure_table_pipeline()
    predict_kw: dict[str, Any] = dict(
        use_layout_detection=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_table_orientation_classify=False,
    )
    if cfg.table_e2e:
        predict_kw["use_e2e_wired_table_rec_model"] = True
        predict_kw["use_e2e_wireless_table_rec_model"] = True

    outputs = list(pipe.predict(str(crop_path), **predict_kw))
    if not outputs:
        return {"pred_html": "", "table_type": None}

    entries = _extract_table_entries(outputs[0])
    if not entries:
        return {"pred_html": "", "table_type": None}

    best = entries[0]
    return {
        "pred_html": best.get("pred_html") or "",
        "table_type": best.get("table_type") or best.get("label"),
        "cell_box_list": best.get("cell_box_list"),
    }


def _prepare_region(
    page_im: Any,
    bi: int,
    box: dict[str, Any],
) -> tuple[int, dict[str, Any], Any | None]:
    return bi, box, _crop_from_image(page_im, box)


def _layout_cache_path(scratch_dir: Path) -> Path:
    return scratch_dir / "layout_boxes.json"


def _save_layout_cache(scratch_dir: Path, *, image_path: Path, boxes: list[dict[str, Any]]) -> None:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    payload = {"image": str(image_path.resolve()), "boxes": boxes}
    _layout_cache_path(scratch_dir).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_layout_cache(scratch_dir: Path, work_image: Path) -> list[dict[str, Any]] | None:
    path = _layout_cache_path(scratch_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("image") != str(work_image.resolve()):
        return None
    boxes = payload.get("boxes")
    return list(boxes) if boxes else None


def _recognize_one_region(
    adapter: OcrPipelineAdapter,
    models: _TextModels,
    *,
    bi: int,
    box: dict[str, Any],
    region_im: Any,
    page_index: int,
    cfg: PipelineConfig,
    scratch_dir: Path,
    crops_dir: Path | None,
) -> dict[str, Any]:
    label = box.get("label", "")
    region: dict[str, Any] = {
        "box_index": bi,
        "order": box.get("order"),
        "label": label,
        "cls_id": box.get("cls_id"),
        "layout_score": box.get("score"),
        "coordinate": box.get("coordinate"),
        "text": "",
        "rec_score": None,
    }
    region_tag = f"p{page_index:04d}_b{bi:03d}"

    if label == TABLE_LABEL:
        crop_path = scratch_dir / f"{region_tag}_table.png"
        _save_pil(crop_path, region_im)
        table_result = _recognize_table(adapter, crop_path, cfg=cfg)
        region["table_result"] = table_result
        region["text"] = table_result.get("pred_html") or ""
        return region

    if label in FORMULA_LABELS:
        crop_path = scratch_dir / f"{region_tag}_formula.png"
        _save_pil(crop_path, region_im)
        latex, rec_score = _recognize_formula(adapter, crop_path, cfg=cfg)
        region["text"] = latex
        region["rec_score"] = rec_score
        region["formula_result"] = {"latex": latex, "rec_score": rec_score}
        return region

    text, rec_score = _recognize_region(
        models,
        region_im,
        label,
        cfg=cfg,
        scratch_dir=scratch_dir,
        region_tag=region_tag,
        crops_dir=crops_dir,
    )
    region["text"] = text
    region["rec_score"] = rec_score
    if not cfg.fast and crops_dir:
        region["crop"] = str(crops_dir / f"{region_tag}.png")
    return region


def retry_failed_regions(
    *,
    adapter: OcrPipelineAdapter,
    cfg: PipelineConfig,
    work_image: Path,
    page_index: int,
    merge_page: dict[str, Any],
    retry_indices: set[int],
    scratch_dir: Path,
    layout_scratch_dir: Path,
    crops_dir: Path | None,
) -> dict[str, Any]:
    """仅重试 QC 失败的区域，复用已缓存的版面框。"""
    from PIL import Image

    boxes = _load_layout_cache(layout_scratch_dir, work_image)
    if not boxes:
        raise FileNotFoundError("无可用 layout 缓存，无法区域级重试")

    models = adapter.text_models
    page_im = Image.open(work_image)
    page_im.load()
    old_by_bi = {r["box_index"]: r for r in merge_page.get("regions") or []}
    regions: list[dict[str, Any]] = []

    print(f"  区域级重试: {len(retry_indices)} 个块")
    for bi, box in enumerate(boxes):
        label = box.get("label", "")
        base = dict(old_by_bi.get(bi, {}))
        base.update(
            {
                "box_index": bi,
                "order": box.get("order"),
                "label": label,
                "cls_id": box.get("cls_id"),
                "layout_score": box.get("score"),
                "coordinate": box.get("coordinate"),
            }
        )
        if bi not in retry_indices or label in SKIP_RECOGNITION_LABELS:
            regions.append(base)
            continue

        region_im = _crop_from_image(page_im, box)
        if region_im is None:
            regions.append(base)
            continue

        region = _recognize_one_region(
            adapter,
            models,
            bi=bi,
            box=box,
            region_im=region_im,
            page_index=page_index,
            cfg=cfg,
            scratch_dir=scratch_dir,
            crops_dir=crops_dir,
        )
        order_str = region["order"] if region["order"] is not None else "-"
        preview = (region.get("text") or "")[:40]
        if len(preview) >= 40:
            preview += "…"
        score = region.get("rec_score")
        score_s = f"{score:.3f}" if score is not None else "-"
        print(f"  [{order_str}] {label} (retry): {preview!r} ({score_s})")
        regions.append(region)

    page_im.close()
    out = dict(merge_page)
    out["regions"] = regions
    out["image"] = str(work_image.resolve())
    return out


def process_page(
    *,
    image_path: Path,
    page_index: int,
    adapter: OcrPipelineAdapter,
    cfg: PipelineConfig,
    layout_out: Path | None,
    crops_dir: Path | None,
    scratch_dir: Path,
) -> dict[str, Any]:
    from PIL import Image

    models = adapter.text_models
    t0 = time.perf_counter()
    print(f"\n版面分析: {image_path.name}")

    layout_results = list(
        models.layout.predict(
            str(image_path),
            batch_size=1,
            layout_nms=True,
            threshold=cfg.threshold,
        )
    )
    if not layout_results:
        return {"page_index": page_index, "image": str(image_path), "regions": []}

    layout_res = layout_results[0]
    layout_json: str | None = None
    if not cfg.fast and layout_out:
        layout_res.save_to_img(save_path=str(layout_out))
        layout_json_path = layout_out / f"layout_page_{page_index:04d}.json"
        layout_res.save_to_json(save_path=str(layout_json_path))
        layout_json = str(layout_json_path.resolve())

    boxes = _sort_boxes(_boxes_from_layout_result(layout_res))
    page_im = Image.open(image_path)
    page_im.load()

    prepared: list[tuple[int, dict[str, Any], Any]] = []
    if cfg.crop_threads > 1 and len(boxes) > 1:
        with ThreadPoolExecutor(max_workers=cfg.crop_threads) as pool:
            futs = [
                pool.submit(_prepare_region, page_im, bi, box)
                for bi, box in enumerate(boxes)
            ]
            for fut in as_completed(futs):
                bi, box, crop = fut.result()
                if crop is not None:
                    prepared.append((bi, box, crop))
        prepared.sort(key=lambda x: x[0])
    else:
        for bi, box in enumerate(boxes):
            crop = _crop_from_image(page_im, box)
            if crop is not None:
                prepared.append((bi, box, crop))

    crop_by_bi = {bi: (box, crop) for bi, box, crop in prepared}
    regions: list[dict[str, Any]] = []

    for bi, box in enumerate(boxes):
        label = box.get("label", "")
        region: dict[str, Any] = {
            "box_index": bi,
            "order": box.get("order"),
            "label": label,
            "cls_id": box.get("cls_id"),
            "layout_score": box.get("score"),
            "coordinate": box.get("coordinate"),
            "text": "",
            "rec_score": None,
        }

        if label in SKIP_RECOGNITION_LABELS:
            regions.append(region)
            continue

        item = crop_by_bi.get(bi)
        if item is None:
            regions.append(region)
            continue

        _, region_im = item
        region = _recognize_one_region(
            adapter,
            models,
            bi=bi,
            box=box,
            region_im=region_im,
            page_index=page_index,
            cfg=cfg,
            scratch_dir=scratch_dir,
            crops_dir=crops_dir,
        )
        regions.append(region)
        if label not in (TABLE_LABEL,) and label not in FORMULA_LABELS:
            order_str = region["order"] if region["order"] is not None else "-"
            text = region.get("text") or ""
            preview = text[:40] + "…" if len(text) > 40 else text
            rec_score = region.get("rec_score") or 0.0
            print(f"  [{order_str}] {label}: {preview!r} ({rec_score:.3f})")
        elif label == TABLE_LABEL:
            order_str = region["order"] if region["order"] is not None else "-"
            tr = region.get("table_result") or {}
            print(
                f"  [{order_str}] table ({tr.get('table_type')}): "
                f"HTML {len(region.get('text') or '')} 字符"
            )

    page_im.close()
    _save_layout_cache(scratch_dir, image_path=image_path, boxes=boxes)
    elapsed = time.perf_counter() - t0
    print(f"  本页耗时: {elapsed:.1f}s")

    out: dict[str, Any] = {
        "page_index": page_index,
        "image": str(image_path.resolve()),
        "regions": regions,
    }
    if layout_json:
        out["layout_json"] = layout_json
    return out

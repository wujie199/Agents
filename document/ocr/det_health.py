# -*- coding: utf-8 -*-
"""文本检测探针与 MKLDNN 自动回退。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from document.ocr.ocr_adapter import OcrPipelineAdapter, PipelineConfig


def _is_det_inference_error(exc: BaseException) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "onednn" in msg or "convertpirattribute" in msg
    return False


def probe_text_detection(det_model: object, scratch_dir: Path) -> bool:
    """用小图探测 det 是否可推理。"""
    from PIL import Image

    scratch_dir.mkdir(parents=True, exist_ok=True)
    probe_path = scratch_dir / "_det_probe.png"
    Image.new("RGB", (64, 32), color=(255, 255, 255)).save(probe_path)
    try:
        list(det_model.predict(str(probe_path), batch_size=1))
        return True
    except Exception as exc:
        if _is_det_inference_error(exc):
            return False
        raise


def ensure_det_available(
    adapter: OcrPipelineAdapter,
    cfg: PipelineConfig,
    *,
    scratch_dir: Path,
) -> PipelineConfig:
    """探针失败时关闭 MKLDNN 并重载文本模型。"""
    from dataclasses import replace

    if not cfg.device.startswith("cpu"):
        return cfg

    det = adapter.text_models.det
    if probe_text_detection(det, scratch_dir):
        return cfg

    if not cfg.enable_mkldnn:
        print("  警告: 文本 det 探针失败，后续可能对大块正文使用行切分退化识别")
        return cfg

    print("  文本 det 探针失败，自动关闭 MKLDNN 并重载模型…")
    adapter.reset_text_models()
    cfg = replace(cfg, enable_mkldnn=False)
    if probe_text_detection(adapter.text_models.det, scratch_dir):
        print("  det 探针通过（已关闭 MKLDNN）")
    else:
        print("  警告: 关闭 MKLDNN 后 det 仍失败，将使用行切分退化识别")
    return cfg

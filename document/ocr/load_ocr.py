"""OCR 模型权重目录与校验（默认 /Volumes/wj/model/ocr，可用 OCR_MODEL_ROOT 覆盖）。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from document.model_mount import (
    DEFAULT_OCR_MODEL_ROOT,
    require_mounted_volume,
    unmounted_reminder,
    warn_if_unmounted,
)

_log = logging.getLogger(__name__)

_OCR_DIR = Path(__file__).resolve().parent
_DEFAULT_ROOT = Path(
    os.environ.get("OCR_MODEL_ROOT", str(DEFAULT_OCR_MODEL_ROOT))
)

LAYOUT_MODEL_NAME = "PP-DocLayoutV3"
DET_MODEL_NAME = "PP-OCRv5_server_det"
REC_MODEL_NAME = "PP-OCRv5_server_rec"

TABLE_CLS_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
TABLE_STRUCT_MODEL_NAME = "SLANet_plus"
TABLE_CELL_WIRED_MODEL_NAME = "RT-DETR-L_wired_table_cell_det"
TABLE_CELL_WIRELESS_MODEL_NAME = "RT-DETR-L_wireless_table_cell_det"

DOC_ORI_MODEL_NAME = "PP-LCNet_x1_0_doc_ori"
UVDOC_MODEL_NAME = "UVDoc"
FORMULA_MODEL_NAME = os.environ.get(
    "OCR_FORMULA_MODEL", "PP-FormulaNet_plus-S"
)

PIPELINE_VERSION = "1.0.0"
REQUIRED_INFERENCE_FILES = ("inference.pdiparams", "inference.yml")

DEFAULT_TEST_PDF = str(_OCR_DIR / "test.pdf")


def get_model_root(override: str | Path | None = None) -> Path:
    if override is not None and str(override).strip():
        root = Path(override).expanduser().resolve()
    else:
        root = _DEFAULT_ROOT.expanduser().resolve()
    return root


def ensure_ocr_model_root(override: str | Path | None = None) -> Path:
    """校验外置盘已挂载；未挂载时抛出含提醒的 FileNotFoundError（不下载）。"""
    root = get_model_root(override)
    require_mounted_volume(
        root,
        purpose="OCR 模型（版面/检测/识别/表格）",
        env_hint="OCR_MODEL_ROOT",
    )
    if not root.is_dir():
        raise FileNotFoundError(
            unmounted_reminder(
                path=root,
                purpose="OCR 模型目录不存在",
                env_hint="OCR_MODEL_ROOT",
            )
        )
    missing = validate_core_models(root)
    if missing:
        _log.warning(
            "OCR 核心模型文件不完整（%s），请确认外置盘 %s 内 ocr 权重完整；"
            "本项目不会自动下载。",
            missing,
            root,
        )
    return root


def _model_dir(root: Path, name: str) -> Path:
    return root / name


def model_dirs(root: Path | None = None) -> dict[str, Path]:
    base = root or get_model_root()
    return {
        "layout": _model_dir(base, LAYOUT_MODEL_NAME),
        "det": _model_dir(base, DET_MODEL_NAME),
        "rec": _model_dir(base, REC_MODEL_NAME),
        "table_cls": _model_dir(base, TABLE_CLS_MODEL_NAME),
        "table_struct": _model_dir(base, TABLE_STRUCT_MODEL_NAME),
        "table_cell_wired": _model_dir(base, TABLE_CELL_WIRED_MODEL_NAME),
        "table_cell_wireless": _model_dir(base, TABLE_CELL_WIRELESS_MODEL_NAME),
        "doc_ori": _model_dir(base, DOC_ORI_MODEL_NAME),
        "uvdoc": _model_dir(base, UVDOC_MODEL_NAME),
        "formula": _model_dir(base, FORMULA_MODEL_NAME),
    }


def table_model_dirs(root: Path | None = None) -> tuple[Path, ...]:
    dirs = model_dirs(root)
    return (
        dirs["table_cls"],
        dirs["table_struct"],
        dirs["table_cell_wired"],
        dirs["table_cell_wireless"],
    )


TABLE_MODEL_DIRS = table_model_dirs()


def all_local_model_dirs(root: Path | None = None) -> dict[str, str]:
    """PPStructureV3 兼容参数字典（供旧代码引用）。"""
    dirs = model_dirs(root)
    base = root or get_model_root()
    struct = dirs["table_struct"]
    return {
        "layout_detection_model_name": LAYOUT_MODEL_NAME,
        "layout_detection_model_dir": str(dirs["layout"]),
        "text_detection_model_name": DET_MODEL_NAME,
        "text_detection_model_dir": str(dirs["det"]),
        "text_recognition_model_name": REC_MODEL_NAME,
        "text_recognition_model_dir": str(dirs["rec"]),
        "table_classification_model_name": TABLE_CLS_MODEL_NAME,
        "table_classification_model_dir": str(dirs["table_cls"]),
        "wireless_table_structure_recognition_model_name": TABLE_STRUCT_MODEL_NAME,
        "wireless_table_structure_recognition_model_dir": str(struct),
        "wired_table_structure_recognition_model_name": TABLE_STRUCT_MODEL_NAME,
        "wired_table_structure_recognition_model_dir": str(struct),
        "wired_table_cells_detection_model_name": TABLE_CELL_WIRED_MODEL_NAME,
        "wired_table_cells_detection_model_dir": str(dirs["table_cell_wired"]),
        "wireless_table_cells_detection_model_name": TABLE_CELL_WIRELESS_MODEL_NAME,
        "wireless_table_cells_detection_model_dir": str(dirs["table_cell_wireless"]),
    }


def validate_model_dir(model_dir: Path) -> list[str]:
    if not model_dir.is_dir():
        return list(REQUIRED_INFERENCE_FILES)
    return [
        name
        for name in REQUIRED_INFERENCE_FILES
        if not (model_dir / name).exists()
    ]


def validate_core_models(root: Path | None = None) -> dict[str, list[str]]:
    dirs = model_dirs(root)
    labels = {
        "layout": "版面",
        "det": "文本检测",
        "rec": "文本识别",
    }
    missing: dict[str, list[str]] = {}
    for key, label in labels.items():
        gaps = validate_model_dir(dirs[key])
        if gaps:
            missing[label] = gaps
    return missing


def ensure_table_deps() -> None:
    try:
        from paddlex.utils.deps import is_extra_available
    except ImportError as exc:
        raise ImportError(
            "未安装 paddlex，请执行: pip install \"paddlex[ocr]==3.5.2\""
        ) from exc
    if not is_extra_available("ocr"):
        raise RuntimeError(
            "表格识别需要 paddlex[ocr] 扩展依赖。\n"
            "请执行: pip install \"paddlex[ocr]==3.5.2\""
        )

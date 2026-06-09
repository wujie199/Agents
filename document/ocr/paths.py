"""本地 OCR / 表格模型路径（相对 load_ocr.MODEL_ROOT）。"""

from __future__ import annotations

from pathlib import Path

from document.ocr.load_ocr import (
    DEFAULT_TEST_PDF,
    DET_MODEL_NAME,
    FORMULA_MODEL_NAME,
    LAYOUT_MODEL_NAME,
    REC_MODEL_NAME,
    TABLE_CLS_MODEL_NAME,
    TABLE_CELL_WIRED_MODEL_NAME,
    TABLE_CELL_WIRELESS_MODEL_NAME,
    TABLE_STRUCT_MODEL_NAME,
    all_local_model_dirs,
    get_model_root,
    validate_model_dir,
)

_OCR_DIR = Path(__file__).resolve().parent
MODEL_ROOT = get_model_root()

LAYOUT = MODEL_ROOT / LAYOUT_MODEL_NAME
TEXT_DET = MODEL_ROOT / DET_MODEL_NAME
TEXT_REC = MODEL_ROOT / REC_MODEL_NAME
TABLE_CLS = MODEL_ROOT / TABLE_CLS_MODEL_NAME
TABLE_STRUCTURE = MODEL_ROOT / TABLE_STRUCT_MODEL_NAME
TABLE_CELL_WIRED = MODEL_ROOT / TABLE_CELL_WIRED_MODEL_NAME
TABLE_CELL_WIRELESS = MODEL_ROOT / TABLE_CELL_WIRELESS_MODEL_NAME
FORMULA = MODEL_ROOT / FORMULA_MODEL_NAME


def model_dir(name: str, *, root: Path | None = None) -> str:
    path = (root or get_model_root()) / name
    if not (path / "inference.yml").exists() and not (path / "inference.pdiparams").exists():
        raise FileNotFoundError(f"模型目录不完整: {path}")
    return str(path)

"""本地 OCR / 表格模型路径（相对 document/ocr/weights/）。"""

from pathlib import Path

_OCR_DIR = Path(__file__).resolve().parent
MODEL_ROOT = _OCR_DIR / "weights"

LAYOUT = MODEL_ROOT / "PP-DocLayoutV3"
TEXT_DET = MODEL_ROOT / "PP-OCRv5_server_det"
TEXT_REC = MODEL_ROOT / "PP-OCRv5_server_rec"
TABLE_CLS = MODEL_ROOT / "PP-LCNet_x1_0_table_cls"
TABLE_STRUCTURE = MODEL_ROOT / "SLANet_plus"
TABLE_CELL_WIRED = MODEL_ROOT / "RT-DETR-L_wired_table_cell_det"
TABLE_CELL_WIRELESS = MODEL_ROOT / "RT-DETR-L_wireless_table_cell_det"

DEFAULT_TEST_PDF = str(_OCR_DIR / "test.pdf")


def model_dir(name: str) -> str:
    path = MODEL_ROOT / name
    if not (path / "inference.yml").exists() and not (path / "inference.pdiparams").exists():
        raise FileNotFoundError(f"模型目录不完整: {path}")
    return str(path)


def all_local_model_dirs() -> dict:
    """返回 PPStructureV3 所需的本地模型目录参数字典。"""
    return {
        "layout_detection_model_name": "PP-DocLayoutV3",
        "layout_detection_model_dir": str(LAYOUT),
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_detection_model_dir": str(TEXT_DET),
        "text_recognition_model_name": "PP-OCRv5_server_rec",
        "text_recognition_model_dir": str(TEXT_REC),
        "table_classification_model_name": "PP-LCNet_x1_0_table_cls",
        "table_classification_model_dir": str(TABLE_CLS),
        "wireless_table_structure_recognition_model_name": "SLANet_plus",
        "wireless_table_structure_recognition_model_dir": str(TABLE_STRUCTURE),
        "wired_table_structure_recognition_model_name": "SLANet_plus",
        "wired_table_structure_recognition_model_dir": str(TABLE_STRUCTURE),
        "wired_table_cells_detection_model_name": "RT-DETR-L_wired_table_cell_det",
        "wired_table_cells_detection_model_dir": str(TABLE_CELL_WIRED),
        "wireless_table_cells_detection_model_name": "RT-DETR-L_wireless_table_cell_det",
        "wireless_table_cells_detection_model_dir": str(TABLE_CELL_WIRELESS),
    }

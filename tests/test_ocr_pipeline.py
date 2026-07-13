"""UniversalOcrPipeline 集成与 IR 转换测试（不加载 Paddle 模型）。"""

import pytest

from document.ocr.load_ocr import (
    FORMULA_MODEL_NAME,
    ensure_ocr_model_root,
    get_model_root,
    validate_model_dir,
)
from document.ocr.processor import (
    _document_ir_to_pdf_result,
    _region_ir_to_layout,
)
from document.rag.config import IngestConfig, RagPipelineConfig
from document.rag.components.ingest.registry import build_ingest
from document.rag.components.ingest.ocr_processor import OcrProcessorIngestAdapter


def test_get_model_root_override(tmp_path):
    root = get_model_root(str(tmp_path / "models"))
    assert root == (tmp_path / "models").resolve()


def test_formula_model_default():
    assert FORMULA_MODEL_NAME == "PP-FormulaNet_plus-S"


def test_region_ir_to_layout_table():
    region = {
        "label": "table",
        "coordinate": [10, 20, 100, 80],
        "content": {"type": "table", "html": "<table><tr><td>A</td></tr></table>"},
        "rec_score": 0.9,
    }
    layout = _region_ir_to_layout(region)
    assert layout.region_type == "table"
    assert "<table>" in layout.html
    assert layout.content
    assert len(layout.bbox) == 4


def test_region_ir_to_layout_text():
    region = {
        "label": "paragraph_title",
        "coordinate": [0, 0, 50, 20],
        "content": {"type": "text", "text": "标题"},
        "rec_score": 0.95,
    }
    layout = _region_ir_to_layout(region)
    assert layout.content == "标题"
    assert layout.confidence == 0.95


def test_document_ir_to_pdf_result():
    document = {
        "schema": "document_ir/1.0",
        "pipeline_version": "1.0.0",
        "pages": [
            {
                "page_index": 0,
                "image": "/tmp/page.png",
                "regions": [
                    {
                        "label": "text",
                        "content": {"type": "text", "text": "hello"},
                        "coordinate": [0, 0, 10, 10],
                    }
                ],
                "qc": {"status": "pass"},
            }
        ],
        "qc_summary": {"pass_pages": 1, "fail_pages": 0},
    }
    pdf = _document_ir_to_pdf_result(document, "/tmp/doc.pdf")
    assert pdf.total_pages == 1
    assert pdf.pages[0].full_text == "hello"
    assert pdf.metadata["pipeline"] == "UniversalOcrPipeline"


def test_build_ingest_passes_ocr_model_root():
    cfg = RagPipelineConfig(
        ingest=IngestConfig(
            mode="ocr_only",
            ocr_model_root="/Volumes/wj/model/ocr",
            ocr_preprocess="off",
            ocr_enable_formula=False,
        )
    )
    adapter = build_ingest(cfg)
    assert isinstance(adapter, OcrProcessorIngestAdapter)
    assert adapter._model_root == "/Volumes/wj/model/ocr"


def test_ensure_ocr_model_root_raises_when_unmounted(monkeypatch):
    monkeypatch.setattr(
        "document.model_mount.is_external_volume_mounted",
        lambda _p: False,
    )
    with pytest.raises(FileNotFoundError, match="外置模型盘未就绪"):
        ensure_ocr_model_root("/Volumes/wj/model/ocr")


def test_validate_model_dir_missing(tmp_path):
    missing = validate_model_dir(tmp_path / "nonexistent")
    assert "inference.pdiparams" in missing

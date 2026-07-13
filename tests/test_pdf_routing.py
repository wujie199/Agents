"""P0-P2：PDF 路由、IR 后处理、document_ir 分块测试。"""

from document.ocr.ir_postprocess import (
    filter_dropped_regions,
    merge_cross_page_tables,
    postprocess_document_ir,
)
from document.ocr.labels import DROP_LABELS
from document.ocr.reading_order import sort_boxes_reading_order
from document.ocr.pdf_classifier import PdfPageRoute, _classify_page_signals, PdfRouteConfig
from document.rag.application.chunking.ir_to_structure import units_from_document_ir
from document.rag.application.chunking.step1_structure import run_step1_structure
from document.rag.config.chunk_pipeline import ChunkPipelineConfig


def test_classify_native_page():
    cfg = PdfRouteConfig()
    route = _classify_page_signals(500, 595.0, 842.0, cfg)
    assert route == PdfPageRoute.NATIVE


def test_classify_scan_page():
    cfg = PdfRouteConfig()
    route = _classify_page_signals(5, 595.0, 842.0, cfg)
    assert route == PdfPageRoute.SCAN


def test_filter_dropped_regions():
    regions = [
        {"label": "header", "text": "页眉"},
        {"label": "text", "text": "正文", "content": {"type": "text", "text": "正文"}},
        {"label": "footer", "text": "页脚"},
    ]
    out = filter_dropped_regions(regions)
    assert len(out) == 1
    assert out[0]["label"] == "text"
    assert "header" in DROP_LABELS


def test_xy_cut_reading_order_two_columns():
    boxes = [
        {"order": None, "coordinate": [400, 10, 580, 30], "label": "text"},
        {"order": None, "coordinate": [10, 10, 280, 30], "label": "text"},
        {"order": None, "coordinate": [10, 50, 280, 70], "label": "text"},
        {"order": None, "coordinate": [400, 50, 580, 70], "label": "text"},
    ]
    sorted_boxes = sort_boxes_reading_order(boxes, page_width=600)
    xs = [b["coordinate"][0] for b in sorted_boxes]
    assert xs[0] < 300
    assert xs[-1] > 300


def test_merge_cross_page_tables():
    doc = {
        "pages": [
            {
                "page_index": 0,
                "regions": [
                    {
                        "label": "table",
                        "content": {
                            "type": "table",
                            "html": "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>",
                        },
                    }
                ],
            },
            {
                "page_index": 1,
                "regions": [
                    {
                        "label": "table",
                        "content": {
                            "type": "table",
                            "html": "<table><tr><td>A</td><td>B</td></tr><tr><td>3</td><td>4</td></tr></table>",
                        },
                    }
                ],
            },
        ]
    }
    merged = merge_cross_page_tables(doc)
    pages = merged["pages"]
    assert len(pages) == 2
    first_tables = [r for r in pages[0]["regions"] if r.get("label") == "table"]
    second_tables = [r for r in pages[1]["regions"] if r.get("label") == "table"]
    assert len(first_tables) == 1
    assert "3" in first_tables[0]["content"]["html"]
    assert len(second_tables) == 0


def test_units_from_document_ir_heading_path():
    ir = {
        "pages": [
            {
                "page_index": 0,
                "regions": [
                    {
                        "label": "doc_title",
                        "content": {"type": "text", "text": "第三章 合同"},
                        "coordinate": [0, 0, 100, 20],
                    },
                    {
                        "label": "text",
                        "content": {"type": "text", "text": "违约金为合同总额的百分之五。"},
                        "coordinate": [0, 30, 200, 50],
                    },
                ],
            }
        ]
    }
    units = units_from_document_ir(ir)
    assert len(units) >= 2
    body = [u for u in units if u.unit_type == "paragraph"]
    assert body
    assert "第三章" in body[0].heading_path
    assert body[0].metadata.get("page") == 1
    assert body[0].metadata.get("block_type") == "text"


def test_step1_uses_document_ir_when_present():
    ir = {
        "pages": [
            {
                "page_index": 0,
                "regions": [
                    {
                        "label": "paragraph_title",
                        "content": {"type": "text", "text": "安装步骤"},
                        "coordinate": [0, 0, 80, 20],
                    },
                    {
                        "label": "text",
                        "content": {"type": "text", "text": "请先安装依赖。"},
                        "coordinate": [0, 30, 200, 50],
                    },
                ],
            }
        ]
    }
    cfg = ChunkPipelineConfig()
    units = run_step1_structure(
        "ignored plain",
        cfg,
        {"document_ir": ir, "ingest_backend": "ocr_processor"},
    )
    assert any(u.heading_path for u in units)
    assert any(u.metadata.get("page") == 1 for u in units)


def test_postprocess_document_ir_filters_and_orders():
    doc = {
        "pages": [
            {
                "page_index": 0,
                "page_size": [600, 800],
                "regions": [
                    {"label": "footer", "text": "1", "coordinate": [0, 750, 50, 780]},
                    {"label": "text", "text": "hello", "content": {"type": "text", "text": "hello"}, "coordinate": [0, 10, 100, 30]},
                ],
            }
        ]
    }
    out = postprocess_document_ir(doc)
    regions = out["pages"][0]["regions"]
    assert all(r.get("label") != "footer" for r in regions)

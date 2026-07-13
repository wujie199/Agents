"""OCR 结构保留清洗与 Step1 FAQ/IR 路径测试。"""

from core.ports.ingest import DocumentFormat, IngestResult, IngestStatus
from document.rag.application.cleaning_pipeline import apply_ingest_cleaning
from document.rag.application.chunking.ir_to_structure import units_from_document_ir
from document.rag.application.chunking.step1_structure import run_step1_structure
from document.rag.config import RagPipelineConfig
from document.rag.config.chunk_pipeline import ChunkPipelineConfig
from document.rag.shared.ocr_ingest_text import rebuild_content_from_document_ir


SAMPLE_IR = {
    "pages": [
        {
            "page_index": 0,
            "regions": [
                {
                    "label": "paragraph_title",
                    "text": "#扫地机器人100问",
                    "content": {"type": "text", "text": "#扫地机器人100问"},
                },
                {
                    "label": "text",
                    "text": "6.**吸力单位Pa？**\n-帕斯卡(Pascal)，2000Pa 可吸起绿豆，1.5-2cm 越障。",
                    "content": {
                        "type": "text",
                        "text": "6.**吸力单位Pa？**\n-帕斯卡(Pascal)，2000Pa 可吸起绿豆，1.5-2cm 越障。",
                    },
                },
                {
                    "label": "paragraph_title",
                    "text": "12.**最大越障高度是多少？**",
                    "content": {
                        "type": "text",
                        "text": "12.**最大越障高度是多少？**",
                    },
                },
            ],
        }
    ]
}

FAQ_BODY = """#扫地机器人100问
1.**LDS激光导航和VSLAM视觉导航哪个更好？**
-LDS精度更高、不受光线影响。
2.**什么是dToF导航技术？**
-直接飞行时间测距。
3.**为什么有些机器人会"迷路"？**
-环境光线变化导致定位丢失。
4.**如何提高建图精度？**
-选择激光雷达+AI算法机型。
5.**吸力单位Pa是什么意思？**
-帕斯卡(Pascal)，2000Pa。
"""

FAQ_IR_TEXT = f"=== 第 1 页 ===\n{FAQ_BODY}"


def test_rebuild_content_from_document_ir():
    text = rebuild_content_from_document_ir({"document_ir": SAMPLE_IR})
    assert text is not None
    assert "#扫地机器人100问" in text
    assert "1.5-2cm" in text
    assert "2000Pa" in text


def test_ocr_structure_preserving_cleaning_skips_special_char_strip():
    cfg = RagPipelineConfig()
    ingest = IngestResult(
        content="legacy flat text without hash",
        metadata={
            "ingest_backend": "ocr_processor",
            "document_ir": SAMPLE_IR,
            "doc_id": "doc_test",
        },
        status=IngestStatus.SUCCESS,
        pages=[{"page_num": 1, "content": "legacy"}],
    )
    apply_ingest_cleaning(ingest, DocumentFormat.PDF, cfg)
    assert ingest.metadata.get("cleaning_mode") == "ocr_structure_light"
    assert "#扫地机器人100问" in ingest.content
    assert "1.5-2cm" in ingest.content
    assert "2000Pa" in ingest.content
    assert ingest.pages[0]["content"].startswith("#扫地机器人100问")


def test_ir_heading_skips_numbered_faq_question():
    units = units_from_document_ir({"pages": SAMPLE_IR["pages"]})
    paragraph_units = [u for u in units if u.unit_type == "paragraph"]
    assert any("12.**最大越障高度" in u.content for u in paragraph_units)
    q6 = [u for u in units if "6.**" in u.content]
    assert q6
    assert "12.**" not in q6[0].heading_path


def test_step1_prefers_faq_split_for_100问(tmp_path):
    pdf_path = tmp_path / "扫地机器人100问.pdf"
    pdf_path.write_text("x", encoding="utf-8")
    cfg = ChunkPipelineConfig(domain="faq", preserve_faq_pairs=True)
    meta = {
        "ingest_backend": "ocr_processor",
        "source_path": str(pdf_path),
        "document_ir": {
            "pages": [
                {
                    "page_index": 0,
                    "regions": [
                        {
                            "label": "text",
                            "text": FAQ_BODY,
                            "content": {"type": "text", "text": FAQ_BODY},
                        }
                    ],
                }
            ]
        },
    }
    units = run_step1_structure(FAQ_IR_TEXT, cfg, meta)
    qa_units = [u for u in units if u.unit_type == "qa"]
    assert len(qa_units) >= 2
    assert any("LDS" in u.content for u in qa_units)

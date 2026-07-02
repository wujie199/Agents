"""FAQ 专用切块测试。"""

from document.rag.application.indexing.faq_chunker import (
    FaqChunker,
    extract_trailing_section,
    format_faq_block,
    is_section_title,
    normalize_faq_text,
    split_faq_items,
)


OCR_SAMPLE = (
    "影响VSLAM成本更低可识别更多物体细节"
    "3什么是dToF导航技术 -直接飞行时间测距directTime-of-Flight比传统LDS测距更精准探测距离可达10米"
    "4为什么有些机器人会\"迷路\" -环境光线变化反光表面干扰或传感器故障导致定位丢失"
    "5如何提高扫地机器人的建图精度 -选择配备激光雷达AI算法的机型保持环境光线稳定定期清洁传感器"
)

Q5_WITH_SECTION = (
    "一基础与技术类 导航与路径规划 "
    "5如何提高扫地机器人的建图精度 -选择配备激光雷达AI算法的机型保持环境光线稳定定期清洁传感器"
    "清洁系统"
    "6扫地机器人的吸力单位Pa是什么意思 -帕斯卡Pascal表示真空吸力大小"
)


def test_normalize_inserts_newlines_before_question_numbers():
    text = normalize_faq_text(OCR_SAMPLE)
    assert "\n3" in text
    assert "\n4" in text
    assert "\n5" in text


def test_is_section_title():
    assert is_section_title("清洁系统")
    assert is_section_title("一基础与技术类")
    assert is_section_title("导航与路径规划")
    assert not is_section_title("5. 如何提高建图精度？")


def test_extract_trailing_section():
    clean, title = extract_trailing_section(
        "选择配备激光雷达AI算法的机型保持环境光线稳定定期清洁传感器\n清洁系统"
    )
    assert title == "清洁系统"
    assert "清洁系统" not in clean
    assert "定期清洁传感器" in clean


def test_split_faq_items_one_per_question():
    items = split_faq_items(OCR_SAMPLE)
    assert len(items) >= 3
    assert items[0].content.startswith("3.")
    assert items[1].faq_number == "4"
    assert items[2].faq_number == "5"


def test_format_faq_block_strips_trailing_section():
    item = format_faq_block(
        "5如何提高扫地机器人的建图精度 -选择配备激光雷达AI算法的机型保持环境光线稳定定期清洁传感器清洁系统"
    )
    assert "清洁系统" not in item.content
    assert "建图精度" in item.content


def test_faq_chunker_section_metadata():
    chunker = FaqChunker(chunk_size=800)
    chunks = chunker.chunk(Q5_WITH_SECTION, "doc_test")
    q5 = next(c for c in chunks if c.metadata.get("faq_number") == "5")
    assert "清洁系统" not in q5.content
    assert q5.metadata.get("faq_category") == "一基础与技术类"
    assert q5.metadata.get("faq_section") == "导航与路径规划"

    q6 = next(c for c in chunks if c.metadata.get("faq_number") == "6")
    assert q6.metadata.get("faq_section") == "清洁系统"


def test_faq_chunker_single_question_per_chunk():
    chunker = FaqChunker(chunk_size=800)
    chunks = chunker.chunk(OCR_SAMPLE, "doc_test")
    for chunk in chunks:
        assert chunk.content.lstrip()[0].isdigit()


def test_sanitize_faq_content_removes_dash_question_mark():
    from document.rag.application.indexing.faq_chunker import format_faq_block, sanitize_faq_content

    item = format_faq_block("4 APP无法连接机器人怎么办 -确认手机和机器人连接同一WiFi")
    assert "-？" not in item.content
    assert sanitize_faq_content("1. 问题\n-？\n答案") == "1. 问题\n答案"

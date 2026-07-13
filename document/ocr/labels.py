# -*- coding: utf-8 -*-
"""版面 label 常量（qc / adapter / document_ir 共用）。"""

ASSET_LABELS = frozenset(
    {
        "image",
        "chart",
        "seal",
        "footer_image",
        "header_image",
    }
)

# 版面层直接丢弃，不参与 OCR 与正文拼接（P1）
DROP_LABELS = frozenset(
    {
        "header",
        "footer",
        "number",
        "footnote",
        "footer_image",
        "header_image",
        "watermark",
    }
)

FORMULA_LABELS = frozenset({"inline_formula", "display_formula"})

TABLE_LABEL = "table"

TITLE_LABELS = frozenset(
    {
        "doc_title",
        "paragraph_title",
        "figure_title",
        "table_title",
    }
)

DIRECT_REC_LABELS = frozenset(
    {
        "header",
        "footer",
        "paragraph_title",
        "doc_title",
        "figure_title",
        "table_title",
        "number",
        "footnote",
        "formula_number",
    }
)

SKIP_RECOGNITION_LABELS = ASSET_LABELS | DROP_LABELS

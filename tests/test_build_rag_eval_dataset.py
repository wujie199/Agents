# -*- coding: utf-8 -*-
"""黄金集构建：PDF / TXT / 合并 test_docs。"""

from __future__ import annotations

from pathlib import Path

import pytest

from document.rag.evaluation.dataset import load_eval_dataset

REPO = Path(__file__).resolve().parents[1]
PDF = REPO / "data/test_docs/扫地机器人100问.pdf"
TXT_DIR = REPO / "data/test_docs"
FAQ2 = TXT_DIR / "扫地机器人100问2.txt"
MERGED = REPO / "data/rag_eval/golden/test_docs_merged.jsonl"

@pytest.mark.skipif(not MERGED.is_file(), reason="merged golden not built")
def test_merged_golden_dataset():
    samples = load_eval_dataset(MERGED)
    assert len(samples) >= 400
    ids = {s.id for s in samples}
    assert len(ids) == len(samples)
    assert any(s.id.startswith("faq2-") for s in samples)
    assert any(s.id.startswith("buy-") for s in samples)
    assert any(s.id.startswith("maint-") for s in samples)
    for s in samples[:20]:
        assert s.ground_truth
        assert s.reference_contexts
        assert s.tenant_id == "default"
        assert not s.ground_truth.startswith("？")
        assert "-？" not in s.reference_contexts[0]
    buy = next(s for s in samples if s.id.startswith("buy-"))
    assert "选购扫地机器人时" in buy.question
    assert "方面有哪些建议" not in buy.question


@pytest.mark.skipif(not FAQ2.is_file(), reason="FAQ2 txt missing")
def test_build_samples_from_txt_align_index():
    from document.rag.evaluation.dataset_builder import build_samples_from_txt

    rows = build_samples_from_txt(FAQ2, align_index=True)
    assert len(rows) >= 90
    assert "首次使用" in rows[0].question
    assert rows[0].reference_contexts[0].startswith("1.")


@pytest.mark.skipif(not FAQ2.is_file(), reason="FAQ2 txt missing")
def test_build_merged_from_txt_dir(tmp_path):
    from document.rag.evaluation.dataset_builder import build_merged_from_txt_dir

    out = tmp_path / "merged.jsonl"
    meta = build_merged_from_txt_dir(TXT_DIR, out, glob="扫地机器人100问2.txt")
    assert meta["count"] >= 90
    assert load_eval_dataset(out)


@pytest.mark.skipif(not PDF.is_file(), reason="FAQ PDF missing")
def test_build_pdf_dataset(tmp_path):
    from document.rag.evaluation.dataset_builder import build_pdf_dataset

    out = tmp_path / "out.jsonl"
    meta = build_pdf_dataset(PDF, out, target_count=100)
    assert meta["count"] == 100
    assert load_eval_dataset(out)

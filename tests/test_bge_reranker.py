"""本地 BGE reranker 单元测试（需 weights 与 sentence-transformers）。"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from document.rag.adapters.rerank.local_bge import LocalBgeReranker, print_rerank_report

pytestmark = pytest.mark.skipif(
    not (__import__("pathlib").Path(__file__).resolve().parents[1]
         / "document/rag/weights/bge-reranker-base/model.safetensors").is_file(),
    reason="bge-reranker-base weights not found",
)


@pytest.fixture(scope="module")
def reranker():
    st = pytest.importorskip("sentence_transformers")
    del st
    return LocalBgeReranker(device="cpu")


def test_rerank_orders_relevant_first(reranker):
    query = "异步 HTTP 请求 Python"
    docs = [
        "今天天气很好。",
        "httpx 和 aiohttp 支持 async 异步 HTTP。",
        "红烧肉的做法。",
    ]
    out = reranker.rerank(query, docs, top_n=len(docs))
    print_rerank_report(query, docs, out, top_n_label=len(docs))
    assert len(out) >= 2
    top2 = out[:2]
    assert top2[0]["index"] == 1
    assert top2[0]["score"] > top2[1]["score"]


def test_rerank_empty_docs(reranker):
    assert reranker.rerank("q", [], top_n=5) == []


if __name__ == "__main__":
    # -s：显示 print_rerank_report 的重排结果
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

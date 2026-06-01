"""本地 BAAI/bge-reranker-base 重排（sentence-transformers CrossEncoder）。"""

from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "weights" / "bge-reranker-base"
)


class LocalBgeReranker:
    """
    使用本地 bge-reranker-base 对 query-document 对打分重排。
    接口与 core.ports.rag.rerank.RerankPort / apply_rerank 一致。
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder

        self._model_dir = str(
            Path(model_dir).resolve() if model_dir else _DEFAULT_MODEL_DIR
        )
        if not (Path(self._model_dir) / "config.json").is_file():
            raise FileNotFoundError(
                f"本地 reranker 目录不完整: {self._model_dir}\n"
                "请确认存在 config.json 与 model.safetensors（或 pytorch_model.bin）"
            )
        kwargs: Dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        self._model = CrossEncoder(
            self._model_dir,
            max_length=max_length,
            **kwargs,
        )

    @property
    def model_dir(self) -> str:
        return self._model_dir

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        pairs = [[query, doc or ""] for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(
            enumerate(float(s) for s in scores),
            key=lambda x: x[1],
            reverse=True,
        )
        limit = min(top_n, len(ranked)) if top_n > 0 else len(ranked)
        return [
            {"index": idx, "score": score}
            for idx, score in ranked[:limit]
        ]

    def score_pairs(self, query: str, documents: List[str]) -> List[float]:
        """返回与 documents 等长的分数列表（不重排）。"""
        if not documents:
            return []
        pairs = [[query, doc or ""] for doc in documents]
        return [float(s) for s in self._model.predict(pairs)]


def print_rerank_report(
    query: str,
    documents: List[str],
    results: List[Dict[str, Any]],
    *,
    top_n_label: Optional[int] = None,
) -> None:
    """将 rerank 结果打印到 stdout（pytest 需加 -s 才能看到）。"""
    label = top_n_label if top_n_label is not None else len(results)
    print(f"\n查询: {query!r}")
    print(f"候选 {len(documents)} 条 → 展示 Top {label}\n")
    print("--- 重排前（原始顺序）---")
    for i, doc in enumerate(documents):
        preview = doc if len(doc) <= 100 else doc[:100] + "..."
        print(f"  [{i}] {preview}")
    print(f"\n--- 重排后 ---")
    if not results:
        print("  (无结果)")
        return
    for rank, item in enumerate(results, 1):
        idx = int(item["index"])
        score = float(item["score"])
        doc = documents[idx]
        preview = doc if len(doc) <= 100 else doc[:100] + "..."
        print(f"  #{rank}  index={idx}  score={score:.4f}")
        print(f"      {preview}")
    print()

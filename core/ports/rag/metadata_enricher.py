"""元数据 enrichment 端口（规则打标、LLM 打标等实现均适配此接口）。"""

from typing import Any, Dict, List, Optional, Protocol

from core.ports.rag.ingest import IngestResult


class MetadataEnricherPort(Protocol):
    """对摄取结果补充/归一化文档级 metadata（如 tags、categories）。"""

    def enrich(
        self,
        ingest_result: IngestResult,
        *,
        doc_format: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        ...

    def enrich_batch(
        self,
        items: List[IngestResult],
        **kwargs: Any,
    ) -> List[IngestResult]:
        ...

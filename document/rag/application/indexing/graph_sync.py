import logging
import re
from typing import Any, Dict, List, Optional


_ENTITY_PATTERN = re.compile(
    r"\b([A-Z]{2,6}-\d{2,6}|[A-Z]{2,6}\d{2,6})\b"
)


class RagGraphSync:
    """Sync document nodes and entity relations to graph store on index."""

    def __init__(self, graph_port: Any):
        self._graph = graph_port
        self._logger = logging.getLogger("knowledge.pipeline.index.graph_sync")

    def upsert_document(
        self,
        doc_id: str,
        tenant_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not hasattr(self._graph, "upsert_node"):
            self._logger.debug("Graph port has no upsert_node; skip graph sync")
            return

        meta = metadata or {}
        title = meta.get("title") or doc_id
        self._graph.upsert_node(
            doc_id,
            "Document",
            {"doc_id": doc_id, "tenant_id": tenant_id, "title": title},
        )

        for entity in self._extract_entities(content)[:20]:
            self._graph.upsert_node(
                entity,
                "Entity",
                {"name": entity, "tenant_id": tenant_id},
            )
            if hasattr(self._graph, "create_edge"):
                try:
                    self._graph.create_edge(
                        doc_id,
                        entity,
                        "MENTIONS",
                        {"tenant_id": tenant_id},
                    )
                except ValueError:
                    pass

    def delete_document(self, doc_id: str) -> None:
        if hasattr(self._graph, "delete_node"):
            self._graph.delete_node(doc_id)

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        return list(dict.fromkeys(_ENTITY_PATTERN.findall(text or "")))

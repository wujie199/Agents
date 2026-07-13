"""父 Chunk 持久化存储。"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from document.rag.application.chunking.models import ScoredChunk

_logger = logging.getLogger("rag.chunking.parent_store")


class ParentChunkStore:
    """按 collection/doc 存储父 chunk 原文，供检索后扩展上下文。"""

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)

    def save(
        self,
        collection: str,
        doc_id: str,
        parents: List[ScoredChunk],
    ) -> None:
        if not parents:
            return
        out_dir = self._base / collection
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{doc_id}.json"
        payload = [
            {
                "chunk_id": p.metadata.get("chunk_id"),
                "content": p.content,
                "heading_path": p.heading_path,
                "child_ids": p.child_ids,
                "metadata": p.metadata,
            }
            for p in parents
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _logger.debug("Saved %d parent chunks to %s", len(parents), path)

    def load(self, collection: str, doc_id: str) -> List[Dict[str, Any]]:
        path = self._base / collection / f"{doc_id}.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def get_by_id(
        self,
        collection: str,
        doc_id: str,
        parent_id: str,
    ) -> Optional[str]:
        for item in self.load(collection, doc_id):
            if item.get("chunk_id") == parent_id:
                return item.get("content")
        return None

    def delete(self, collection: str, doc_id: str) -> None:
        path = self._base / collection / f"{doc_id}.json"
        if path.exists():
            path.unlink()

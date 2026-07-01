"""Rerank 端口 — 从 core.ports 重新导出，保持单一来源。"""

from core.ports.rag.rerank import RerankPort

__all__ = ["RerankPort"]

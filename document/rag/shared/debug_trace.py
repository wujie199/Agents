"""Debug 会话 NDJSON 埋点 + 流水线逐步 artifact（OCR / 七步分块）。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEBUG_LOG = Path(__file__).resolve().parents[3] / ".cursor" / "debug-43cec4.log"
_SESSION_ID = "43cec4"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "data" / "rag_offline" / "pipeline_trace"

_trace_context: Dict[str, Any] = {}


def debug_trace_enabled() -> bool:
    """默认开启；设 RAG_DEBUG_TRACE=0 关闭。"""
    return os.environ.get("RAG_DEBUG_TRACE", "1") != "0"


def set_trace_context(**kwargs: Any) -> None:
    """建库/OCR 入口设置 doc_id、source_path 等，供子步骤 artifact 命名。"""
    _trace_context.update({k: v for k, v in kwargs.items() if v is not None})


def get_trace_doc_id(fallback: str = "unknown") -> str:
    doc_id = _trace_context.get("doc_id")
    return str(doc_id) if doc_id else fallback


def pipeline_trace_dir() -> Path:
    raw = os.environ.get("RAG_PIPELINE_TRACE_DIR", str(_DEFAULT_ARTIFACT_DIR))
    return Path(raw)


def preview_text(text: str, max_chars: int = 400) -> str:
    t = (text or "").strip().replace("\r\n", "\n")
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + f"...[+{len(t) - max_chars} chars]"


def sample_texts(items: List[str], n: int = 2, max_chars: int = 300) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(items[:n]):
        text = (raw or "").strip()
        out.append({"index": i, "chars": len(text), "preview": preview_text(text, max_chars)})
    return out


def sample_chunks(chunks: Any, n: int = 2, max_chars: int = 300) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(list(chunks)[:n]):
        content = getattr(c, "content", None)
        if content is None and isinstance(c, dict):
            content = c.get("content", "")
        meta = getattr(c, "metadata", None) or (c.get("metadata") if isinstance(c, dict) else {}) or {}
        out.append(
            {
                "index": i,
                "chars": len(content or ""),
                "preview": preview_text(str(content or ""), max_chars),
                "unit_type": meta.get("unit_type") if isinstance(meta, dict) else getattr(c, "unit_type", None),
                "heading_path": getattr(c, "heading_path", None) or (meta.get("heading_path") if isinstance(meta, dict) else None),
            }
        )
    return out


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def summarize_structural_units(units: Any, *, max_items: int = 50) -> Dict[str, Any]:
    items = list(units or [])
    type_counts: Dict[str, int] = {}
    serialized: List[Dict[str, Any]] = []
    for u in items[:max_items]:
        ut = getattr(u, "unit_type", None) or (u.get("unit_type") if isinstance(u, dict) else "?")
        type_counts[str(ut)] = type_counts.get(str(ut), 0) + 1
        content = getattr(u, "content", None) or (u.get("content") if isinstance(u, dict) else "")
        serialized.append(
            {
                "unit_type": ut,
                "heading_path": getattr(u, "heading_path", None) or (u.get("heading_path") if isinstance(u, dict) else ""),
                "position": getattr(u, "position", None) or (u.get("position") if isinstance(u, dict) else None),
                "chars": len(content or ""),
                "preview": preview_text(str(content or ""), 200),
            }
        )
    for u in items[max_items:]:
        ut = getattr(u, "unit_type", None) or (u.get("unit_type") if isinstance(u, dict) else "?")
        type_counts[str(ut)] = type_counts.get(str(ut), 0) + 1
    return {
        "total": len(items),
        "type_counts": type_counts,
        "items": serialized,
        "truncated": len(items) > max_items,
    }


def summarize_boundaries(boundaries: Any) -> Dict[str, Any]:
    def _cuts(name: str) -> List[Dict[str, Any]]:
        pts = getattr(boundaries, name, None) or []
        out: List[Dict[str, Any]] = []
        for p in pts:
            out.append(
                {
                    "sentence_index": getattr(p, "sentence_index", None),
                    "confidence": getattr(p, "confidence", None),
                    "reason": getattr(p, "reason", None),
                }
            )
        return out

    forbidden = getattr(boundaries, "forbidden", None) or []
    return {
        "confirmed": _cuts("confirmed"),
        "weak_a": _cuts("weak_a"),
        "weak_b": _cuts("weak_b"),
        "forbidden": [
            {
                "start": getattr(f, "start", None),
                "end": getattr(f, "end", None),
                "forbidden_type": getattr(f, "forbidden_type", None),
            }
            for f in forbidden
        ],
    }


def summarize_scored_chunks(chunks: Any, *, max_items: int = 100) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(list(chunks or [])[:max_items]):
        content = getattr(c, "content", None) or (c.get("content") if isinstance(c, dict) else "")
        out.append(
            {
                "index": i,
                "chars": len(content or ""),
                "unit_type": getattr(c, "unit_type", None) or (c.get("unit_type") if isinstance(c, dict) else None),
                "heading_path": getattr(c, "heading_path", None) or (c.get("heading_path") if isinstance(c, dict) else None),
                "score": getattr(c, "score", None) or (c.get("score") if isinstance(c, dict) else None),
                "quality": getattr(c, "quality", None) or (c.get("quality") if isinstance(c, dict) else None),
                "chunk_role": getattr(c, "chunk_role", None) or (c.get("chunk_role") if isinstance(c, dict) else None),
                "preview": preview_text(str(content or ""), 300),
            }
        )
    return out


def summarize_ocr_regions(regions: Any, *, max_items: int = 200) -> Dict[str, Any]:
    items = list(regions or [])
    label_counts: Dict[str, int] = {}
    serialized: List[Dict[str, Any]] = []
    for r in items[:max_items]:
        label = str(r.get("label", "?") if isinstance(r, dict) else "?")
        label_counts[label] = label_counts.get(label, 0) + 1
        text = r.get("text") if isinstance(r, dict) else ""
        serialized.append(
            {
                "box_index": r.get("box_index"),
                "order": r.get("order"),
                "label": r.get("label"),
                "layout_score": r.get("layout_score"),
                "rec_score": r.get("rec_score"),
                "route": r.get("route") or r.get("source"),
                "chars": len(text or ""),
                "preview": preview_text(str(text or ""), 150),
            }
        )
    for r in items[max_items:]:
        label = str(r.get("label", "?") if isinstance(r, dict) else "?")
        label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "total": len(items),
        "label_counts": label_counts,
        "regions": serialized,
        "truncated": len(items) > max_items,
    }


def dump_pipeline_artifact(
    phase: str,
    step: str,
    payload: Any,
    *,
    doc_id: Optional[str] = None,
) -> Optional[str]:
    """将单步完整结果写入 data/rag_offline/pipeline_trace/{doc_id}/{phase}_{step}.json。"""
    if not debug_trace_enabled():
        return None
    doc = doc_id or get_trace_doc_id("unknown")
    safe_doc = "".join(c if c.isalnum() or c in "-_" else "_" for c in doc)
    out_dir = pipeline_trace_dir() / safe_doc
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{phase}_{step}.json"
        body = {
            "sessionId": _SESSION_ID,
            "doc_id": doc,
            "phase": phase,
            "step": step,
            "timestamp": int(time.time() * 1000),
            "payload": _json_safe(payload),
        }
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
    except OSError:
        return None


def trace_pipeline_step(
    phase: str,
    step: str,
    message: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    artifact: Any = None,
    doc_id: Optional[str] = None,
    hypothesis_id: Optional[str] = None,
    run_id: str = "build",
) -> None:
    """NDJSON 摘要 + 可选 artifact 落盘。"""
    artifact_path = None
    if artifact is not None:
        artifact_path = dump_pipeline_artifact(phase, step, artifact, doc_id=doc_id)
    payload = dict(data or {})
    if artifact_path:
        payload["artifact_path"] = artifact_path
    debug_trace(
        f"{phase}/{step}",
        message,
        data=payload,
        hypothesis_id=hypothesis_id,
        run_id=run_id,
    )


def debug_trace(
    location: str,
    message: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    hypothesis_id: Optional[str] = None,
    run_id: str = "build",
) -> None:
    if not debug_trace_enabled():
        return
    entry: Dict[str, Any] = {
        "sessionId": _SESSION_ID,
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data or {},
        "runId": run_id,
    }
    if hypothesis_id:
        entry["hypothesisId"] = hypothesis_id
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass

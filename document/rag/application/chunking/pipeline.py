"""七步分块主流水线编排。"""

import logging
from typing import Any, Callable, Dict, List, Optional

from document.rag.config.chunk_pipeline import ChunkPipelineConfig
from document.rag.application.chunking.models import PipelineResult, ScoredChunk
from document.rag.application.chunking.step1_structure import run_step1_structure
from document.rag.application.chunking.step2_boundaries import run_step2_boundaries
from document.rag.application.chunking.step3_granularity import run_step3_granularity
from document.rag.application.chunking.step4_hierarchy import run_step4_hierarchy
from document.rag.application.chunking.step5_quality import run_step5_quality
from document.rag.application.chunking.step6_repair import run_step6_repair
from document.rag.application.chunking.step7_dedupe import run_step7_dedupe
from document.rag.application.chunking.text_utils import flatten_unit_sentences
from document.rag.shared.debug_trace import (
    preview_text,
    sample_chunks,
    set_trace_context,
    summarize_boundaries,
    summarize_scored_chunks,
    summarize_structural_units,
    trace_pipeline_step,
)

_logger = logging.getLogger("rag.chunking.pipeline")


class SevenStepChunkPipeline:
    """Step1–Step7 完整分块流水线。"""

    def __init__(
        self,
        cfg: ChunkPipelineConfig,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self._cfg = cfg.with_domain(cfg.domain)
        self._embed_fn = embed_fn

    def run(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        meta = dict(metadata or {})
        domain = str(meta.get("chunk_domain") or meta.get("doc_domain") or self._cfg.domain)
        cfg = self._cfg.with_domain(domain)
        set_trace_context(doc_id=doc_id, phase="chunking")

        # #region agent log
        trace_pipeline_step(
            "chunking",
            "run_start",
            "七步分块开始",
            data={
                "doc_id": doc_id,
                "domain": domain,
                "input_chars": len(content or ""),
                "input_preview": preview_text(content or "", 500),
                "has_document_ir": bool(meta.get("document_ir")),
                "ingest_backend": meta.get("ingest_backend"),
            },
            artifact={
                "doc_id": doc_id,
                "domain": domain,
                "input_chars": len(content or ""),
                "content": content or "",
                "metadata_keys": sorted(meta.keys()),
            },
            doc_id=doc_id,
            hypothesis_id="H3",
        )
        # #endregion

        # Step1
        units = run_step1_structure(content, cfg, meta)
        sentences = flatten_unit_sentences(units)
        # #region agent log
        trace_pipeline_step(
            "chunking",
            "step1_structure",
            "Step1 结构识别",
            data={
                "doc_id": doc_id,
                "units": len(units),
                "sentences": len(sentences),
            },
            artifact=summarize_structural_units(units),
            doc_id=doc_id,
            hypothesis_id="H3",
        )
        # #endregion

        # Step2
        boundaries = run_step2_boundaries(units, sentences, cfg, self._embed_fn)
        # #region agent log
        trace_pipeline_step(
            "chunking",
            "step2_boundaries",
            "Step2 语义边界",
            data={
                "doc_id": doc_id,
                "confirmed_cuts": len(boundaries.confirmed),
                "weak_a_cuts": len(boundaries.weak_a),
                "weak_b_cuts": len(boundaries.weak_b),
                "forbidden": len(boundaries.forbidden),
            },
            artifact={
                "sentences_count": len(sentences),
                "boundaries": summarize_boundaries(boundaries),
            },
            doc_id=doc_id,
            hypothesis_id="H4",
        )
        # #endregion

        # Step3
        base_chunks = run_step3_granularity(units, sentences, boundaries, cfg)
        # #region agent log
        trace_pipeline_step(
            "chunking",
            "step3_granularity",
            "Step3 粒度切分",
            data={"doc_id": doc_id, "base_chunks": len(base_chunks)},
            artifact={"chunks": summarize_scored_chunks(base_chunks)},
            doc_id=doc_id,
            hypothesis_id="H4",
        )
        # #endregion

        # Step4
        retrieval, parents = run_step4_hierarchy(base_chunks, doc_id, cfg)
        # #region agent log
        trace_pipeline_step(
            "chunking",
            "step4_hierarchy",
            "Step4 父子层级",
            data={
                "doc_id": doc_id,
                "retrieval_chunks": len(retrieval),
                "parent_chunks": len(parents),
            },
            artifact={
                "retrieval": summarize_scored_chunks(retrieval),
                "parents": summarize_scored_chunks(parents),
            },
            doc_id=doc_id,
            hypothesis_id="H4",
        )
        # #endregion

        # Step5（对 retrieval chunks 评分）
        scored, repair_tasks = run_step5_quality(retrieval, cfg)
        # #region agent log
        trace_pipeline_step(
            "chunking",
            "step5_quality",
            "Step5 质量评分",
            data={
                "doc_id": doc_id,
                "scored": len(scored),
                "repair_tasks": len(repair_tasks),
                "avg_score": round(
                    sum(getattr(c, "score", 0) or 0 for c in scored) / max(len(scored), 1),
                    4,
                ),
            },
            artifact={
                "scored": summarize_scored_chunks(scored),
                "repair_tasks": [
                    {
                        "chunk_index": getattr(t, "chunk_index", None),
                        "task_type": getattr(t, "task_type", None),
                        "detail": getattr(t, "detail", None),
                    }
                    for t in repair_tasks
                ],
            },
            doc_id=doc_id,
            hypothesis_id="H4",
        )
        # #endregion

        # Step6
        repaired = run_step6_repair(scored, repair_tasks, content, cfg)
        # #region agent log
        trace_pipeline_step(
            "chunking",
            "step6_repair",
            "Step6 上下文修复",
            data={"doc_id": doc_id, "repaired": len(repaired)},
            artifact={"chunks": summarize_scored_chunks(repaired)},
            doc_id=doc_id,
            hypothesis_id="H4",
        )
        # #endregion

        # Step7
        final = run_step7_dedupe(
            repaired,
            cfg,
            embed_fn=self._embed_fn,
            has_parent_child=cfg.enable_parent_child and bool(parents),
        )
        stats = {
            "units": len(units),
            "sentences": len(sentences),
            "confirmed_cuts": len(boundaries.confirmed),
            "weak_a_cuts": len(boundaries.weak_a),
            "weak_b_cuts": len(boundaries.weak_b),
            "base_chunks": len(base_chunks),
            "retrieval_chunks": len(final),
            "parent_chunks": len(parents),
            "repair_tasks": len(repair_tasks),
            "domain": domain,
        }
        # #region agent log
        trace_pipeline_step(
            "chunking",
            "step7_dedupe",
            "Step7 去重完成",
            data={"doc_id": doc_id, "final_chunks": len(final), "stats": stats},
            artifact={
                "final_chunks": summarize_scored_chunks(final),
                "stats": stats,
            },
            doc_id=doc_id,
            hypothesis_id="H4",
        )
        # #endregion

        _logger.info(
            "Seven-step chunking doc=%s units=%d retrieval=%d parents=%d",
            doc_id,
            len(units),
            len(final),
            len(parents),
        )
        return PipelineResult(
            retrieval_chunks=final,
            parent_chunks=parents,
            repair_tasks=repair_tasks,
            stats=stats,
        )


def apply_seven_step_chunking(
    content: str,
    doc_id: str,
    cfg: ChunkPipelineConfig,
    metadata: Optional[Dict[str, Any]] = None,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> PipelineResult:
    pipeline = SevenStepChunkPipeline(cfg=cfg, embed_fn=embed_fn)
    return pipeline.run(content, doc_id, metadata)

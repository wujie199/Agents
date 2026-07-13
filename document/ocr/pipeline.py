# -*- coding: utf-8 -*-
"""通用 OCR 编排：预处理 → 识别 → QC → 加重重试。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from document.ocr.det_health import ensure_det_available
from document.ocr.document_ir import build_document_ir, normalize_page
from document.ocr.ocr_adapter import (
    OcrPipelineAdapter,
    PipelineConfig,
    _load_layout_cache,
    process_page,
    retry_failed_regions,
)
from document.ocr.preprocess import PagePreprocessor, resolve_preprocess_plan
from document.ocr.qc import (
    PageQcResult,
    evaluate_page_qc,
    failed_region_indices,
    needs_full_page_rerun,
    qc_rank,
)
from document.rag.shared.debug_trace import summarize_ocr_regions, trace_pipeline_step


class UniversalOcrPipeline:
    """与业务类型无关的文档 OCR 状态机。"""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.adapter = OcrPipelineAdapter(cfg)
        self._preprocessor: PagePreprocessor | None = None
        self._det_checked = False

    def _get_preprocessor(self) -> PagePreprocessor:
        if self._preprocessor is None:
            self._preprocessor = PagePreprocessor(
                model_root=self.cfg.model_root,
                device=self.cfg.device,
                enable_mkldnn=self.cfg.enable_mkldnn,
            )
        return self._preprocessor

    def _ensure_det_once(self, scratch_dir: Path) -> None:
        if self._det_checked:
            return
        self.cfg = ensure_det_available(
            self.adapter, self.cfg, scratch_dir=scratch_dir / "_probe"
        )
        self.adapter.cfg = self.cfg
        self._det_checked = True

    def _page_output_dirs(
        self,
        page_index: int,
        layout_out: Path | None,
        crops_dir: Path | None,
    ) -> tuple[Path | None, Path | None]:
        if layout_out is not None:
            d = layout_out / f"page_{page_index:04d}"
            d.mkdir(parents=True, exist_ok=True)
            layout_out = d
        if crops_dir is not None:
            d = crops_dir / f"page_{page_index:04d}"
            d.mkdir(parents=True, exist_ok=True)
            crops_dir = d
        return layout_out, crops_dir

    def process_single_page(
        self,
        *,
        image_path: Path,
        page_index: int,
        layout_out: Path | None,
        crops_dir: Path | None,
        scratch_dir: Path,
        original_image: Path | None = None,
    ) -> dict[str, Any]:
        cfg = self.cfg
        source_path = original_image or image_path
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_det_once(scratch_dir)
        page_layout, page_crops = self._page_output_dirs(page_index, layout_out, crops_dir)

        best_page: dict[str, Any] | None = None
        best_rank: tuple[int | float, ...] | None = None
        prev_qc: PageQcResult | None = None

        for attempt in range(cfg.max_attempts):
            suggest_heavy = bool(prev_qc and prev_qc.suggest_heavier_preprocess)
            suggest_table_only = bool(
                prev_qc
                and prev_qc.suggest_table_e2e
                and not prev_qc.suggest_heavier_preprocess
            )
            if suggest_heavy and cfg.preprocess_mode == "off":
                print(
                    "  提示: QC 建议加重预处理，但 preprocess=off，"
                    "将仅做区域级重试/表格 e2e"
                )
            plan = resolve_preprocess_plan(
                attempt,
                preprocess_mode=cfg.preprocess_mode,
                suggest_heavier_preprocess=suggest_heavy,
                suggest_table_e2e_only=suggest_table_only,
            )

            pre_scratch = scratch_dir / f"attempt_{attempt}"
            pre_result = self._get_preprocessor().run(
                image_path, pre_scratch, plan
            )
            work_image = pre_result.image_path
            ocr_scratch = pre_scratch / "ocr"
            # #region agent log
            trace_pipeline_step(
                "ocr",
                f"page_{page_index:04d}_attempt_{attempt}_preprocess",
                f"第 {page_index + 1} 页预处理 attempt {attempt + 1}",
                data={
                    "page_index": page_index,
                    "attempt": attempt,
                    "orientation_applied": pre_result.orientation_applied,
                    "unwarp_applied": pre_result.unwarp_applied,
                    "skipped": pre_result.skipped,
                    "work_image": str(work_image),
                },
                artifact={
                    "plan": {
                        "use_orientation": plan.use_orientation,
                        "use_unwarp": plan.use_unwarp,
                    },
                    "preprocess": {
                        "orientation_angle": pre_result.orientation_angle,
                        "orientation_applied": pre_result.orientation_applied,
                        "unwarp_applied": pre_result.unwarp_applied,
                        "skipped": pre_result.skipped,
                    },
                },
                hypothesis_id="H2",
            )
            # #endregion

            attempt_cfg = cfg
            if prev_qc is not None and prev_qc.suggest_table_e2e:
                attempt_cfg = replace(attempt_cfg, table_e2e=True)

            preprocess_changed = (
                pre_result.orientation_applied or pre_result.unwarp_applied
            )
            use_region_retry = False
            if (
                attempt > 0
                and best_page is not None
                and prev_qc is not None
                and prev_qc.status != "pass"
                and not needs_full_page_rerun(
                    prev_qc,
                    preprocess_mode=cfg.preprocess_mode,
                    preprocess_applied=preprocess_changed,
                )
            ):
                prev_cache_dir = scratch_dir / f"attempt_{attempt - 1}" / "ocr"
                retry_idx = failed_region_indices(prev_qc)
                if retry_idx and _load_layout_cache(prev_cache_dir, work_image):
                    use_region_retry = True

            if use_region_retry and best_page is not None and prev_qc is not None:
                page = retry_failed_regions(
                    adapter=self.adapter,
                    cfg=attempt_cfg,
                    work_image=work_image,
                    page_index=page_index,
                    merge_page=best_page,
                    retry_indices=failed_region_indices(prev_qc),
                    scratch_dir=ocr_scratch,
                    layout_scratch_dir=scratch_dir / f"attempt_{attempt - 1}" / "ocr",
                    crops_dir=page_crops,
                )
            else:
                page = process_page(
                    image_path=work_image,
                    page_index=page_index,
                    adapter=self.adapter,
                    cfg=attempt_cfg,
                    layout_out=page_layout,
                    crops_dir=page_crops,
                    scratch_dir=ocr_scratch,
                )

            page["source_image"] = str(source_path.resolve())
            page["processed_image"] = str(work_image.resolve())
            page["attempt"] = attempt
            page["preprocess"] = {
                "orientation_angle": pre_result.orientation_angle,
                "orientation_applied": pre_result.orientation_applied,
                "unwarp_applied": pre_result.unwarp_applied,
                "skipped": pre_result.skipped,
            }

            is_last = attempt >= cfg.max_attempts - 1
            qc = evaluate_page_qc(
                page,
                rec_score_threshold=cfg.rec_score_threshold,
                layout_score_threshold=cfg.layout_score_threshold,
                low_confidence_page_ratio=cfg.low_confidence_page_ratio,
                is_last_attempt=is_last,
            )
            page["qc"] = qc.to_dict()
            rank = qc_rank(qc)
            # #region agent log
            trace_pipeline_step(
                "ocr",
                f"page_{page_index:04d}_attempt_{attempt}_ocr_qc",
                f"第 {page_index + 1} 页 OCR+QC attempt {attempt + 1}",
                data={
                    "page_index": page_index,
                    "attempt": attempt,
                    "use_region_retry": use_region_retry,
                    "qc_status": qc.status,
                    "issue_count": qc.issue_count,
                    "regions": len(page.get("regions") or []),
                },
                artifact={
                    "page_index": page_index,
                    "attempt": attempt,
                    "preprocess": page.get("preprocess"),
                    "qc": qc.to_dict(),
                    "regions": summarize_ocr_regions(page.get("regions") or []),
                },
                hypothesis_id="H2",
            )
            # #endregion

            if best_page is None or rank < best_rank or rank == best_rank:  # type: ignore[operator]
                best_page = page
                best_rank = rank

            prev_qc = qc

            if qc.status == "pass":
                break

            if qc.status == "retry" and not is_last:
                hints = []
                if qc.suggest_heavier_preprocess and cfg.preprocess_mode != "off":
                    hints.append("加重预处理")
                elif qc.suggest_heavier_preprocess:
                    hints.append("区域重试")
                if qc.suggest_table_e2e:
                    hints.append("表格e2e")
                hint_str = "、".join(hints) if hints else "调整参数"
                print(
                    f"  第 {page_index + 1} 页 QC 未通过 (attempt {attempt + 1}, "
                    f"{qc.issue_count} 项)，{hint_str} 后重试…"
                )
                continue

        assert best_page is not None
        final_qc = best_page.get("qc") or {}
        final_status = final_qc.get("status", "fail")
        if final_status == "retry":
            final_qc = dict(final_qc)
            final_qc["status"] = "fail"
            best_page = dict(best_page)
            best_page["qc"] = final_qc
            final_status = "fail"
        if final_status != "pass":
            print(f"  第 {page_index + 1} 页 QC 最终状态: {final_status}")
        final_page = normalize_page(best_page)
        # #region agent log
        trace_pipeline_step(
            "ocr",
            f"page_{page_index:04d}_final",
            f"第 {page_index + 1} 页 OCR 最终结果",
            data={
                "page_index": page_index,
                "final_qc_status": final_status,
                "regions": len(final_page.get("regions") or []),
            },
            artifact=final_page,
            hypothesis_id="H2",
        )
        # #endregion
        return final_page

    def process_all_pages(
        self,
        image_paths: list[Path],
        *,
        source: Path,
        layout_out: Path | None,
        crops_dir: Path | None,
        scratch_root: Path,
    ) -> dict[str, Any]:
        pages: list[dict[str, Any]] = []
        for page_index, image_path in enumerate(image_paths):
            scratch = scratch_root / f"page_{page_index:04d}"
            pages.append(
                self.process_single_page(
                    image_path=image_path,
                    page_index=page_index,
                    layout_out=layout_out,
                    crops_dir=crops_dir,
                    scratch_dir=scratch,
                )
            )

        document = build_document_ir(
            source=source,
            pages=pages,
            pipeline_version=self.cfg.pipeline_version,
            config_summary=self.cfg.summary(),
        )
        # #region agent log
        trace_pipeline_step(
            "ocr",
            "all_pages_document_ir",
            "全页 OCR document_ir 完成",
            data={
                "source": str(source),
                "pages": len(pages),
                "regions_total": sum(len(p.get("regions") or []) for p in pages),
            },
            artifact={
                "config_summary": self.cfg.summary(),
                "page_summaries": [
                    {
                        "page_index": p.get("page_index"),
                        "qc_status": (p.get("qc") or {}).get("status"),
                        "region_count": len(p.get("regions") or []),
                    }
                    for p in pages
                ],
            },
            hypothesis_id="H2",
        )
        # #endregion
        return document

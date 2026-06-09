# -*- coding: utf-8 -*-
"""通用质量门禁：仅依赖分数与结构，与业务文档类型无关。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from document.ocr.labels import ASSET_LABELS, FORMULA_LABELS, TABLE_LABEL


@dataclass
class QcIssue:
    code: str
    message: str
    region_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.region_index is not None:
            d["region_index"] = self.region_index
        return d


@dataclass
class PageQcResult:
    status: str  # pass | retry | fail
    issues: list[QcIssue] = field(default_factory=list)
    low_confidence_ratio: float = 0.0
    suggest_heavier_preprocess: bool = False
    suggest_table_e2e: bool = False

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
            "issue_count": self.issue_count,
            "low_confidence_ratio": round(self.low_confidence_ratio, 4),
            "suggest_heavier_preprocess": self.suggest_heavier_preprocess,
            "suggest_table_e2e": self.suggest_table_e2e,
        }


def qc_rank(qc: PageQcResult) -> tuple[int | float, ...]:
    """数值越小表示质量越好，用于多轮尝试选最优页。"""
    if qc.status == "pass":
        return (0, 0, 0.0)
    return (1, qc.issue_count, qc.low_confidence_ratio)


def failed_region_indices(qc: PageQcResult) -> set[int]:
    return {i.region_index for i in qc.issues if i.region_index is not None}


def needs_full_page_rerun(
    qc: PageQcResult,
    *,
    preprocess_mode: str,
    preprocess_applied: bool,
) -> bool:
    """几何预处理变更或无法在区域级修复时需整页重跑。"""
    if preprocess_applied:
        return True
    if qc.suggest_heavier_preprocess and preprocess_mode != "off":
        return True
    return False


def _is_valid_table_html(html: str) -> bool:
    h = (html or "").strip().lower()
    return bool(h) and "<table" in h


def evaluate_page_qc(
    page: dict[str, Any],
    *,
    rec_score_threshold: float,
    layout_score_threshold: float,
    low_confidence_page_ratio: float,
    is_last_attempt: bool,
) -> PageQcResult:
    regions = page.get("regions") or []
    issues: list[QcIssue] = []
    low_conf = 0
    textual_count = 0
    layout_low = 0

    for r in regions:
        label = r.get("label", "")
        bi = r.get("box_index")

        if label in ASSET_LABELS:
            continue

        layout_score = r.get("layout_score")
        if layout_score is not None and float(layout_score) < layout_score_threshold:
            layout_low += 1
            issues.append(
                QcIssue(
                    code="low_layout_score",
                    message=f"版面置信度 {layout_score:.3f} < {layout_score_threshold}",
                    region_index=bi,
                )
            )

        if label == TABLE_LABEL:
            textual_count += 1
            html = ""
            if r.get("table_result"):
                html = r["table_result"].get("pred_html") or ""
            html = html or r.get("text") or ""
            if not _is_valid_table_html(html):
                issues.append(
                    QcIssue(
                        code="invalid_table",
                        message="表格 HTML 无效或为空",
                        region_index=bi,
                    )
                )
            continue

        if label in FORMULA_LABELS:
            textual_count += 1
            latex = r.get("text") or ""
            if not latex.strip():
                issues.append(
                    QcIssue(
                        code="empty_formula",
                        message="公式区域识别为空",
                        region_index=bi,
                    )
                )
            else:
                score = r.get("rec_score")
                if score is not None and float(score) < rec_score_threshold:
                    low_conf += 1
            continue

        textual_count += 1
        text = (r.get("text") or "").strip()
        if not text:
            issues.append(
                QcIssue(
                    code="empty_text_region",
                    message=f"文本区域 [{label}] 识别为空",
                    region_index=bi,
                )
            )
            continue

        score = r.get("rec_score")
        if score is not None:
            if float(score) < rec_score_threshold:
                low_conf += 1
                issues.append(
                    QcIssue(
                        code="low_rec_score",
                        message=f"识别置信度 {score:.3f} < {rec_score_threshold}",
                        region_index=bi,
                    )
                )

    ratio = (low_conf / textual_count) if textual_count else 0.0
    if textual_count and ratio >= low_confidence_page_ratio:
        issues.append(
            QcIssue(
                code="page_low_confidence_ratio",
                message=f"低置信区域占比 {ratio:.1%} >= {low_confidence_page_ratio:.1%}",
            )
        )

    suggest_preprocess = layout_low > 0 or any(
        i.code in ("empty_text_region", "page_low_confidence_ratio") for i in issues
    )
    suggest_table_e2e = any(i.code == "invalid_table" for i in issues)

    if not issues:
        return PageQcResult(status="pass", low_confidence_ratio=ratio)

    if is_last_attempt:
        return PageQcResult(
            status="fail",
            issues=issues,
            low_confidence_ratio=ratio,
            suggest_heavier_preprocess=suggest_preprocess,
            suggest_table_e2e=suggest_table_e2e,
        )

    return PageQcResult(
        status="retry",
        issues=issues,
        low_confidence_ratio=ratio,
        suggest_heavier_preprocess=suggest_preprocess,
        suggest_table_e2e=suggest_table_e2e,
    )

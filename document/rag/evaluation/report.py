# -*- coding: utf-8 -*-
"""Evaluation report writers: summary.json, details.jsonl, failures.jsonl, report.md/html."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from document.rag.evaluation.pipeline import PipelineRow


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _latency_stats(rows: list[PipelineRow]) -> dict[str, float]:
    ret = [r.retrieval_ms for r in rows if r.retrieval_ms > 0]
    gen = [r.generation_ms for r in rows if r.generation_ms > 0]
    def _avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 2) if vals else 0.0
    return {
        "retrieval_ms_mean": _avg(ret),
        "generation_ms_mean": _avg(gen),
        "total_ms_mean": _avg([r.retrieval_ms + r.generation_ms for r in rows]),
    }


def _failure_counts(rows: list[PipelineRow]) -> dict[str, int]:
    return {
        "pipeline_errors": sum(1 for r in rows if r.error),
        "retrieval_empty": sum(1 for r in rows if r.retrieval_empty),
        "retrieval_degraded": sum(1 for r in rows if r.retrieval_degraded),
    }


def build_summary(
    *,
    run_id: str,
    mode: str,
    profile: str,
    dataset_path: str,
    sample_count: int,
    rows: list[PipelineRow],
    metric_means: dict[str, float | None],
    config_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": _utc_now(),
        "mode": mode,
        "profile": profile,
        "dataset": dataset_path,
        "sample_count": sample_count,
        "metrics": metric_means,
        "failures": _failure_counts(rows),
        "latency": _latency_stats(rows),
        "config": config_meta,
    }


def build_failures(rows: list[PipelineRow], *, preview_chars: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.error or row.retrieval_empty or row.retrieval_degraded:
            detail = row.to_detail_dict(preview_chars=preview_chars)
            detail["failure_type"] = (
                "pipeline_error"
                if row.error
                else "retrieval_empty"
                if row.retrieval_empty
                else "retrieval_degraded"
            )
            out.append(detail)
    return out


def render_report_md(summary: dict[str, Any]) -> str:
    lines = [
        f"# RAG Evaluation Report — {summary.get('run_id')}",
        "",
        f"- **Created**: {summary.get('created_at')}",
        f"- **Mode**: {summary.get('mode')}",
        f"- **Profile**: {summary.get('profile')}",
        f"- **Dataset**: {summary.get('dataset')}",
        f"- **Samples**: {summary.get('sample_count')}",
        "",
        "## Metrics (mean)",
        "",
    ]
    metrics = summary.get("metrics") or {}
    if metrics:
        lines.append("| Metric | Score |")
        lines.append("| --- | --- |")
        for name, score in sorted(metrics.items()):
            val = "—" if score is None else f"{score:.4f}"
            lines.append(f"| {name} | {val} |")
    else:
        lines.append("_No RAGAS metrics computed._")

    lines.extend(["", "## Failures", ""])
    failures = summary.get("failures") or {}
    for key, count in failures.items():
        lines.append(f"- **{key}**: {count}")

    lines.extend(["", "## Latency (ms)", ""])
    latency = summary.get("latency") or {}
    for key, val in latency.items():
        lines.append(f"- **{key}**: {val}")

    return "\n".join(lines) + "\n"


def _plotly_chart_html(metric_means: dict[str, float | None]) -> str:
    labels = []
    values = []
    for name, score in sorted(metric_means.items()):
        if score is None:
            continue
        labels.append(name)
        values.append(score)
    if not labels:
        return "<p>No metric scores to chart.</p>"

    try:
        import plotly.graph_objects as go

        bar = go.Figure(
            data=[go.Bar(x=labels, y=values, marker_color="#4C78A8")],
        )
        bar.update_layout(
            title="Metric Scores (mean)",
            yaxis=dict(range=[0, 1]),
            margin=dict(l=40, r=20, t=50, b=80),
        )
        if len(labels) >= 3:
            radar = go.Figure(
                data=[
                    go.Scatterpolar(
                        r=values + [values[0]],
                        theta=labels + [labels[0]],
                        fill="toself",
                        name="scores",
                    )
                ],
                layout=go.Layout(
                    polar=dict(radialaxis=dict(range=[0, 1])),
                    title="Radar (mean metrics)",
                    margin=dict(l=40, r=40, t=50, b=40),
                ),
            )
            return bar.to_html(full_html=False, include_plotlyjs="cdn") + radar.to_html(
                full_html=False, include_plotlyjs=False
            )
        return bar.to_html(full_html=False, include_plotlyjs="cdn")
    except ImportError:
        bars = []
        for label, val in zip(labels, values):
            width = int(max(0, min(100, val * 100)))
            bars.append(
                f"<div><strong>{label}</strong> {val:.3f}"
                f"<div style='background:#eee;height:12px'>"
                f"<div style='width:{width}%;background:#4C78A8;height:12px'></div>"
                f"</div></div>"
            )
        return "".join(bars)


def render_report_html(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics") or {}
    chart = _plotly_chart_html(metrics)
    md_body = render_report_md(summary).replace("\n", "<br>\n")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>RAG Eval — {summary.get('run_id')}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; }}
    h1 {{ color: #1a1a1a; }}
    .chart {{ max-width: 900px; margin: 1.5rem 0; }}
    .meta {{ color: #555; }}
  </style>
</head>
<body>
  <h1>RAG Evaluation Report</h1>
  <p class="meta">Run: {summary.get('run_id')} · Mode: {summary.get('mode')} · Profile: {summary.get('profile')}</p>
  <div class="chart">{chart}</div>
  <hr/>
  <div>{md_body}</div>
</body>
</html>
"""


def write_run_reports(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    detail_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": str(output_dir / "summary.json"),
        "details": str(output_dir / "details.jsonl"),
        "failures": str(output_dir / "failures.jsonl"),
        "report_md": str(output_dir / "report.md"),
        "report_html": str(output_dir / "report.html"),
    }
    write_json(Path(paths["summary"]), summary)
    write_jsonl(Path(paths["details"]), detail_rows)
    write_jsonl(Path(paths["failures"]), failure_rows)
    Path(paths["report_md"]).write_text(render_report_md(summary), encoding="utf-8")
    Path(paths["report_html"]).write_text(render_report_html(summary), encoding="utf-8")
    return paths

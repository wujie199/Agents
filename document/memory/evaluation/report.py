# -*- coding: utf-8
"""Memory evaluation HTML/Markdown reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_memory_eval_reports(result: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "rows"}
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = result.get("rows") or []
    with (out / "details.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    failures = [r for r in rows if not r.get("passed")]
    if failures:
        with (out / "failures.jsonl").open("w", encoding="utf-8") as fh:
            for row in failures:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_markdown(summary, rows, out / "report.md")
    _write_html(summary, rows, out / "report.html")


def _write_markdown(summary: dict, rows: list, path: Path) -> None:
    lines = [
        "# Memory Evaluation Report",
        "",
        f"- **pass_rate**: {summary.get('pass_rate', 0):.2%}",
        f"- **total**: {summary.get('total', 0)}",
        f"- **passed**: {summary.get('passed', 0)}",
        "",
        "## By kind",
        "",
    ]
    for kind, stats in (summary.get("by_kind") or {}).items():
        lines.append(
            f"- `{kind}`: {stats.get('passed')}/{stats.get('count')} "
            f"(avg {stats.get('avg_score', 0):.2f})"
        )
    lines.extend(["", "## Rows", "", "| id | kind | pass | score |", "|---|---|---|---|"])
    for r in rows:
        lines.append(
            f"| {r.get('id')} | {r.get('kind')} | {r.get('passed')} | {r.get('score')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(summary: dict, rows: list, path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    row_html = "".join(
        f"<tr><td>{r.get('id')}</td><td>{r.get('kind')}</td>"
        f"<td>{'✅' if r.get('passed') else '❌'}</td>"
        f"<td>{r.get('score')}</td>"
        f"<td><pre>{(r.get('detail') or '')[:120]}</pre></td></tr>"
        for r in rows
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Memory Eval</title>
<style>body{{font-family:sans-serif;margin:2rem}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:6px}} pre{{white-space:pre-wrap;margin:0}}</style></head>
<body>
<h1>Memory Evaluation</h1>
<p>Generated {ts}</p>
<p>Pass rate: <strong>{summary.get('pass_rate', 0):.2%}</strong>
 ({summary.get('passed', 0)}/{summary.get('total', 0)})</p>
<table><tr><th>id</th><th>kind</th><th>pass</th><th>score</th><th>detail</th></tr>
{row_html}
</table></body></html>"""
    path.write_text(html, encoding="utf-8")

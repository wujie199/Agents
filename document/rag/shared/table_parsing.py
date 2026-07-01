"""表格文本提取：Markdown / HTML / CSV / JSON / Excel → 结构化行文本。"""

import csv
import json
import logging
import os
import re
from collections import Counter
from html.parser import HTMLParser
from typing import Any, List, Optional, Union

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import pandas as pd
except ImportError:
    pd = None

LOGGER = logging.getLogger(__name__)
_METRICS: Counter = Counter()


def _increment_metric(name: str, amount: int = 1) -> None:
    _METRICS[name] += amount


def _safe_call(fn, *args, fallback=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, ValueError, OSError):
        LOGGER.exception("Data cleaner failed inside %s", fn.__name__)
        _increment_metric(f"{fn.__name__}_errors")
        return fallback


# ── 判断辅助 ──────────────────────────────────────────────


def _looks_like_json(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    return s.startswith("{") or s.startswith("[")


def _is_markdown_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return "| " in lines[0] or lines[0].count("|") >= 2 and re.match(r"^[\s|:\-]+$", lines[1]) is not None


def _is_table_like_text(text: str) -> bool:
    if not text or text.count("\n") < 2:
        return False
    if "|" in text or "\t" in text:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    separators = sum(1 for line in lines if re.search(r"\s{2,}", line))
    return separators >= max(1, len(lines) // 2)


def _is_csv_like_text(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    has_comma = any("," in line for line in lines)
    has_tab = any("\t" in line for line in lines)
    return has_comma or has_tab


def _is_html_text(text: str) -> bool:
    lower = text.strip().lower()
    return "<table" in lower or "<html" in lower or "<thead" in lower or "<tbody" in lower


# ── 格式 → 行 ─────────────────────────────────────────────


def _markdown_table_to_rows(text: str) -> List[List[str]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        if re.match(r"^[\s|:\-]+$", line):
            continue
        cells = [cell.strip() for cell in re.split(r"\s*\|\s*", line.strip("|"))]
        if cells:
            rows.append(cells)
    return rows


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, ValueError, UnicodeDecodeError):
        LOGGER.exception("Failed to read table file: %s", path)
        _increment_metric("table_file_errors")
        return ""


def _json_table_to_rows(value: Any) -> List[List[str]]:
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        headers: List[str] = []
        seen = set()
        for item in value:
            for key in item.keys():
                if key not in seen:
                    seen.add(key)
                    headers.append(key)
        rows = [headers]
        for item in value:
            rows.append([str(item.get(k, "") or "") for k in headers])
        return rows
    if isinstance(value, dict) and value and all(isinstance(v, (list, tuple)) for v in value.values()):
        lengths = {len(v) for v in value.values()}
        if len(lengths) == 1:
            headers = list(value.keys())
            rows = [headers] + [list(values) for values in zip(*value.values())]
            return rows
    return []


def _html_table_to_rows(html: str) -> List[List[str]]:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return []
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        return rows

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: List[List[str]] = []
            self.current_row: List[str] = []
            self.current_data: List[str] = []
            self.in_cell = False

        def handle_starttag(self, tag, attrs):
            if tag in ("tr", "td", "th"):
                if tag == "tr":
                    self.current_row = []
                else:
                    self.current_data = []
                    self.in_cell = True

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self.current_row.append("".join(self.current_data).strip())
                self.in_cell = False
            elif tag == "tr":
                if self.current_row:
                    self.rows.append(self.current_row)

        def handle_data(self, data):
            if self.in_cell:
                self.current_data.append(data)

    parser = _Parser()
    parser.feed(html)
    return parser.rows


def _csv_text_to_rows(text: str, delimiter: str = ",", quotechar: str = '"') -> List[List[str]]:
    text = text.strip()
    if not text:
        return []
    reader = csv.reader(text.splitlines(), delimiter=delimiter, quotechar=quotechar)
    return [row for row in reader if any(cell.strip() for cell in row)]


def _rows_to_text(rows: List[List[Any]]) -> str:
    return "\n".join(" | ".join(str(cell).strip() for cell in row) for row in rows)


def _normalize_table_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_excel_text(source: Union[str, bytes], sheet_name: Optional[str] = None) -> str:
    if pd is None:
        raise RuntimeError("pandas is required to extract Excel tables")
    if isinstance(source, bytes):
        import io

        source = io.BytesIO(source)
    df = pd.read_excel(source, sheet_name=sheet_name)
    rows = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    return _rows_to_text(rows)


# ── 公共 API ──────────────────────────────────────────────


def extract_table_text(
    table: Any,
    delimiter: str = ",",
    quotechar: str = '"',
    sheet_name: Optional[str] = None,
) -> str:
    """
    支持 list[list]、list[dict]、dict、DataFrame、HTML/CSV/Markdown/JSON 文本，以及 Excel bytes/path。
    """
    if table is None:
        return ""

    if isinstance(table, bytes):
        try:
            table = table.decode("utf-8")
        except UnicodeDecodeError:
            table = table.decode("latin1", errors="ignore")

    if isinstance(table, str):
        s = table.strip()
        if not s:
            return ""

        if os.path.isfile(s):
            content = _read_text_file(s)
            ext = os.path.splitext(s.lower())[1]
            if ext in (".csv", ".txt"):
                return _rows_to_text(
                    _safe_call(_csv_text_to_rows, content, delimiter=delimiter, quotechar=quotechar, fallback=[])
                )
            if ext == ".tsv":
                return _rows_to_text(
                    _safe_call(_csv_text_to_rows, content, delimiter="\t", quotechar=quotechar, fallback=[])
                )
            if ext == ".md":
                rows = _safe_call(_markdown_table_to_rows, content, fallback=[])
                return _rows_to_text(rows) if rows else _normalize_table_text(content)
            if ext == ".json":
                return extract_table_text(content, delimiter=delimiter, quotechar=quotechar, sheet_name=sheet_name)
            if ext in (".xls", ".xlsx", ".xlsm", ".xlsb"):
                return _safe_call(_extract_excel_text, s, sheet_name=sheet_name, fallback="")
            s = content.strip()
            if not s:
                return ""

        if _looks_like_json(s):
            parsed = _safe_call(json.loads, s, fallback=None)
            if parsed is not None:
                rows = _json_table_to_rows(parsed)
                if rows:
                    return _rows_to_text(rows)
                return extract_table_text(parsed, delimiter=delimiter, quotechar=quotechar, sheet_name=sheet_name)

        if _is_html_text(s):
            rows = _safe_call(_html_table_to_rows, s, fallback=[])
            return _rows_to_text(rows) if rows else _normalize_table_text(s)

        if _is_markdown_table(s):
            rows = _safe_call(_markdown_table_to_rows, s, fallback=[])
            return _rows_to_text(rows) if rows else _normalize_table_text(s)

        if _is_csv_like_text(s):
            return _rows_to_text(
                _safe_call(_csv_text_to_rows, s, delimiter=delimiter, quotechar=quotechar, fallback=[])
            )

        return s

    if pd is not None and isinstance(table, pd.DataFrame):
        rows = [list(table.columns)] + table.fillna("").astype(str).values.tolist()
        return _rows_to_text(rows)

    if isinstance(table, dict):
        if all(isinstance(v, (list, tuple)) for v in table.values()):
            lengths = {len(v) for v in table.values()}
            if len(lengths) == 1:
                headers = list(table.keys())
                rows = [headers] + [list(values) for values in zip(*table.values())]
                return _rows_to_text(rows)
        if all(isinstance(v, dict) for v in table.values()):
            rows = []
            for key, value in table.items():
                row = [str(key)] + [f"{k}: {v}" for k, v in value.items()]
                rows.append(row)
            return _rows_to_text(rows)
        return _rows_to_text([[f"{k}: {v}" for k, v in table.items()]])

    if isinstance(table, (list, tuple)):
        if table and all(isinstance(row, dict) for row in table):
            headers = list(table[0].keys())
            rows = [headers] + [[row.get(header, "") for header in headers] for row in table]
            return _rows_to_text(rows)
        out_lines = []
        for row in table:
            if isinstance(row, dict):
                out_lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
            elif isinstance(row, (list, tuple)):
                out_lines.append(" | ".join(str(x) for x in row))
            else:
                out_lines.append(str(row))
        return "\n".join(out_lines)

    return str(table)

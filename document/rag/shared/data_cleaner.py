"""Recovered docstring."""
from datetime import datetime
from typing import List, Dict, Any, Generator, Iterable, Optional, Tuple, Callable, Union
import csv
from collections import Counter
from html.parser import HTMLParser
import json
import logging
import os
import re
import unicodedata
import hashlib
from hashlib import md5
import math

# Language detection: langdetect if installed, else heuristic
try:
    from langdetect import detect as _ld_detect
except Exception:
    _ld_detect = None

# (encoding fixed)
try:
    from dateutil.parser import parse as _date_parse
except ImportError:
    _date_parse = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import pandas as pd
except Exception:
    pd = None

# 常用 metadata 字段别名映射
_METADATA_FIELD_ALIASES = {
    "author": ["author", "作者", "writer", "written_by", "creator"],
    "title": ["title", "标题", "name"],
    "source": ["source", "来源", "origin"],
    "source_id": ["source_id", "sourceid", "source-id", "id", "doc_id", "document_id"],
    "created_at": ["created_at", "created", "creation_date", "date", "date_created", "创建时间"],
    "updated_at": ["updated_at", "updated", "modified", "modified_at", "date_modified", "更新时间"],
    "tags": ["tags", "tag", "keywords", "keyword", "labels", "标签"],
    "summary": ["summary", "description", "desc", "摘要"],
}

_ALIAS_LOOKUP = {alias: canonical for canonical, aliases in _METADATA_FIELD_ALIASES.items() for alias in aliases}

LOGGER = logging.getLogger(__name__)
_DATA_CLEANER_METRICS: Counter = Counter()


def _increment_metric(name: str, amount: int = 1) -> None:
    _DATA_CLEANER_METRICS[name] += amount


def get_data_cleaner_metrics() -> Dict[str, int]:
    return dict(_DATA_CLEANER_METRICS)


def reset_data_cleaner_metrics() -> None:
    _DATA_CLEANER_METRICS.clear()


def _normalize_metadata_key(key: Any) -> str:
    if key is None:
        return ''
    s = str(key).strip().lower().replace(' ', '_').replace('-', '_')
    s = re.sub(r"[^\w\u4e00-\u9fff_]", '', s, flags=re.UNICODE)
    return _ALIAS_LOOKUP.get(s, s)


def _normalize_date_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, datetime):
        return value.isoformat()
    s = str(value).strip()
    if not s:
        return ''
    if _date_parse:
        try:
            return _date_parse(s, fuzzy=True).isoformat()
        except Exception:
            pass
    # (encoding fixed)
    match = re.search(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', s)
    if match:
        y, m, d = match.groups()
        try:
            return datetime(int(y), int(m), int(d)).isoformat()
        except ValueError:
            return s
    return s


def clean_text(text: str) -> str:
    """Basic text cleanup."""
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", s)
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_table_like_text(text: str) -> bool:
    if not text or text.count('\n') < 2:
        return False
    if '|' in text or '\t' in text:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    separators = sum(1 for line in lines if re.search(r"\s{2,}", line))
    return separators >= max(1, len(lines) // 2)


def _normalize_table_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)


def _safe_call(fn: Callable[..., Any], *args: Any, fallback: Any = None, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:
        LOGGER.exception('Data cleaner failed inside %s', fn.__name__)
        _increment_metric(f'{fn.__name__}_errors')
        return fallback


def _looks_like_json(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    return s.startswith('{') or s.startswith('[')


def _is_markdown_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return '| ' in lines[0] or lines[0].count('|') >= 2 and re.match(r'^[\s\|:\-]+$', lines[1]) is not None


def _markdown_table_to_rows(text: str) -> List[List[str]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or '|' not in line:
            continue
        if re.match(r'^[\s\|:\-]+$', line):
            continue
        cells = [cell.strip() for cell in re.split(r'\s*\|\s*', line.strip('|'))]
        if cells:
            rows.append(cells)
    return rows


def _read_text_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        LOGGER.exception('Failed to read table file: %s', path)
        _increment_metric('table_file_errors')
        return ''


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
            rows.append([str(item.get(k, '') or '') for k in headers])
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
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        if not table:
            return []
        rows = []
        for tr in table.find_all('tr'):
            cells = [cell.get_text(strip=True) for cell in tr.find_all(['td', 'th'])]
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
            if tag in ('tr', 'td', 'th'):
                if tag == 'tr':
                    self.current_row = []
                else:
                    self.current_data = []
                    self.in_cell = True
        def handle_endtag(self, tag):
            if tag in ('td', 'th'):
                self.current_row.append(''.join(self.current_data).strip())
                self.in_cell = False
            elif tag == 'tr':
                if self.current_row:
                    self.rows.append(self.current_row)
        def handle_data(self, data):
            if self.in_cell:
                self.current_data.append(data)
    parser = _Parser()
    parser.feed(html)
    return parser.rows


def _csv_text_to_rows(text: str, delimiter: str = ',', quotechar: str = '"') -> List[List[str]]:
    text = text.strip()
    if not text:
        return []
    reader = csv.reader(text.splitlines(), delimiter=delimiter, quotechar=quotechar)
    return [row for row in reader if any(cell.strip() for cell in row)]


def _rows_to_text(rows: List[List[Any]]) -> str:
    return '\n'.join(' | '.join(str(cell).strip() for cell in row) for row in rows)


def _is_csv_like_text(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    has_comma = any(',' in line for line in lines)
    has_tab = any('\t' in line for line in lines)
    return has_comma or has_tab


def _is_html_text(text: str) -> bool:
    lower = text.strip().lower()
    return '<table' in lower or '<html' in lower or '<thead' in lower or '<tbody' in lower


def _extract_excel_text(source: Union[str, bytes], sheet_name: Optional[str] = None) -> str:
    if pd is None:
        raise RuntimeError('pandas is required to extract Excel tables')
    if isinstance(source, bytes):
        import io
        source = io.BytesIO(source)
    df = pd.read_excel(source, sheet_name=sheet_name)
    rows = [list(df.columns)] + df.fillna('').astype(str).values.tolist()
    return _rows_to_text(rows)


def _merge_ocr_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    merged = []
    for line in lines:
        if not line:
            continue
        if not merged:
            merged.append(line)
            continue
        prev = merged[-1]
        sentence_end = prev.endswith((".", "。", "！", "？", "!", "?"))
        if sentence_end or re.match(r"^[A-Z]", line) or re.match(r"^[\u4e00-\u9fff]", line):
            merged.append(line)
        else:
            merged[-1] = prev + ' ' + line
    return '\n'.join(merged)


def _clean_preserved_lines(text: str) -> str:
    lines = [clean_text(line) for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)


def _strip_ocr_noise(text: str) -> str:
    s = text.replace("\ufeff", "").replace("\ufffd", "")
    s = re.sub(r"-\s*\n\s*", '', s)
    s = re.sub(r"[\|\u2500-\u257f]+", ' ', s)
    s = re.sub(r"\.{2,}", '.', s)
    s = re.sub(r"\s{2,}", ' ', s)
    return s.strip()


def ocr_confidence_score(text: str) -> float:
    """Estimate OCR confidence from text noise heuristics."""
    if not text:
        return 0.0
    s = str(text)
    allowed = re.compile(
        r"[A-Za-z0-9\u4e00-\u9fff\s\.,!;:'\"\-\(\)\[\]\{\}\\/]+"
    )
    invalid = sum(1 for ch in s if not allowed.match(ch))
    ratio = invalid / len(s)
    bad_repeat = len(re.findall(r"(.)\1{4,}", s))
    score = max(0.0, 1.0 - min(0.8, ratio + bad_repeat * 0.05))
    return round(score, 3)


def postprocess_ocr(
    text: str,
    bbox: Dict[str, Any] = None,
    preserve_layout: bool = False,
    preserve_tables: bool = True,
) -> str:
    """Post-process OCR output: layout repair and table/noise handling."""
    if text is None:
        return ''
    s = str(text)
    table_like = preserve_tables and _is_table_like_text(s)
    s = _strip_ocr_noise(s)
    if table_like:
        return _clean_preserved_lines(_normalize_table_text(s))
    if preserve_layout:
        return _clean_preserved_lines(_merge_ocr_lines(s))
    return clean_text(s.replace('\n', ' '))


def detect_language(text: str) -> str:
    """Detect language of `text`.

    Returns an ISO 639-1 code like 'en', 'zh', or 'und' for undetermined.
    Uses `langdetect` when available; otherwise a simple CJK/ASCII heuristic.
    """
    if not text:
        return 'und'
    s = str(text)
    if _ld_detect:
        try:
            return _ld_detect(s)
        except Exception:
            pass
    if re.search(r"[\u4e00-\u9fff]", s):
        return 'zh'
    if re.search(r"[A-Za-z]", s):
        return 'en'
    return 'und'


def tokenize_text(text: str) -> List[str]:
    """Lightweight tokenizer: whitespace + punctuation split.

    For production you may replace this with `jieba`, `spacy`, or a BPE tokenizer.
    """
    if not text:
        return []
    s = re.sub(r"\s+", " ", str(text)).strip()
    parts = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", s)
    return parts


def extract_table_text(
    table: Any,
    delimiter: str = ',',
    quotechar: str = '"',
    sheet_name: Optional[str] = None,
) -> str:
    """
    æ¯æä¼ å¥ list[list]ãlist[dict]ãdictãDataFrameãHTML/CSV/Markdown/JSON ææ¬ï¼ä»¥å?Excel bytes/pathã    """
    if table is None:
        return ""

    if isinstance(table, bytes):
        try:
            table = table.decode('utf-8')
        except UnicodeDecodeError:
            table = table.decode('latin1', errors='ignore')

    if isinstance(table, str):
        s = table.strip()
        if not s:
            return ''

        if os.path.isfile(s):
            content = _read_text_file(s)
            ext = os.path.splitext(s.lower())[1]
            if ext in ('.csv', '.txt'):
                return _rows_to_text(_safe_call(_csv_text_to_rows, content, delimiter=delimiter, quotechar=quotechar, fallback=[]))
            if ext == '.tsv':
                return _rows_to_text(_safe_call(_csv_text_to_rows, content, delimiter='\t', quotechar=quotechar, fallback=[]))
            if ext == '.md':
                rows = _safe_call(_markdown_table_to_rows, content, fallback=[])
                return _rows_to_text(rows) if rows else _normalize_table_text(content)
            if ext == '.json':
                return extract_table_text(content, delimiter=delimiter, quotechar=quotechar, sheet_name=sheet_name)
            if ext in ('.xls', '.xlsx', '.xlsm', '.xlsb'):
                return _safe_call(_extract_excel_text, s, sheet_name=sheet_name, fallback='')
            s = content.strip()
            if not s:
                return ''

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
            return _rows_to_text(_safe_call(_csv_text_to_rows, s, delimiter=delimiter, quotechar=quotechar, fallback=[]))

        return s

    if pd is not None and isinstance(table, pd.DataFrame):
        rows = [list(table.columns)] + table.fillna('').astype(str).values.tolist()
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
            rows = [headers] + [[row.get(header, '') for header in headers] for row in table]
            return _rows_to_text(rows)
        out_lines = []
        for row in table:
            if isinstance(row, dict):
                out_lines.append(' | '.join(f"{k}: {v}" for k, v in row.items()))
            elif isinstance(row, (list, tuple)):
                out_lines.append(' | '.join(str(x) for x in row))
            else:
                out_lines.append(str(row))
        return '\n'.join(out_lines)

    return str(table)


def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metadata keys and values."""
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        key = _normalize_metadata_key(k)
        if not key:
            continue
        if key in ('created_at', 'updated_at', 'date'):
            out[key] = _normalize_date_value(v)
            continue
        if key == 'tags':
            if isinstance(v, str):
                tags = [t.strip() for t in re.split(r'[;,\|]+', v) if t.strip()]
                out[key] = tags
                continue
            if isinstance(v, (list, tuple, set)):
                out[key] = [str(t).strip() for t in v if t is not None]
                continue
        if isinstance(v, str):
            out[key] = clean_text(v)
        else:
            out[key] = v
    return out


def _chunk_hash(chunk_text: str) -> str:
    return md5(chunk_text.encode('utf-8')).hexdigest()


def _hash_shingle(shingle: str, seed: int) -> int:
    h = hashlib.sha256(f'{seed}:{shingle}'.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'big', signed=False)


def _shingle_set(text: str, k: int = 3) -> set:
    tokens = tokenize_text(text)
    if len(tokens) < k:
        return set(tokens)
    return set(' '.join(tokens[i:i + k]) for i in range(len(tokens) - k + 1))


def _minhash_signature(shingles: set, num_hashes: int = 64) -> Tuple[int, ...]:
    if not shingles:
        return tuple([0] * num_hashes)
    signature = []
    for seed in range(num_hashes):
        min_hash = min(_hash_shingle(shingle, seed) for shingle in shingles)
        signature.append(min_hash)
    return tuple(signature)


def _minhash_similarity(sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def _vector_norm(vector: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = _vector_norm(a) * _vector_norm(b)
    return dot / norm if norm else 0.0


def semantic_dedupe(
    chunks: Iterable[Dict[str, Any]],
    key_field: str = 'chunk_text',
    threshold: float = 0.85,
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
    num_hashes: int = 64,
    min_shingle_size: int = 3,
) -> List[Dict[str, Any]]:
    """Semantic dedupe via embeddings or MinHash."""
    seen_signatures: List[Tuple[int, ...]] = []
    seen_embeddings: List[List[float]] = []
    out: List[Dict[str, Any]] = []

    for item in chunks:
        if not isinstance(item, dict):
            continue
        text = str(item.get(key_field, '') or '')
        if embedding_fn is not None:
            emb = embedding_fn(text)
            is_duplicate = any(_cosine_similarity(emb, existing) >= threshold for existing in seen_embeddings)
            if not is_duplicate:
                seen_embeddings.append(emb)
                out.append(item)
        else:
            shingles = _shingle_set(text, k=min_shingle_size)
            sig = _minhash_signature(shingles, num_hashes=num_hashes)
            is_duplicate = any(_minhash_similarity(sig, existing) >= threshold for existing in seen_signatures)
            if not is_duplicate:
                seen_signatures.append(sig)
                out.append(item)
    return out


def dedupe_chunks(chunks: Iterable[Dict[str, Any]], key_fields: Tuple[str, ...] = ('chunk_text',)) -> List[Dict[str, Any]]:
    """Dedupe chunk dicts by key fields."""
    seen = set()
    out = []
    for item in chunks:
        key_vals = tuple(item.get(k, '') for k in key_fields)
        key_text = '||'.join(map(str, key_vals))
        h = _chunk_hash(key_text)
        if h in seen:
            continue
        seen.add(h)
        out.append(item)
    return out


def _safe_clean_record(rec: Any) -> Optional[Dict[str, Any]]:
    try:
        if not isinstance(rec, dict):
            raise ValueError('record must be a dict')
        item = _clean_record(rec)
        _increment_metric('records_cleaned')
        return item
    except Exception:
        LOGGER.exception('Failed to clean record: %r', rec)
        _increment_metric('record_errors')
        return None


def _clean_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    raw = rec.get('raw_text') or rec.get('text') or ''
    text = clean_text(raw)
    metadata = normalize_metadata(rec.get('meta') or {})
    lang = detect_language(text)
    tokens = tokenize_text(text)
    rid = _chunk_hash((rec.get('source_id', '') + '||' + text)[:1024])
    return {
        'id': rid,
        'chunk_text': text,
        'metadata': metadata,
        'source': rec.get('source'),
        'lang': lang,
        'tokens': tokens,
        'token_count': len(tokens),
    }


def batch_clean(records: Iterable[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """Batch clean records (compat API)."""
    for rec in records:
        item = _safe_clean_record(rec)
        if item is not None:
            yield item


def batch_clean_parallel(records: Iterable[Dict[str, Any]], workers: int = 4, use_process: bool = False, chunk_size: int = 64) -> Generator[Dict[str, Any], None, None]:
    """Parallel batch clean."""
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    from itertools import islice

    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    it = iter(records)

    with Executor(max_workers=workers) as ex:
        futures = []

        while True:
            chunk = list(islice(it, chunk_size))
            if not chunk:
                break
            for rec in chunk:
                futures.append(ex.submit(_safe_clean_record, rec))
            if len(futures) >= workers * 4:
                for fut in futures:
                    try:
                        item = fut.result()
                    except Exception:
                        LOGGER.exception('Parallel worker failed')
                        _increment_metric('parallel_errors')
                        continue
                    if item is not None:
                        yield item
                futures = []

        for fut in futures:
            try:
                item = fut.result()
            except Exception:
                LOGGER.exception('Parallel worker failed')
                _increment_metric('parallel_errors')
                continue
            if item is not None:
                yield item


__all__ = [
    'clean_text',
    'postprocess_ocr',
    'ocr_confidence_score',
    'extract_table_text',
    'normalize_metadata',
    'dedupe_chunks',
    'semantic_dedupe',
    'batch_clean',
    'batch_clean_parallel',
    'detect_language',
    'tokenize_text',
    'get_data_cleaner_metrics',
    'reset_data_cleaner_metrics',
]

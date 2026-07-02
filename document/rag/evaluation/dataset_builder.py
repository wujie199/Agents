# -*- coding: utf-8 -*-
"""从 PDF/TXT 生成 RAG 评测黄金集，reference 与 FaqChunker 建库块对齐。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.ports.chunker import Chunk
from document.rag.evaluation.dataset import load_eval_dataset
from document.rag.evaluation.ir_metrics import contexts_match
from document.rag.evaluation.text_norm import normalize_match_text

DocKind = Literal["faq_md", "tips", "pdf"]

_JUNK_CHUNK_RE = re.compile(r"^\d{1,3}\.\s*条\s*$")
_FAQ_MD_RE = re.compile(
    r"^(\d{1,3})\.\s*\*\*(.+?)\*\*\s*\n-\s*(.+?)(?=\n\d{1,3}\.\s*\*\*|\n###|\Z)",
    re.MULTILINE | re.DOTALL,
)
_HEADER3_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_HEADER2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_NUMBERED_LINE_RE = re.compile(r"^(\d{1,3})\.\s*(.+)$")
_PDF_QA_RE = re.compile(
    r"^(\d{1,3})\.\s*\*\*(.+?)\*\*\s*\n-\s*(.+?)(?=\n\d{1,3}\.\s*\*\*|\n##|\n###|\Z)",
    re.MULTILINE | re.DOTALL,
)
_PDF_HEADER2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_PDF_HEADER3_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_PDF_Q_POS_RE = re.compile(r"^(\d{1,3})\.\s*\*\*", re.MULTILINE)

# 清洗后 chunk 常见残缺片段（跨条合并导致）
_FRAGMENT_HEAD_RE = re.compile(
    r"^(分钟|转分钟|小时|㎡|m|Pa|ml|L|GWiFi|APP|WiFi)\b",
    re.IGNORECASE,
)

_SECTION_TAG_MAP = {
    "基础使用类": "usage",
    "清洁效果类": "cleaning_effect",
    "耗材与维护类": "consumables",
    "故障与售后类": "troubleshoot",
    "APP与智能功能类": "app",
    "通用基础维护": "daily_care",
    "扫地机器人专属维护": "vacuum_care",
    "扫拖一体拖地专属维护": "mop_care",
    "耗材专项维护": "consumables",
    "环境适配维护": "environment",
    "长期存放维护": "storage",
    "故障预防维护": "prevention",
}

_FILE_META: dict[str, dict[str, str]] = {
    "扫地机器人100问2.txt": {"slug": "faq2", "kind": "faq_md", "domain": "faq"},
    "选购指南.txt": {"slug": "buy", "kind": "tips", "domain": "purchase"},
    "维护保养.txt": {"slug": "maint", "kind": "tips", "domain": "maintenance"},
}


@dataclass
class GoldenRow:
    id: str
    question: str
    ground_truth: str
    reference_contexts: list[str]
    tenant_id: str
    tags: list[str]
    source_file: str
    chunk_id: str | None = None


@dataclass
class NumberedLine:
    number: int
    text: str
    section: str | None


def _normalize_compact(text: str) -> str:
    return normalize_match_text(text)


def _clean_question(text: str) -> str:
    q = re.sub(r"\*+", "", text).strip()
    q = re.sub(r"\s+", " ", q)
    q = q.lstrip("？? ").strip()
    if q and not q.endswith(("？", "?")):
        q += "？"
    return q


def _maintenance_question(text: str) -> str:
    """问句嵌入 chunk 首句关键词，提升向量/BM25 对齐。"""
    text = text.strip().rstrip("。")
    if "，" in text:
        lead, rest = text.split("，", 1)
        hint = rest.split("，")[0].strip()[:28]
        if len(hint) >= 4:
            return _clean_question(f"扫地机器人{lead}，{hint}如何做")
        return _clean_question(f"扫地机器人{lead}如何做")
    snippet = text[:32].rstrip("，。； ")
    return _clean_question(f"扫地机器人{snippet}如何做")


def _clean_answer(text: str) -> str:
    ans = re.sub(r"\*+", "", text.strip())
    ans = re.sub(r"\s*\n\s*", " ", ans)
    ans = ans.lstrip("？? ").strip()
    return ans.strip()


def _section_tag(section: str | None) -> str | None:
    if not section:
        return None
    if section in _SECTION_TAG_MAP:
        return _SECTION_TAG_MAP[section]
    return re.sub(r"\s+", "_", section)[:24]


_PURCHASE_TOPIC_OVERRIDES: dict[str, str] = {
    "选购核心": "选购扫地机器人时，应该优先明确哪些使用场景和需求",
    "品牌选择": "选购扫地机器人时，品牌应该怎么选",
    "价格区间": "选购扫地机器人时，不同预算可以买到什么配置",
    "促销时机": "选购扫地机器人时，什么时候入手比较划算",
    "实测体验": "选购扫地机器人时，现场试用应该关注哪些方面",
    "口碑参考": "选购扫地机器人时，用户评价应该怎么看",
    "翻新机鉴别": "选购扫地机器人时，如何鉴别翻新机",
    "宠物专属": "选购扫地机器人时，养宠家庭应该关注什么",
    "操作难度": "选购扫地机器人时，操作难度应该如何考量",
    "售后政策": "选购扫地机器人时，售后政策应该怎么看",
    "配件成本": "选购扫地机器人时，长期使用成本应该怎么估算",
    "包装保护": "选购扫地机器人时，开箱验货应该检查什么",
}


def _purchase_question(topic: str) -> str:
    topic = topic.strip().rstrip("：:")
    if topic in _PURCHASE_TOPIC_OVERRIDES:
        return _clean_question(_PURCHASE_TOPIC_OVERRIDES[topic])

    if topic.endswith("选择"):
        base = topic[:-2]
        return _clean_question(f"选购扫地机器人时，{base}应该怎么选")
    if topic.endswith("参数"):
        return _clean_question(f"选购扫地机器人时，{topic}应该怎么看")
    if topic.endswith(("能力", "技术", "算法")):
        return _clean_question(f"选购扫地机器人时，{topic}怎么选")
    if topic.endswith("功能"):
        return _clean_question(f"选购扫地机器人时，{topic}应该关注哪些要点")
    if topic.endswith(("容量", "时长", "功率", "面积")):
        return _clean_question(f"选购扫地机器人时，{topic}选多少合适")
    if topic.endswith(("设计", "配置", "系统", "类型", "材质", "接口")):
        return _clean_question(f"选购扫地机器人时，{topic}应该注意什么")
    if topic.endswith(("适配", "控制", "调节", "处理", "联动", "支持")):
        return _clean_question(f"选购扫地机器人时，{topic}怎么考虑")
    if topic.endswith(("体验", "参考", "鉴别", "时机", "区间", "政策", "成本", "难度")):
        return _clean_question(f"选购扫地机器人时，{topic}应该关注什么")

    return _clean_question(f"选购扫地机器人时，{topic}怎么选")


def _is_fragment_line(text: str) -> bool:
    text = text.strip()
    if len(text) < 12:
        return True
    if _FRAGMENT_HEAD_RE.match(text):
        return True
    if re.match(r"^\d+\s*[㎡m]", text):
        return True
    return False


def _chunk_matches_line(chunk_content: str, line_text: str) -> bool:
    return contexts_match(line_text, chunk_content, min_overlap_chars=16)


def _find_faq_chunk(
    chunks: list[Chunk],
    number: int,
    question: str,
    answer: str,
) -> str | None:
    q_key = _normalize_compact(question)[:30]
    a_key = _normalize_compact(answer)[:30]
    prefix = f"{number}."
    best: str | None = None
    best_score = 0
    for c in chunks:
        content = c.content.strip()
        if "-？" in content or "-?" in content:
            continue
        cn = _normalize_compact(content)
        if q_key not in cn:
            continue
        score = 2
        if a_key[:20] in cn:
            score += 3
        if content.startswith(prefix):
            score += 2
        if score > best_score:
            best_score = score
            best = content
    return best if best_score >= 5 else None


def _find_chunk_for_line(chunks: list[Chunk], number: int, line_text: str) -> str | None:
    num_s = str(number)
    by_num = [
        c
        for c in chunks
        if str(c.metadata.get("faq_number") or "").lstrip("0") == num_s.lstrip("0")
        or str(c.metadata.get("faq_number") or "") == num_s
    ]
    for c in by_num:
        if _chunk_matches_line(c.content, line_text):
            return c.content.strip()
    for c in chunks:
        if _chunk_matches_line(c.content, line_text):
            return c.content.strip()
    if len(by_num) == 1 and not _is_fragment_line(by_num[0].content.split("\n")[0].split(".", 1)[-1]):
        return by_num[0].content.strip()
    return None


def parse_numbered_lines(txt_path: Path) -> list[NumberedLine]:
    raw = txt_path.read_text(encoding="utf-8")
    section: str | None = None
    items: list[NumberedLine] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        h2 = _HEADER2_RE.match(stripped)
        if h2:
            section = h2.group(1).strip()
            continue
        h3 = _HEADER3_RE.match(stripped)
        if h3:
            section = h3.group(1).strip()
            continue
        if stripped.startswith("#"):
            continue
        m = _NUMBERED_LINE_RE.match(stripped)
        if m:
            items.append(
                NumberedLine(
                    number=int(m.group(1)),
                    text=m.group(2).strip(),
                    section=section,
                )
            )
    return items


def extract_tip_rows_from_raw(
    txt_path: Path,
    chunks: list[Chunk],
    *,
    slug: str,
    domain: str,
) -> list[GoldenRow]:
    rows: list[GoldenRow] = []
    for seq, item in enumerate(parse_numbered_lines(txt_path), start=1):
        text = item.text.rstrip("。").strip()
        if _is_fragment_line(text):
            continue

        if "：" in text:
            topic, body = text.split("：", 1)
            topic, body = topic.strip(), body.strip()
            if domain == "purchase":
                question = _purchase_question(topic)
            else:
                question = _maintenance_question(body or text)
            ground_truth = _clean_answer(body or text)
        else:
            question = _maintenance_question(text)
            ground_truth = _clean_answer(text)

        if len(ground_truth) < 8:
            continue

        ref = _find_chunk_for_line(chunks, item.number, text)
        if ref is None or not _chunk_matches_line(ref, text):
            ref = f"{item.number}. {text}"

        tags = ["eval", domain, slug]
        st = _section_tag(item.section)
        if st:
            tags.append(st)

        rows.append(
            GoldenRow(
                id=f"{slug}-{seq:03d}",
                question=question,
                ground_truth=ground_truth,
                reference_contexts=[ref],
                tenant_id="default",
                tags=tags,
                source_file=txt_path.name,
            )
        )
    return rows


def extract_faq_md_rows(txt_path: Path, chunks: list[Chunk] | None = None) -> list[GoldenRow]:
    """FAQ Markdown txt（**问题** + - 答案）。"""
    raw = txt_path.read_text(encoding="utf-8")
    meta = _FILE_META.get(txt_path.name, {"slug": txt_path.stem, "domain": "faq"})
    slug = meta["slug"]
    domain = meta.get("domain", "faq")
    chunk_list = chunks or []
    rows: list[GoldenRow] = []
    seen: set[int] = set()

    for num_s, q_raw, a_raw in _FAQ_MD_RE.findall(raw):
        num = int(num_s)
        if num in seen:
            continue
        seen.add(num)

        pos = raw.find(f"{num_s}. **")
        section: str | None = None
        if pos >= 0:
            prefix = raw[:pos]
            for m in _HEADER3_RE.finditer(prefix):
                section = m.group(1).strip()

        question = _clean_question(q_raw)
        answer = _clean_answer(a_raw)
        if len(answer) < 4:
            continue

        line_stub = f"{q_raw.strip()} {a_raw.strip()}"
        ref = _find_faq_chunk(chunk_list, num, question, answer)
        if ref is None:
            ref = _find_chunk_for_line(chunk_list, num, line_stub)
        if ref is None or "-？" in ref or "-?" in ref:
            ref = f"{num_s}. {question.rstrip('？?')}\n{answer}"

        tags = ["eval", domain, slug]
        st = _section_tag(section)
        if st:
            tags.append(st)

        rows.append(
            GoldenRow(
                id=f"{slug}-{num:03d}",
                question=question,
                ground_truth=answer,
                reference_contexts=[ref],
                tenant_id="default",
                tags=tags,
                source_file=txt_path.name,
            )
        )
    rows.sort(key=lambda r: r.id)
    return rows


def build_chunks_from_txt(
    txt_path: Path,
    *,
    config_dir: str = "config",
    profile: str = "faq",
    doc_id: str | None = None,
) -> list[Chunk]:
    """与离线建库相同：ingest → clean → FaqChunker 切块。"""
    from document.build_rag_index import step2_ingest_file, step3_clean_text
    from document.rag.application.indexing.chunker import (
        create_chunker,
        parse_chunk_strategy,
    )
    from document.rag.bootstrap.offline import (
        build_offline_ingest_port,
        load_offline_config,
    )
    from document.rag.config import resolve_rag_pipeline_config_path
    from document.rag.application.indexing.index_manifest import doc_id_from_file_md5, file_md5_hex

    resolved = resolve_rag_pipeline_config_path(config_dir=config_dir, profile=profile)
    cfg = load_offline_config(config_dir, config_path=resolved)
    ingest_port = build_offline_ingest_port(cfg)
    fid = doc_id or doc_id_from_file_md5(file_md5_hex(txt_path))
    tenant_id = "default"

    ingest = step2_ingest_file(
        txt_path, fid, tenant_id, cfg, ingest_port, file_md5=file_md5_hex(txt_path)
    )
    ingest = step3_clean_text(ingest, txt_path, cfg)
    if not (ingest.content or "").strip():
        return []

    metadata = {
        "tenant_id": tenant_id,
        "doc_id": fid,
        "source_path": str(txt_path.resolve()),
    }
    chunker = create_chunker(
        parse_chunk_strategy(cfg.chunk_strategy),
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    return chunker.chunk(ingest.content or "", fid, metadata)


def build_samples_from_txt(
    txt_path: Path,
    *,
    align_index: bool = True,
    config_dir: str = "config",
    profile: str = "faq",
) -> list[GoldenRow]:
    meta = _FILE_META.get(
        txt_path.name,
        {"slug": txt_path.stem[:12], "kind": "tips", "domain": "faq"},
    )
    slug = meta["slug"]
    domain = meta.get("domain", "faq")
    kind = meta.get("kind", "tips")

    chunks: list[Chunk] = []
    if align_index:
        chunks = build_chunks_from_txt(
            txt_path, config_dir=config_dir, profile=profile
        )

    if kind == "faq_md":
        return extract_faq_md_rows(txt_path, chunks)

    if kind == "tips":
        return extract_tip_rows_from_raw(
            txt_path, chunks, slug=slug, domain=domain
        )

    return []


def extract_pdf_rows(pdf_path: Path) -> list[GoldenRow]:
    from pypdf import PdfReader

    raw = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages
    )
    rows: list[GoldenRow] = []
    for num_s, q_raw, a_raw in _PDF_QA_RE.findall(raw):
        pos_m = _PDF_Q_POS_RE.search(raw, raw.find(f"{num_s}. **"))
        pos = pos_m.start() if pos_m else 0
        _cat, section = _headers_before(
            raw, pos, header2_re=_PDF_HEADER2_RE, header3_re=_PDF_HEADER3_RE
        )
        question = _clean_question(q_raw)
        answer = _clean_answer(a_raw)
        ref = f"{num_s}. {question.rstrip('？?')}\n{answer}"
        tags = ["eval", "vacuum", "pdf"]
        st = _section_tag(section)
        if st:
            tags.append(st)
        rows.append(
            GoldenRow(
                id=f"vacuum-{int(num_s):03d}",
                question=question,
                ground_truth=answer,
                reference_contexts=[ref],
                tenant_id="default",
                tags=tags,
                source_file=pdf_path.name,
            )
        )
    return rows


def _headers_before(
    text: str,
    pos: int,
    *,
    header2_re: re.Pattern[str],
    header3_re: re.Pattern[str],
) -> tuple[str | None, str | None]:
    prefix = text[:pos]
    category: str | None = None
    section: str | None = None
    cat_matches = list(header2_re.finditer(prefix))
    section_prefix = prefix
    if cat_matches:
        category = cat_matches[-1].group(1).strip()
        section_prefix = prefix[cat_matches[-1].start() :]
    for m in header3_re.finditer(section_prefix):
        section = m.group(1).strip()
    return category, section


def rows_to_dicts(rows: list[GoldenRow]) -> list[dict[str, Any]]:
    return [
        {
            "id": r.id,
            "question": r.question,
            "ground_truth": r.ground_truth,
            "reference_contexts": r.reference_contexts,
            "tenant_id": r.tenant_id,
            "tags": r.tags,
        }
        for r in rows
    ]


def write_jsonl(samples: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_merged_from_txt_dir(
    txt_dir: Path,
    output_path: Path,
    *,
    glob: str = "*.txt",
    tenant_id: str = "default",
    align_index: bool = True,
    config_dir: str = "config",
    profile: str = "faq",
) -> dict[str, Any]:
    paths = sorted(txt_dir.glob(glob))
    if not paths:
        raise FileNotFoundError(f"no txt files in {txt_dir} (glob={glob})")

    all_rows: list[GoldenRow] = []
    per_file: dict[str, int] = {}
    for path in paths:
        rows = build_samples_from_txt(
            path,
            align_index=align_index,
            config_dir=config_dir,
            profile=profile,
        )
        for row in rows:
            row.tenant_id = tenant_id
        per_file[path.name] = len(rows)
        all_rows.extend(rows)

    seen: set[str] = set()
    unique: list[GoldenRow] = []
    for row in all_rows:
        rid = row.id
        if rid in seen:
            rid = f"{rid}-dup{len(unique)}"
        seen.add(rid)
        unique.append(
            GoldenRow(
                id=rid,
                question=row.question,
                ground_truth=row.ground_truth,
                reference_contexts=row.reference_contexts,
                tenant_id=row.tenant_id,
                tags=row.tags,
                source_file=row.source_file,
                chunk_id=row.chunk_id,
            )
        )

    samples = rows_to_dicts(unique)
    write_jsonl(samples, output_path)
    load_eval_dataset(output_path)
    return {
        "output": str(output_path),
        "count": len(samples),
        "files": per_file,
        "id_range": (samples[0]["id"], samples[-1]["id"]) if samples else (None, None),
    }


def build_pdf_dataset(
    pdf_path: Path,
    output_path: Path,
    *,
    tenant_id: str = "default",
    target_count: int | None = 100,
) -> dict[str, Any]:
    rows = extract_pdf_rows(pdf_path)
    if target_count is not None and len(rows) < target_count:
        raise RuntimeError(
            f"expected>={target_count} rows from {pdf_path}, got {len(rows)}"
        )
    if target_count is not None:
        rows = rows[:target_count]
    for row in rows:
        row.tenant_id = tenant_id
    samples = rows_to_dicts(rows)
    write_jsonl(samples, output_path)
    load_eval_dataset(output_path)
    return {
        "pdf": str(pdf_path),
        "output": str(output_path),
        "count": len(samples),
        "id_range": (samples[0]["id"], samples[-1]["id"]) if samples else (None, None),
    }

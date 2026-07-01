"""基于 YAML 规则与关键词的文档级 metadata 打标。"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import yaml

from core.ports.ingest import IngestResult
from document.rag.shared.data_cleaner import normalize_metadata

_log = logging.getLogger("document.rag.adapters.metadata.rule_keyword")

MatchMode = Literal["any", "all"]


@dataclass
class TaggingRule:
    name: str
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    match: MatchMode = "any"


def load_tagging_rules(path: str) -> tuple[List[TaggingRule], Dict[str, List[str]]]:
    """加载 rules 与 extension_tags。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"metadata 规则文件不存在: {path}")

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    rules: List[TaggingRule] = []
    for item in raw.get("rules") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        rules.append(
            TaggingRule(
                name=str(item["name"]),
                tags=[str(t) for t in (item.get("tags") or [])],
                keywords=[str(k) for k in (item.get("keywords") or []) if k],
                match=str(item.get("match", "any")).lower(),  # type: ignore[arg-type]
            )
        )

    ext_tags: Dict[str, List[str]] = {}
    for ext, tags in (raw.get("extension_tags") or {}).items():
        ext_tags[str(ext).lower().lstrip(".")] = [str(t) for t in (tags or [])]

    return rules, ext_tags


def _keyword_in_text(keyword: str, text: str) -> bool:
    if not keyword or not text:
        return False
    if keyword.isascii():
        return keyword.lower() in text.lower()
    return keyword in text


def _rule_matches(rule: TaggingRule, text: str) -> bool:
    if not rule.keywords:
        return False
    hits = [_keyword_in_text(kw, text) for kw in rule.keywords]
    if rule.match == "all":
        return all(hits)
    return any(hits)


class RuleKeywordMetadataEnricher:
    """规则/关键词打标：写入 metadata.tags、metadata.categories、metadata.matched_rules。"""

    def __init__(
        self,
        rules: Optional[List[TaggingRule]] = None,
        extension_tags: Optional[Dict[str, List[str]]] = None,
        *,
        rules_path: Optional[str] = None,
        max_tags: int = 32,
        tag_filename: bool = True,
    ):
        if rules_path:
            loaded_rules, loaded_ext = load_tagging_rules(rules_path)
            self._rules = rules or loaded_rules
            self._extension_tags = extension_tags if extension_tags is not None else loaded_ext
        else:
            self._rules = rules or []
            self._extension_tags = extension_tags or {}
        self._max_tags = max(1, max_tags)
        self._tag_filename = tag_filename

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def enrich(
        self,
        ingest_result: IngestResult,
        *,
        doc_format: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        text = ingest_result.content or ""
        meta = dict(ingest_result.metadata)
        if extra:
            meta.update(extra)

        collected: List[str] = []
        matched_rules: List[str] = []

        existing = meta.get("tags")
        if isinstance(existing, str):
            collected.extend(t.strip() for t in re.split(r"[;,\|]+", existing) if t.strip())
        elif isinstance(existing, (list, tuple, set)):
            collected.extend(str(t).strip() for t in existing if t)

        for rule in self._rules:
            if _rule_matches(rule, text):
                matched_rules.append(rule.name)
                collected.extend(rule.tags)

        if self._tag_filename:
            source = meta.get("source_path") or meta.get("ocr_source_path") or ""
            if source:
                name = Path(str(source)).name
                stem = Path(name).stem
                if stem and len(stem) <= 64:
                    collected.append(f"file:{stem}")

        if doc_format:
            ext = doc_format.lower().lstrip(".")
            collected.extend(self._extension_tags.get(ext, []))

        source_path = meta.get("source_path") or meta.get("ocr_source_path")
        if source_path:
            ext = Path(str(source_path)).suffix.lower().lstrip(".")
            if ext:
                collected.extend(self._extension_tags.get(ext, []))

        # 去重保序
        seen: Set[str] = set()
        tags: List[str] = []
        for t in collected:
            key = t.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            tags.append(key)
            if len(tags) >= self._max_tags:
                break

        meta["tags"] = tags
        meta["categories"] = list(tags)
        meta["matched_rules"] = matched_rules
        meta["metadata_tagged"] = True
        meta["metadata_tagger"] = "rule_keyword"

        ingest_result.metadata = normalize_metadata(meta)
        _log.info(
            "metadata tags doc_id=%s rules=%s tags=%s",
            meta.get("doc_id"),
            matched_rules,
            tags,
        )
        return ingest_result

    def enrich_batch(
        self,
        items: List[IngestResult],
        **kwargs: Any,
    ) -> List[IngestResult]:
        return [self.enrich(item, **kwargs) for item in items]

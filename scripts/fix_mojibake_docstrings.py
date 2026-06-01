"""Replace entire corrupted triple-quoted docstrings with short English summaries."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAD = re.compile(r"[æåçéèï¼ãä¸»æº«¯å£å¯ç æ°æ®éç½®ä½¿ç¨å¼åº§ç¯å¢è¦ç¹å®¡¶å½å¾è½åå®¹æ¸æ´è§èå­ç¬¦æ ç¹]|çäº§|å¼å|éç½|æä»¶")

TARGETS = [
    ROOT / "utils" / "async_http.py",
    ROOT / "utils" / "token_counter.py",
    ROOT / "utils" / "retry_tools.py",
    ROOT / "utils" / "prompt_loader.py",
    ROOT / "utils" / "config_handler.py",
    ROOT / "utils" / "json_parser.py",
    ROOT / "utils" / "path_tools.py",
    ROOT / "utils" / "logger_handler.py",
    ROOT / "knowledge" / "shared" / "file_handler.py",
    ROOT / "knowledge" / "query" / "rewrite" / "hyde.py",
    ROOT / "knowledge" / "query" / "rewrite" / "multi_query.py",
    ROOT / "knowledge" / "pipeline" / "index" / "chunker.py",
    ROOT / "knowledge" / "pipeline" / "index" / "embedder.py",
    ROOT / "knowledge" / "pipeline" / "ingest" / "adapters" / "word_adapter.py",
    ROOT / "knowledge" / "query" / "router" / "classifier.py",
    ROOT / "knowledge" / "query" / "router" / "fusion.py",
    ROOT / "knowledge" / "query" / "router" / "router.py",
    ROOT / "knowledge" / "query" / "router" / "rules.py",
]


def fix_triple_quotes(text: str) -> str:
    """Collapse corrupted docstring blocks to a single-line placeholder."""
    out = []
    i = 0
    lines = text.splitlines(keepends=True)
    while i < len(lines):
        line = lines[i]
        if '"""' in line and BAD.search(line):
            indent = line[: len(line) - len(line.lstrip())]
            # start of possibly multi-line docstring
            if line.strip().count('"""') == 2 and line.strip().startswith('"""') and line.strip().endswith('"""'):
                out.append(f'{indent}"""Docstring."""\n')
                i += 1
                continue
            if '"""' in line:
                out.append(f'{indent}"""\n')
                i += 1
                while i < len(lines) and '"""' not in lines[i]:
                    if BAD.search(lines[i]):
                        i += 1
                        continue
                    i += 1
                if i < len(lines):
                    out.append(f'{indent}"""Docstring."""\n')
                    i += 1
                continue
        out.append(line)
        i += 1
    return "".join(out)


def main() -> None:
    for path in TARGETS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if not BAD.search(text):
            continue
        new = fix_triple_quotes(text)
        try:
            ast.parse(new)
            path.write_text(new, encoding="utf-8", newline="\n")
            print("ok", path.relative_to(ROOT))
        except SyntaxError as e:
            print("skip", path.relative_to(ROOT), e.lineno)


if __name__ == "__main__":
    main()

"""Replace mojibake comment/docstring lines with English placeholders."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", "__pycache__", ".git", "scripts", "tests", "layout_results", "yolo"}

# Lines containing these are almost always corrupted UTF-8 comments
BAD = re.compile(
    r"[æåçéèï¼ãä¸»æº«¯å£å¯ç æ°æ®éç½®ä½¿ç¨å¼åº§ç¯å¢è¦ç¹å®¡¶å½å¾è½åå®¹æ¸æ´è§èå­ç¬¦æ ç¹]|"
    r"çäº§|å¼å|éç½|æä»¶|å­ç¬¦|ææ¬|æ£è|é¨è¯"
)


def fix_line(line: str) -> str:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if stripped.startswith("#") and BAD.search(line):
        return f"{indent}# (encoding fixed)\n"
    if stripped.startswith('"""') and BAD.search(line) and stripped.count('"""') >= 2:
        return f'{indent}"""Module docstring."""\n'
    if stripped.startswith('"""') and BAD.search(line):
        return f'{indent}"""\n'
    if stripped.endswith('"""') and BAD.search(line) and '"""' in stripped[3:]:
        inner = stripped[3:-3].strip()
        if BAD.search(inner):
            return f'{indent}"""Docstring."""\n'
    return line


def main() -> None:
    n = 0
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = Path(root) / name
            text = path.read_text(encoding="utf-8", errors="replace")
            if not BAD.search(text):
                continue
            lines = [fix_line(l) if l.endswith("\n") else fix_line(l + "\n") for l in text.splitlines()]
            new = "".join(lines)
            if new == text:
                continue
            try:
                ast.parse(new)
            except SyntaxError:
                continue
            path.write_text(new, encoding="utf-8", newline="\n")
            n += 1
            print(path.relative_to(ROOT))
    print("total", n)


if __name__ == "__main__":
    main()

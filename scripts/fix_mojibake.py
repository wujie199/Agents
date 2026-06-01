"""Fix UTF-8 mojibake in project Python files (comments/docstrings only where possible)."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "__pycache__", ".git", "layout_results", "yolo", ".pytest_cache"}

# Manual replacements for common corrupted fragments (file-agnostic)
REPLACEMENTS = [
    ("构建生产环境 RunContext（企业级实现）。", "构建生产环境 RunContext（企业级实现）。"),
    ("构建开发环境 RunContext。", "构建开发环境 RunContext。"),
    ("配置文件目录", "配置文件目录"),
    ("数据目录", "数据目录"),
    ("Redis 主机", "Redis 主机"),
    ("Redis 端口", "Redis 端口"),
    ("Redis 密码", "Redis 密码"),
    ("S3/OBS 端点", "S3/OBS 端点"),
    ("S3 桶名", "S3 桶名"),
    ("使用内存缓存（开发用）", "使用内存缓存（开发用）"),
    ("使用内存图库（开发用）", "使用内存图库（开发用）"),
    ("覆盖指定 Port", "覆盖指定 Port"),
    ("基础文本清洗：去控制字符、规范空白与标点。", "基础文本清洗：去控制字符、规范空白与标点。"),
    ("语言检测：优先 dateutil.parser", "语言检测：优先 dateutil.parser"),
    ("常用 metadata 字段别名映射", "常用 metadata 字段别名映射"),
    ("文件不存在", "文件不存在"),
    ("不是文件:", "不是文件:"),
]

MOJIBAKE_RE = re.compile(r"[\u00c0-\u00ff]{2,}|[æåçéèï¼ãä¸»æº«¯å£å¯ç æ°æ®éç½®ä½¿ç¨å¼åº§ç¯å¢è¦ç¹å®¡¶å½å¾è½åå®¹æ¸æ´è§èå­ç¬¦æ ç¹]")


def try_recover(text: str) -> str | None:
    for enc in ("utf-8", "gbk", "cp1252", "latin-1"):
        try:
            raw = text.encode(enc)
            decoded = raw.decode("utf-8")
            if decoded != text and not MOJIBAKE_RE.search(decoded):
                return decoded
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return None


def fix_content(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    # line-by-line: try recover whole comment/docstring lines
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        stripped = line.lstrip()
        if MOJIBAKE_RE.search(line):
            # only fix comments and docstrings
            if stripped.startswith("#") or '"""' in line or "'''" in line:
                recovered = try_recover(line.strip().strip('"""').strip("'''"))
                if recovered and len(recovered) > 2:
                    indent = line[: len(line) - len(line.lstrip())]
                    if stripped.startswith("#"):
                        line = f"{indent}# {recovered}\n" if not line.endswith("\n") else f"{indent}# {recovered}\n"
        out.append(line)
    return "".join(out)


def should_scan(path: Path) -> bool:
    parts = path.parts
    return not any(s in SKIP_DIRS for s in parts)


def main() -> None:
    fixed = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = Path(root) / name
            if not should_scan(path):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not MOJIBAKE_RE.search(text):
                continue
            new = fix_content(text)
            if new != text:
                try:
                    ast.parse(new)
                except SyntaxError:
                    continue
                path.write_text(new, encoding="utf-8", newline="\n")
                fixed.append(str(path.relative_to(ROOT)))
    print("fixed", len(fixed), "files")
    for p in fixed[:30]:
        print(" ", p)
    if len(fixed) > 30:
        print(" ...")


if __name__ == "__main__":
    main()

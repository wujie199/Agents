#!/usr/bin/env bash
# 将技能目录同步到 skills/published/ 并触发热加载（通过 query_memory sync-skills）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${1:-$ROOT/skills/source}"
python "$ROOT/document/query_memory.py" sync-skills --source-dir "$SOURCE" "${@:2}"

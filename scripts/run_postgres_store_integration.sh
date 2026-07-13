#!/usr/bin/env bash
# PostgresStore L1 集成：启动本地 PG → setup → pytest
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

eval "$("$REPO_ROOT/scripts/pg_dev_up.sh" --export)"

echo "==> l1_langgraph_store_setup"
python scripts/l1_langgraph_store_setup.py

echo "==> pytest test_langgraph_store_setup"
pytest tests/test_langgraph_store_setup.py -q --tb=short

echo "OK: PostgresStore integration passed"

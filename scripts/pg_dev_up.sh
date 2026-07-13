#!/usr/bin/env bash
# 本地 Postgres（LangGraph PostgresStore / L2 集成测试用）
# 用法: source scripts/pg_dev_up.sh  或  eval "$(scripts/pg_dev_up.sh --export)"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGDATA="${PGDATA:-$REPO_ROOT/data/pg_dev}"
PORT="${PG_DEV_PORT:-5433}"
DB_USER="${PG_DEV_USER:-agents_app}"
DB_NAME="${PG_DEV_DB:-agents}"

find_pg_bin() {
  for d in /opt/homebrew/opt/postgresql@16 /opt/homebrew/opt/postgresql@18 /opt/homebrew/opt/postgresql; do
    if [[ -x "$d/bin/pg_ctl" ]]; then
      echo "$d/bin"
      return 0
    fi
  done
  if command -v pg_ctl >/dev/null 2>&1; then
    dirname "$(command -v pg_ctl)"
    return 0
  fi
  echo "ERROR: 未找到 pg_ctl，请安装 PostgreSQL（brew install postgresql@16）" >&2
  exit 1
}

PG_BIN="$(find_pg_bin)"
export PATH="$PG_BIN:$PATH"

if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
  mkdir -p "$PGDATA"
  initdb -D "$PGDATA" -U "$DB_USER" --auth-local=trust --auth-host=trust -E UTF8
fi

if ! pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
  pg_ctl -D "$PGDATA" -o "-p $PORT" -l "$PGDATA/logfile" start
  sleep 1
fi

if ! psql -p "$PORT" -U "$DB_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1; then
  createdb -p "$PORT" -U "$DB_USER" "$DB_NAME"
fi

DATABASE_URL="postgresql://${DB_USER}@localhost:${PORT}/${DB_NAME}"

if [[ "${1:-}" == "--export" ]]; then
  echo "export DATABASE_URL='$DATABASE_URL'"
  echo "export PGDATA='$PGDATA'"
else
  echo "Postgres dev 已就绪: $DATABASE_URL"
  echo "  export DATABASE_URL='$DATABASE_URL'"
fi

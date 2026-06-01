import sqlite3
import json
import hashlib
import os
import logging
from typing import Any, Optional
from .path_tools import get_project_root

logger = logging.getLogger(__name__)


class SQLiteCache:
    """基于 SQLite 的本地缓存，用于缓存 LLM 请求与响应。"""

    def __init__(self, db_name: str = "agent_cache.db"):
        """初始化缓存数据库连接。"""
        db_path = os.path.join(get_project_root(), db_name)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _generate_key(self, prompt: str, model: str) -> str:
        """根据 model + prompt 生成 MD5 缓存键。"""
        raw_key = f"{model}_{prompt}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def get(self, prompt: str, model: str = "default") -> Optional[Any]:
        """读取缓存；JSON 字符串会自动反序列化。"""
        key = self._generate_key(prompt, model)
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()

        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return row[0]
        return None

    def set(self, prompt: str, value: Any, model: str = "default") -> None:
        """写入或更新缓存。"""
        key = self._generate_key(prompt, model)
        value_str = (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else str(value)
        )
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                    (key, value_str),
                )
        except sqlite3.Error as e:
            logger.error("Failed to set cache: %s", e)

    def close(self) -> None:
        self.conn.close()

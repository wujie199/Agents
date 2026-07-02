# 中文乱码说明

## 原因

P1 迁移 `rag` → `knowledge` 时，曾用 PowerShell 对全仓库做 `rag.` → `knowledge.` 批量替换。  
在 Windows 默认编码下读写 UTF-8 文件，导致**中文注释/文档字符串损坏**（典型乱码：`æå»º`、`éç½®` 等）。

## 已处理

- `composition/production_factory.py` 等核心入口：恢复为正常中文
- `knowledge/bridges/cleaners/domain_cleaners.py`：正则与替换串改为 UTF-8 中文
- `utils/cache_handler.py` 等：重写 docstring
- 批量脚本已移除（历史 `scripts/fix_mojibake*.py`）；新乱码请用 Python 按 UTF-8 单文件修复

## 预防

- 批量改代码请用 **Python 脚本**（`utf-8` 读写），不要用 PowerShell `Get-Content` / `Set-Content` 默认编码
- 或在 PowerShell 中显式：`Get-Content -Encoding utf8` / `Set-Content -Encoding utf8`

## 若仍看到乱码

用编辑器或 Python 脚本按 UTF-8 读写修复；勿用 PowerShell 默认编码批量替换。

测试目录 `tests/` 中部分断言字符串可能仍为乱码，不影响 RAG 主链路；需要时可单独改测试文案。

#!/usr/bin/env python3
"""
测试本地 bge-reranker-base 重排。

用法:
  conda activate py3.11
  pip install sentence-transformers torch   # 若未安装
  python document/test_bge_reranker.py
  python document/test_bge_reranker.py --query "如何申请年假" --top-n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from document.rag.adapters.rerank.local_bge import LocalBgeReranker, print_rerank_report

DEFAULT_QUERY = "Python 中如何实现异步 HTTP 请求？"

DEFAULT_DOCS = [
    "asyncio 与 aiohttp 可以配合实现异步 HTTP 客户端。",
    "制作红烧肉需要先把五花肉焯水再去炒糖色。",
    "httpx 支持 async/await，写法类似 requests 但适合高并发场景。",
    "北京今日晴，最高气温 28 度。",
    "使用 asyncio.gather 可以同时发起多个协程任务。",
    "数据库索引可以加速 WHERE 条件的查询。",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 bge-reranker-base 重排测试")
    parser.add_argument("--query", "-q", default=DEFAULT_QUERY, help="查询文本")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="模型目录，默认 document/rag/weights/bge-reranker-base",
    )
    parser.add_argument("--top-n", type=int, default=3, help="返回前 N 条")
    parser.add_argument(
        "--doc",
        action="append",
        dest="docs",
        help="候选文档（可多次指定）；未指定则用内置样例",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备，如 cpu / mps / cuda:0",
    )
    args = parser.parse_args()

    documents = args.docs if args.docs else list(DEFAULT_DOCS)

    print(f"模型目录: {args.model_dir or '(默认 weights/bge-reranker-base)'}")
    print(f"查询: {args.query!r}")
    print(f"候选数: {len(documents)}\n")

    reranker = LocalBgeReranker(model_dir=args.model_dir, device=args.device)
    print(f"已加载: {reranker.model_dir}\n")

    results = reranker.rerank(args.query, documents, top_n=args.top_n)
    print_rerank_report(args.query, documents, results, top_n_label=args.top_n)
    print("完成。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8
"""Memory eval runner."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from document.memory.evaluation.pipeline import run_memory_eval, write_memory_eval_report
from document.memory.evaluation.stack_factory import (
    build_eval_memory_stack,
    try_load_models,
)


async def run_memory_eval_job(
    *,
    dataset_path: str,
    run_id: str = "memory-eval",
    data_dir: str = "data",
    sample_limit: Optional[int] = None,
    use_llm: bool = False,
    use_mock_only: bool = False,
    config_dir: str = "config",
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        memory, archive_db, ctx, _store = await build_eval_memory_stack(
            tmp_root=tmp_root,
        )
        if use_llm:
            models = try_load_models(config_dir)
            if models is None:
                use_mock_only = True
                llm_fallback = "ModelRegistry unavailable; using mock_extract for l1_extract"
            else:
                ctx.models = models
                llm_fallback = None
        else:
            llm_fallback = None

        async def l4_factory(sample) -> Any:
            sub = tmp_root / f"l4_{sample.id}"
            sub.mkdir(parents=True, exist_ok=True)
            mem, _, _, _ = await build_eval_memory_stack(
                tmp_root=sub,
                enable_l4=True,
                l4_facts=sample.l4_facts,
                tenant_id=sample.tenant_id,
                user_id=sample.user_id,
            )
            return mem

        result = await run_memory_eval(
            memory=memory,
            ctx=ctx,
            dataset_path=dataset_path,
            sample_limit=sample_limit,
            use_mock_only=use_mock_only or not use_llm,
            l4_memory_factory=l4_factory,
        )
        try:
            close = getattr(archive_db, "close", None)
            if callable(close):
                maybe = close()
                if hasattr(maybe, "__await__"):
                    await maybe
        except Exception:
            pass
        out_dir = Path(data_dir) / "memory_eval" / "results" / run_id
        write_memory_eval_report(result, out_dir)
        result["report_dir"] = str(out_dir)
        result["use_llm"] = use_llm
        if llm_fallback:
            result["llm_fallback"] = llm_fallback
        return result

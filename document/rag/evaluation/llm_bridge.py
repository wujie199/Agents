# -*- coding: utf-8 -*-
"""ModelRegistry → RAGAS LangChain LLM + embeddings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from document.rag.evaluation.stack_factory import EvalStackConfig


@dataclass
class RagasModelBundle:
    llm: Any
    embeddings: Any | None = None


class _AsyncCapableLLM:
    """Wrap sync-only LangChain chat models for RAGAS async metrics."""

    def __init__(self, llm: Any):
        self._llm = llm

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        import asyncio

        return await asyncio.to_thread(self._llm.invoke, input, config, **kwargs)

    async def agenerate(self, messages: Any, stop: Any = None, **kwargs: Any) -> Any:
        import asyncio

        return await asyncio.to_thread(
            self._llm.generate, messages, stop=stop, **kwargs
        )


def _resolve_role_instance(
    registry: Any,
    role: str,
    eval_cfg: EvalStackConfig,
) -> str:
    roles = getattr(registry, "_roles", {})
    if role in roles:
        return roles[role].profile
    overlay = eval_cfg.roles.get(role) or {}
    instance = overlay.get("instance") or overlay.get("profile")
    if instance:
        return str(instance)
    raise ValueError(f"Role not found for RAGAS judge: {role}")


def _resolve_profile(registry: Any, instance_name: str) -> Any:
    profiles = getattr(registry, "_profiles", {})
    profile = profiles.get(instance_name)
    if profile is None:
        raise ValueError(f"Model instance not found: {instance_name}")
    resolve = getattr(registry, "_resolve_profile", None)
    if callable(resolve):
        return resolve(profile)
    return profile


def _provider_base_url(registry: Any, profile: Any) -> str | None:
    base_url = getattr(profile, "base_url", None)
    provider = getattr(profile, "provider", "")
    provider_configs = getattr(registry, "_provider_configs", {})
    if provider in provider_configs:
        base_url = base_url or provider_configs[provider].get("base_url")
    return base_url


def _resolve_api_key(profile: Any) -> str | None:
    """优先使用 models.yml 中的 api_key；仅当未配置时才读 api_key_env。"""
    api_key = getattr(profile, "api_key", None)
    if api_key:
        return str(api_key)
    api_key_env = getattr(profile, "api_key_env", None)
    if api_key_env:
        return os.environ.get(api_key_env)
    return None


def _patch_chat_huggingface_async() -> None:
    """RAGAS metrics call ChatHuggingFace._agenerate; HF pipeline is sync-only."""
    from langchain_huggingface.chat_models.huggingface import (
        ChatHuggingFace,
        _is_huggingface_pipeline,
    )

    if getattr(ChatHuggingFace, "_ragas_async_patched", False):
        return

    original = ChatHuggingFace._agenerate

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if _is_huggingface_pipeline(self.llm):
            import asyncio

            llm_input = self._to_chat_prompt(messages)
            llm_result = await asyncio.to_thread(
                self.llm._generate,
                [llm_input],
                stop,
                run_manager,
                **kwargs,
            )
            return self._to_chat_result(llm_result)
        return await original(
            self, messages, stop=stop, run_manager=run_manager, **kwargs
        )

    ChatHuggingFace._agenerate = _agenerate
    ChatHuggingFace._ragas_async_patched = True


def _build_local_hf_ragas_llm(profile: Any) -> Any:
    """RAGAS judge via local HuggingFace weights (no API key)."""
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_path = getattr(profile, "model_path", None)
    if not model_path:
        raise ValueError("local_hf judge requires model_path")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    device = 0 if torch.cuda.is_available() else -1
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        device=device,
    )
    _patch_chat_huggingface_async()
    chat = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen))
    wrapped = _AsyncCapableLLM(chat)
    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(wrapped)
    except ImportError:
        return wrapped


def build_ragas_llm(registry: Any, eval_cfg: EvalStackConfig) -> Any:
    """Build RAGAS-compatible LangChain LLM from ModelRegistry."""
    role = str(eval_cfg.judge.get("model_role") or "eval_judge_llm")
    instance = _resolve_role_instance(registry, role, eval_cfg)
    profile = _resolve_profile(registry, instance)

    provider = str(getattr(profile, "provider", "") or "").lower()
    if provider == "local_hf":
        return _build_local_hf_ragas_llm(profile)

    from langchain_openai import ChatOpenAI

    model_name = profile.model_name
    if profile.model_name_env:
        model_name = os.environ.get(profile.model_name_env) or model_name
    base_url = _provider_base_url(registry, profile)
    api_key = _resolve_api_key(profile) or "not-needed"

    lc_llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout=getattr(profile, "timeout", 120),
        temperature=0,
    )

    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(lc_llm)
    except ImportError:
        return lc_llm


def build_ragas_embeddings(registry: Any, eval_cfg: EvalStackConfig) -> Any | None:
    """Build RAGAS-compatible HuggingFace embeddings from embedding role."""
    role = str(eval_cfg.judge.get("embedding_role") or "embedding")
    try:
        instance = _resolve_role_instance(registry, role, eval_cfg)
    except ValueError:
        return None
    profile = _resolve_profile(registry, instance)
    model_path = getattr(profile, "model_path", None)
    if not model_path:
        return None

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        return None

    normalize = getattr(profile, "normalize", True)
    lc_emb = HuggingFaceEmbeddings(
        model_name=model_path,
        encode_kwargs={"normalize_embeddings": bool(normalize)},
    )
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(lc_emb)
    except ImportError:
        return lc_emb


def build_ragas_models(registry: Any, eval_cfg: EvalStackConfig) -> RagasModelBundle:
    llm = build_ragas_llm(registry, eval_cfg)
    embeddings = build_ragas_embeddings(registry, eval_cfg)
    return RagasModelBundle(llm=llm, embeddings=embeddings)

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
    api_key = getattr(profile, "api_key", None)
    api_key_env = getattr(profile, "api_key_env", None)
    if api_key_env:
        api_key = api_key or os.environ.get(api_key_env)
    return api_key


def build_ragas_llm(registry: Any, eval_cfg: EvalStackConfig) -> Any:
    """Build RAGAS-compatible LangChain ChatOpenAI from ModelRegistry."""
    role = str(eval_cfg.judge.get("model_role") or "eval_judge_llm")
    instance = _resolve_role_instance(registry, role, eval_cfg)
    profile = _resolve_profile(registry, instance)

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

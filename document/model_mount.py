"""外置模型盘挂载检测与用户提醒（不自动下载权重）。"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)

DEFAULT_MODEL_VOLUME = Path("/Volumes/wj")
DEFAULT_MODEL_ROOT = DEFAULT_MODEL_VOLUME / "model"
DEFAULT_OCR_MODEL_ROOT = DEFAULT_MODEL_ROOT / "ocr"
DEFAULT_EMBEDDING_MODEL = DEFAULT_MODEL_ROOT / "embedding" / "bge-small-zh-v1.5"
DEFAULT_RERANK_MODEL = DEFAULT_MODEL_ROOT / "rerank" / "bge-reranker-base"

# 本地对话：外置盘 HF 权重 / 已运行的 Ollama·vLLM（本项目不会 pull / 下载）
DEFAULT_LOCAL_LLM_ROOT = DEFAULT_MODEL_ROOT / "llm"
DEFAULT_LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_CLOUD_CHAT_MODEL = "qwen3.6-plus"


def external_volume_name(path: Path | str) -> str | None:
    p = Path(path)
    if len(p.parts) >= 3 and p.parts[0] == "/" and p.parts[1] == "Volumes":
        return p.parts[2]
    return None


def is_external_volume_mounted(path: Path | str) -> bool:
    """路径在 /Volumes/<name>/ 下时，检查该卷是否已挂载。"""
    vol = external_volume_name(path)
    if vol is None:
        return True
    return Path("/Volumes", vol).is_dir()


def unmounted_reminder(
    *,
    path: Path | str,
    purpose: str,
    env_hint: str | None = None,
) -> str:
    vol = external_volume_name(path)
    if vol:
        mount_hint = f"请先挂载外置盘：/Volumes/{vol}"
    else:
        mount_hint = f"请确认模型路径存在：{path}"
    env_line = (
        f"也可设置环境变量 {env_hint} 指向其他已就绪的模型目录。"
        if env_hint
        else ""
    )
    lines = [
        f"【外置模型盘未就绪】{purpose}",
        f"  期望路径: {path}",
        f"  {mount_hint}",
    ]
    if env_line:
        lines.append(f"  {env_line}")
    lines.append("  本项目不会自动下载模型；挂载外置盘后使用已有权重即可。")
    return "\n".join(lines)


def warn_if_unmounted(
    path: Path | str,
    *,
    purpose: str,
    env_hint: str | None = None,
) -> bool:
    """未挂载时打 warning，返回 False；已挂载返回 True。"""
    if is_external_volume_mounted(path):
        return True
    _log.warning(unmounted_reminder(path=path, purpose=purpose, env_hint=env_hint))
    return False


def require_mounted_volume(
    path: Path | str,
    *,
    purpose: str,
    env_hint: str | None = None,
) -> Path:
    """未挂载时抛出 FileNotFoundError（含挂载提醒，不触发下载）。"""
    p = Path(path).expanduser()
    if not is_external_volume_mounted(p):
        raise FileNotFoundError(
            unmounted_reminder(path=p, purpose=purpose, env_hint=env_hint)
        )
    return p

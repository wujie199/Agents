import yaml
from utils.path_tools import get_abs_path
from document.rag.config.rag_yaml import load_rag_chroma_section


def _load_yaml_file(config_path: str, *, encoding: str = "utf-8") -> dict:
    path = get_abs_path(config_path)
    if not path.is_file():
        return {}
    with path.open("r", encoding=encoding) as f:
        return yaml.safe_load(f) or {}


def load_agent_config(
    config_path: str = get_abs_path("config/agent.yml"),
    encoding: str = "utf-8",
) -> dict:
    return _load_yaml_file(config_path, encoding=encoding)


def load_chroma_config(config_path: str | None = None, encoding: str = "utf-8"):
    if config_path:
        return _load_yaml_file(config_path, encoding=encoding)
    return load_rag_chroma_section(get_abs_path("config"))


def load_prompts_config(
    config_path: str = get_abs_path("config/prompt.yml"),
    encoding: str = "utf-8",
) -> dict:
    return _load_yaml_file(config_path, encoding=encoding)


def load_tools_config(
    config_path: str = get_abs_path("config/tools.yml"),
    encoding: str = "utf-8",
) -> dict:
    return _load_yaml_file(config_path, encoding=encoding)


def load_subprompt_desc(
    config_path: str = get_abs_path("config/prompt.yml"),
    encoding: str = "utf-8",
) -> list[dict[str, str]]:
    """读取 prompt.yml 中 subPrompt 全部子项，按文件顺序返回。"""
    cfg = load_prompts_config(config_path, encoding)
    sub = cfg.get("subPrompt")
    if not isinstance(sub, dict):
        return []
    out: list[dict[str, str]] = []
    for name, body in sub.items():
        if not isinstance(body, dict):
            continue
        desc = body.get("desc")
        path = body.get("path")
        out.append(
            {
                "name": str(name),
                "desc": "" if desc is None else str(desc),
                "path": "" if path is None else str(path),
            }
        )
    return out


chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()
tools_config = load_tools_config()
sub_prompt_desc = load_subprompt_desc()


if __name__ == "__main__":
    print(sub_prompt_desc)

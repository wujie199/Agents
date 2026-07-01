import yaml
from utils.path_tools import get_abs_path

def load_agent_config(config_path:str = get_abs_path('config/agent.yml'), encoding:str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_chroma_config(config_path:str = get_abs_path('config/chroma.yml'), encoding:str = 'utf-8'):
    with open(config_path,'r',encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_prompts_config(config_path:str = get_abs_path('config/prompt.yml'), encoding:str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_tools_config(config_path:str = get_abs_path('config/tools.yml'), encoding:str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_subprompt_desc(
    config_path: str = get_abs_path("config/prompt.yml"),
    encoding: str = "utf-8",
) -> list[dict[str, str]]:
    """
    读取 prompt.yml 中 subPrompt 全部子项，按文件顺序返回。
    每项含 name（键名，如 OverviewPrompt）、desc、path。
    """
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

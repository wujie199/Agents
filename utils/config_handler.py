import yaml
from utils.path_tools import get_abs_path

def load_rag_config(config_path:str = get_abs_path('config/rag.yml'), encoding:str = 'utf-8'):
    with open(config_path, 'r', encoding = encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

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

def load_llm_config(config_path: str = get_abs_path('config/llm.yml'), encoding: str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_subprompt_desc(
    config_path: str = get_abs_path("config/prompt.yml"),
    encoding: str = "utf-8",
) -> list[dict[str, str]]:
    """
    è¯»å prompt.yml ä¸­ subPrompt ä¸å¨é¨å­é¡¹ï¼ææä»¶é¡ºåºè¿åã
    æ¯é¡¹å« nameï¼é®åï¼å¦ OverviewPromptï¼ãdescãpathã
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

rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()
tools_config = load_tools_config()
sub_prompt_desc = load_subprompt_desc()

# (encoding fixed)
llm_config = load_llm_config()


if __name__ == "__main__":
    print(sub_prompt_desc)

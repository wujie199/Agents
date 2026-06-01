import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
from utils.config_handler import agent_config, prompts_config
from utils.path_tools import get_abs_path
from utils.logger_handler import logger


def _read_text_file(path: str) -> str:
    """Module docstring."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()

def load_main_agent_prompts():
    try:
        main_agent_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except KeyError as e:
        logger.error("[load_main_agent_prompt]å¨ yaml éç½®é¡¹ä¸­æ²¡æ main_prompt_path éç½®é¡¹")
        raise e

    try:
        return _read_text_file(main_agent_prompt_path)
    except Exception as e:
        logger.error(f"[load_main_agent_prompt]è§£æä¸» Agent æç¤ºè¯æä»¶å¤±è´¥: {str(e)}")
        raise e


def load_system_prompt():
    """Module docstring."""
    return load_main_agent_prompts()


def load_rag_prompt():
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompt]å¨ yaml éç½®é¡¹ä¸­æ²¡æ rag_summarize_prompt_path éç½®é¡¹")
        raise e

    try:
        return _read_text_file(rag_prompt_path)
    except Exception as e:
        logger.error(f"[load_rag_prompt]è§£æ RAG æ»ç»æç¤ºè¯æä»¶å¤±è´¥: {str(e)}")
        raise e


def load_report_prompt():
    try:
        raw = prompts_config.get("report_prompt_path")
    except Exception as e:
        logger.error("[load_report_prompt]è¯»å prompts éç½®å¤±è´¥")
        raise e
    if raw is None or not str(raw).strip():
        return ""
    report_prompt_path = get_abs_path(str(raw).strip())
    try:
        return _read_text_file(report_prompt_path)
    except Exception as e:
        logger.error(f"[load_report_prompt]è§£æ report æ¥åçææç¤ºè¯æä»¶å¤±è´¥: {str(e)}")
        raise e

def load_router_prompt():
    try:
        report_prompt_path = get_abs_path(prompts_config["router_path"])
    except KeyError as e:
        logger.error(f"[load_router_prompt]å¨ yaml éç½®é¡¹ä¸­æ²¡æ router_path éç½®é¡¹")
        raise e

    try:
        return _read_text_file(report_prompt_path)
    except Exception as e:
        logger.error(f"[load_router_prompt]è§£æreportæ¥åçææç¤ºè¯æä»¶å¤±è´¥: {str(e)}")
        raise e

def load_sub_prompt(file:str):
    try:
        subPrompt = get_abs_path(prompts_config["subPrompt"][file]["path"])
    except KeyError as e:
        logger.error(f"[load_sub_prompt]å¨ yaml éç½®é¡¹ä¸­æ²¡æ subPrompt éç½®é¡¹")
        raise e

    try:
        return _read_text_file(subPrompt)
    except Exception as e:
        logger.error(f"[load_report_prompt]è§£æreportæ¥åçææç¤ºè¯æä»¶å¤±è´¥: {str(e)}")
        raise e

loadMainAgentPrompts = load_main_agent_prompts()
loadSystemPrompt = load_system_prompt()
loadRagPrompt = load_rag_prompt()
loadReportPrompt = load_report_prompt()
loadRouterPrompt = load_router_prompt()



if __name__ == "__main__":
    print(load_sub_prompt("EngineeringConstructionConditionsPrompt"))

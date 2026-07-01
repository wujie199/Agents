"""
Utils package exports: provide stable aliases for commonly used helpers.
"""
from .config_handler import (
    load_agent_config,
    load_chroma_config,
    load_prompts_config,
    load_tools_config,
    agent_config,
    chroma_config,
    prompts_config,
    tools_config,
    sub_prompt_desc,
)
from .logger_handler import get_logger as setup_logger, logger
from .path_tools import get_project_root, get_abs_path
from document.rag.shared.file_handler import (
    get_file_md5_hex,
    pdf_loader,
    txt_loader,
    parse_data_txt_records,
)
from .token_counter import count_tokens
# data_cleaner implementation now lives under rag; import from there to
# keep top-level compatibility while ensuring single canonical implementation.
from document.rag.shared.data_cleaner import (
    clean_text,
    postprocess_ocr,
    extract_table_text,
    normalize_metadata,
    dedupe_chunks,
    semantic_dedupe,
    batch_clean,
    batch_clean_parallel,
    ocr_confidence_score,
    get_data_cleaner_metrics,
    reset_data_cleaner_metrics,
)

# (encoding fixed)
from utils.path_tools import get_abs_path

def load_prompt():
    try:
        main_path = get_abs_path(prompts_config.get('main_prompt_path', 'prompts/main.txt'))
        with open(main_path, 'r', encoding='utf-8-sig') as f:
            return f.read()
    except Exception:
        return ''
from .retry_tools import auto_retry
from .json_parser import extract_and_parse_json
from document.rag.shared.text_chunker import split_text_into_chunks
from .async_http import AsyncHttpClient
from .cache_handler import SQLiteCache

__all__ = [
    # config
    "load_agent_config",
    "load_chroma_config",
    "load_prompts_config",
    "load_tools_config",
    "agent_config",
    "chroma_config",
    "prompts_config",
    "tools_config",
    "sub_prompt_desc",
    # logger
    "setup_logger",
    "logger",
    # path & file
    "get_project_root",
    "get_abs_path",
    "get_file_md5_hex",
    "pdf_loader",
    "txt_loader",
    "parse_data_txt_records",
    # prompts
    "load_prompt",
    # advanced
    "count_tokens",
    "clean_text",
    "postprocess_ocr",
    "extract_table_text",
    "normalize_metadata",
    "dedupe_chunks",
    "semantic_dedupe",
    "batch_clean",
    "batch_clean_parallel",
    "ocr_confidence_score",
    "get_data_cleaner_metrics",
    "reset_data_cleaner_metrics",
    "auto_retry",
    "extract_and_parse_json",
    "split_text_into_chunks",
    "AsyncHttpClient",
    "SQLiteCache",
]

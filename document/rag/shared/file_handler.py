import os, hashlib, re
import yaml
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader

def get_file_md5_hex(file_path:str):
    if not os.path.exists(file_path):
        logger.error(f"文件不存在 {file_path}")
        return

    if not os.path.isfile(file_path):
        logger.error(f"不是文件: {file_path}")
        return

    md5_obj = hashlib.md5()

    # (encoding fixed)
    chunk_size = 4096

    try:
        with open(file_path,'rb') as f: # å¿é¡»ç¨äºè¿å¶è¯»åæä»¶
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"è®¡ç®æä»¶MD5{file_path}å¤±è´¥: {str(e)}")
        return None


# (encoding fixed)
    files = []
    if not os.path.isdir(path):
        logger.error(f"ä¸æ¯æä»¶å¤? {path}")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)

def pdf_loader(file_path:str, password:str = None) -> list[Document]:
    return PyPDFLoader(file_path, password=password).load()

def txt_loader(file_path:str,encoding:str = "utf-8")-> list[Document]:
    return TextLoader(file_path,encoding=encoding).load()

def parse_data_txt_records(file_path: str, encoding: str = "utf-8") -> list[Document]:
    """
    page_content ä»ä¸º embedding å­æ®µææ¬ï¼å¥åºæ¶ç?Chroma ç?embedding_function
    """

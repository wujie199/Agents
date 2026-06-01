import logging
from typing import Union

logger = logging.getLogger(__name__)

try:
    import tiktoken
except ImportError:
    logger.warning("tiktoken åºæªå®è£ï¼è¯·ä½¿ç¨ 'pip install tiktoken' å®è£ä»¥è·ååç¡®ç token è®¡æ°ã")
    tiktoken = None

def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    """
    è®¡ç®ç»å®ææ¬ç Token æ°éã
    
    å¯¹äºå¤§è¯­è¨æ¨¡åï¼å¦ Qwen, OpenAI ç­ï¼ï¼éå¸¸æä¸ä¸æçªå£éå¶ã
    å¨åéé¿ææ¬ï¼å¦ RAG çå¬åææ¡£æé¿å¯¹è¯åå²ï¼åï¼
    ä½¿ç¨æ­¤å·¥å·è®¡ç® token æ°éï¼å¯ææé¿åè¶åºéå¶ï¼å¹¶æå©äºææ¬ä¼°ç®ã
    
    Args:
        text (str): éè¦è®¡ç® token æ°éçåå§ææ¬åå®¹ã
        model_name (str): ç¼ç å¨æ¨¡ååç§°ï¼é»è®¤ä¸º 'cl100k_base'ï¼å¤§å¤æ°ç°ä»£å¤§æ¨¡åçè¿ä¼¼éç¨æ åï¼ã
        
    Returns:
        int: è®¡ç®å¾åºç token æ°éã
             å¦ææªå®è£ tiktoken åºï¼å°è¿ååºäºå­ç¬¦æ°çç²ç¥ä¼°ç®å¼ï¼çº¦ 1 token â 1.5 ä¸­æå­ç¬¦ï¼ã
    """
    if not text:
        return 0
        
    if tiktoken:
        try:
            encoding = tiktoken.get_encoding(model_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Token è®¡ç®å¤±è´¥ï¼ä½¿ç¨è¿ä¼¼ä¼°ç®: {e}")
            
    # (encoding fixed)
    # (encoding fixed)
    return int(len(text) * 0.7)

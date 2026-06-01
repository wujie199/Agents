import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def extract_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    ä» LLM è¿åçææ¬ä¸­æåå¹¶è§£æ JSON æ°æ®ã
    
    å¨ä½¿ç¨ Function Calling æè¦æ±å¤§æ¨¡åè¿åç»æåæ°æ®æ¶ï¼
    æ¨¡åææ¶ä¼éå¸¦ Markdown æ è®° (ä¾å¦ ```json ... ```) æèå¶ä»è§£éæ§åè¨/åè¯­ã
    æ­¤å·¥å·å½æ°ä½¿ç¨æ­£åè¡¨è¾¾å¼åå¤ç§å®¹éç­ç¥ï¼å®å¨å°æååº JSON æ ¸å¿å¹¶è§£æã
    
    Args:
        text (str): LLM è¿åçåå§ææ¬å­ç¬¦ä¸²ã
        
    Returns:
        Optional[Dict[str, Any]]: è§£ææååç Python å­å¸ãå¦ææåæè§£æå¤±è´¥ï¼åè¿å Noneã
    """
    if not text:
        return None

    # # 1. 策略一：尝试直接作为纯 JSON 解析（最快路径，适用于表现完美的 LLM）
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # # 2. 策略二：尝试提取 Markdown 代码块中的 JSON
    # å¹é ```json æè ``` å¼å§ï¼å°ä¸ä¸ä¸ª ``` ç»æä¹é´çåå®¹
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"ä» Markdown ä»£ç åè§£æ JSON å¤±è´¥: {e}\næåçåå§åå®¹: {json_str}")
    
    # 3. ç­ç¥ä¸ï¼æå®½æ¾å¹éï¼å¯»æ¾ç¬¬ä¸ä¸ª '{' åæåä¸ä¸ª '}' ä¹é´çåå®¹
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"å°è¯å®½æ³æå (è±æ¬å·é´) JSON å¤±è´¥: {e}")

    # # 所有策略均失败
    logger.warning("æªè½ä»ç»å®ææ¬ä¸­æåå°ææç JSON æ ¼å¼æ°æ®ã")
    return None

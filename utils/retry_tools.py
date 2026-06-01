import time
import asyncio
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

def auto_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    èªå¨éè¯è£é¥°å¨ï¼æ¯æåæ­¥åå¼æ­¥å½æ°ï¼å¹¶å¸¦æææ°éé¿ï¼Exponential Backoffï¼æºå¶ã
    
    å¨è°ç¨ LLM API æå¤é¨ç½ç»æå¡æ¶ï¼ç»å¸¸ä¼éå°å¶åçç½ç»æ³¢å¨ãå¹¶åéæµç­é®é¢ã
    ä½¿ç¨æ­¤è£é¥°å¨å¯ä»¥ä¼éå°è¿è¡èªå¨éè¯ï¼å¢å¼ºç³»ç»çå¥å£®æ§ã
    
    Args:
        max_retries (int): æå¤§éè¯æ¬¡æ°ãé»è®¤ä¸º 3ã
        delay (float): é¦æ¬¡éè¯åçå»¶è¿æ¶é´ï¼ç§ï¼ãé»è®¤ä¸º 1.0 ç§ã
        backoff (float): æ¯æ¬¡éè¯å»¶è¿çéå¢åæ°ï¼ææ°éé¿ï¼ãé»è®¤ä¸º 2.0ã
                         ä¾å¦ï¼ç¬¬1æ¬¡ç­1ç§ï¼ç¬¬2æ¬¡ç­2ç§ï¼ç¬¬3æ¬¡ç­4ç§ã
        exceptions (tuple): éè¦è§¦åéè¯çå¼å¸¸ç±»ååç»ãé»è®¤ä¸ºæè·ææ Exceptionã
        
    Returns:
        Callable: åè£åçå½æ°ï¼åæ­¥æå¼æ­¥èªå¨ééï¼ã
    """
    def decorator(func: Callable) -> Callable:
        # (encoding fixed)
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                current_delay = delay
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_retries:
                            logger.error(f"å¼æ­¥å½æ° {func.__name__} è¾¾å°æå¤§éè¯æ¬¡æ° ({max_retries})ï¼æåä¸æ¬¡éè¯¯: {e}")
                            raise
                        logger.warning(f"å¼æ­¥å½æ° {func.__name__} æ§è¡å¤±è´¥: {e}. {current_delay} ç§åè¿è¡ç¬¬ {attempt + 1} æ¬¡éè¯...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                current_delay = delay
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_retries:
                            logger.error(f"åæ­¥å½æ° {func.__name__} è¾¾å°æå¤§éè¯æ¬¡æ° ({max_retries})ï¼æåä¸æ¬¡éè¯¯: {e}")
                            raise
                        logger.warning(f"åæ­¥å½æ° {func.__name__} æ§è¡å¤±è´¥: {e}. {current_delay} ç§åè¿è¡ç¬¬ {attempt + 1} æ¬¡éè¯...")
                        time.sleep(current_delay)
                        current_delay *= backoff
            return sync_wrapper
    return decorator

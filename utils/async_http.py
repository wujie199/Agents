import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import aiohttp
except ImportError:
    logger.warning("aiohttp åºæªå®è£ãè¯·ä½¿ç¨ 'pip install aiohttp' ä»¥ä½¿ç¨å¼æ­¥ HTTP å·¥å·ã")
    aiohttp = None

class AsyncHttpClient:
    """
    å¼æ­¥ HTTP å®¢æ·ç«¯å°è£å·¥å·ã
    
    å¨ Agent å¼åä¸­ï¼ç»å¸¸éè¦å¹¶åè°ç¨å¤ä¸ªå¤é¨ APIï¼å¦èç½æç´¢ãå¹¶è¡æ¥è¡¨ç­ï¼ã
    ä½¿ç¨åæ­¥ç requests ä¼å¯¼è´ä¸¥éç I/O é»å¡ï¼ææ¢ Agent çæ´ä½ååºéåº¦ã
    æ­¤å·¥å·åºäº aiohttp å°è£äºå¸¸ç¨ç GET å POST å¼æ­¥è¯·æ±ã
    """
    
    @staticmethod
    async def get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        åèµ·å¼æ­¥ GET è¯·æ±ã
        
        Args:
            url (str): è¯·æ±å°å
            params (dict, optional): URL æ¥è¯¢åæ°
            headers (dict, optional): è¯·æ±å¤´
            timeout (int): è¶æ¶æ¶é´ï¼ç§ï¼
            
        Returns:
            dict: ååºç JSON æ°æ®ãå¦æè¿åçä¸æ¯ JSONï¼ååè£å¨ {"text": "..."} ä¸­ãå¤±è´¥è¿å Noneã
        """
        if not aiohttp:
            raise ImportError("è¯·åå®è£ aiohttp åº")
            
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, params=params, timeout=timeout) as response:
                    response.raise_for_status()
                    # (encoding fixed)
                    try:
                        data = await response.json()
                    except Exception:
                        data = {"text": await response.text()}
                    return data
        except Exception as e:
            logger.error(f"å¼æ­¥ GET è¯·æ±å¤±è´¥ [{url}]: {e}")
            return None

    @staticmethod
    async def post(url: str, json_data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        åèµ·å¼æ­¥ POST è¯·æ±ã
        
        Args:
            url (str): è¯·æ±å°å
            json_data (dict, optional): POST ç JSON æ°æ®ä½
            headers (dict, optional): è¯·æ±å¤´
            timeout (int): è¶æ¶æ¶é´ï¼ç§ï¼
            
        Returns:
            dict: ååºæ°æ®ãå¤±è´¥è¿å Noneã
        """
        if not aiohttp:
            raise ImportError("è¯·åå®è£ aiohttp åº")
            
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(url, json=json_data, timeout=timeout) as response:
                    response.raise_for_status()
                    try:
                        data = await response.json()
                    except Exception:
                        data = {"text": await response.text()}
                    return data
        except Exception as e:
            logger.error(f"å¼æ­¥ POST è¯·æ±å¤±è´¥ [{url}]: {e}")
            return None

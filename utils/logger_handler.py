from datetime import datetime
import logging
from utils.path_tools import get_abs_path
import os

# (encoding fixed)
LOG_DIR = get_abs_path("logs")

# (encoding fixed)
os.makedirs(LOG_DIR, exist_ok=True)
# (encoding fixed)
DEFORT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d: - %(message)s"


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file: str = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # (encoding fixed)
    if logger.hasHandlers():
        return logger

    # (encoding fixed)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(DEFORT_LOG_FORMAT))
    logger.addHandler(console_handler)

    # (encoding fixed)
    if not log_file:
        # (encoding fixed)
        log_file = os.path.join(LOG_DIR, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(DEFORT_LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger


# (encoding fixed)
logger = get_logger()

if __name__ == "__main__":
    logger = get_logger()
    logger.info("This is a test log")
    logger.debug("This is a debug log")
    logger.warning("This is a warning log")
    logger.error("This is an error log")
    logger.critical("This is a critical log")



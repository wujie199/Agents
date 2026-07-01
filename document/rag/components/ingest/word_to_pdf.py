"""Word/HTML 转 PDF，供 OCR 摄取使用。"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ingest.word_to_pdf")

_WORD_EXTENSIONS = {".doc", ".docx"}
_HTML_EXTENSIONS = {".html", ".htm"}


def _find_soffice() -> Optional[str]:
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    mac_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    if os.path.isfile(mac_path):
        return mac_path
    return None


def needs_pdf_conversion(file_path: str) -> bool:
    ext = Path(file_path).suffix.lower()
    return ext in _WORD_EXTENSIONS or ext in _HTML_EXTENSIONS


def convert_to_pdf(
    input_path: str,
    output_dir: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """
    使用 LibreOffice 将 Word/HTML 转为 PDF。

    Returns:
        生成的 PDF 绝对路径

    Raises:
        RuntimeError: 未安装 LibreOffice 或转换失败
    """
    input_path = str(Path(input_path).resolve())
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    soffice = _find_soffice()
    if not soffice:
        raise RuntimeError(
            "未找到 LibreOffice (soffice)。请安装:\n"
            "  macOS: brew install --cask libreoffice\n"
            "  Ubuntu: sudo apt-get install libreoffice"
        )

    out_dir = output_dir or tempfile.mkdtemp(prefix="ingest_pdf_")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Word/HTML 转 PDF: %s -> %s", input_path, out_dir)
    cmd = [
        soffice,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        out_dir,
        input_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"LibreOffice 转换超时 ({timeout}s): {input_path}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"LibreOffice 转换失败: {stderr or 'unknown error'}")

    pdf_path = Path(out_dir) / f"{Path(input_path).stem}.pdf"
    if not pdf_path.is_file():
        raise RuntimeError(f"转换后未找到 PDF: {pdf_path}")

    return str(pdf_path.resolve())

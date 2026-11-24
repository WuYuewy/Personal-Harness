"""Bounded local text extraction for uploaded PDFs."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def extract_pdf_text(data: bytes, *, max_pages: int, max_text_bytes: int) -> str:
    if b"%PDF-" not in data[:1024]:
        raise ValueError("PDF 文件无效或损坏，请确认它是可打开的 PDF。")
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise ValueError("此 PDF 已加密；请先在本机解锁后再上传。")
        page_count = len(reader.pages)
        if page_count > max_pages:
            raise ValueError(f"PDF 共 {page_count} 页，单份附件最多支持 {max_pages} 页。")
        pages: list[str] = []
        used = 0
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            section = f"[PDF 第 {index} 页]\n{text}"
            used += len(section.encode("utf-8"))
            if used > max_text_bytes:
                raise ValueError("PDF 提取文本超过附件读取上限；请拆分 PDF 或使用较短版本。")
            pages.append(section)
    except ValueError:
        raise
    except (PdfReadError, OSError, KeyError, IndexError):
        raise ValueError("无法读取此 PDF；请检查文件是否损坏或格式受限。") from None
    if not pages:
        raise ValueError("PDF 中没有可提取的文本。扫描版或图片版 PDF 暂不支持 OCR。")
    return "\n\n".join(pages)

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from personal_harness.pdf_attachments import extract_pdf_text


def make_pdf(text: str | None, *, pages: int = 1, password: str | None = None) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        if text:
            font = writer._add_object(
                DictionaryObject(
                    {
                        NameObject("/Type"): NameObject("/Font"),
                        NameObject("/Subtype"): NameObject("/Type1"),
                        NameObject("/BaseFont"): NameObject("/Helvetica"),
                    }
                )
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def extract(data: bytes, *, max_pages: int = 100, max_text_bytes: int = 10_000) -> str:
    return extract_pdf_text(data, max_pages=max_pages, max_text_bytes=max_text_bytes)


def test_extracts_local_text_and_page_numbers():
    result = extract(make_pdf("Weekend budget"))
    assert "[PDF 第 1 页]" in result
    assert "Weekend budget" in result


def test_rejects_scanned_empty_encrypted_malformed_and_oversized_pdfs():
    with pytest.raises(ValueError, match="没有可提取的文本"):
        extract(make_pdf(None))
    with pytest.raises(ValueError, match="已加密"):
        extract(make_pdf("secret", password="pass"))
    with pytest.raises(ValueError, match="无效或损坏"):
        extract(b"not a pdf")
    with pytest.raises(ValueError, match="最多支持 1 页"):
        extract(make_pdf("page", pages=2), max_pages=1)
    with pytest.raises(ValueError, match="读取上限"):
        extract(make_pdf("a reasonably long text fragment"), max_text_bytes=8)

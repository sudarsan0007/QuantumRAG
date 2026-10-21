"""Text extraction from uploaded files: .txt, .md, .pdf and .docx.

PDF text is read page by page with pypdf so each passage can cite its page.
DOCX is read directly from the document XML, so it needs no extra package.
Scanned PDFs contain images rather than text and need OCR first; they are
reported as having no extractable text instead of failing silently.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePath
from xml.etree import ElementTree

SUPPORTED = (".txt", ".md", ".pdf", ".docx")
MAX_BYTES = 25 * 1024 * 1024


@dataclass
class Section:
    title: str
    text: str


class ExtractionError(ValueError):
    pass


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"(?<![.!?:;])\n(?=[a-z0-9(])", " ", text)  # unwrap soft line breaks
    return text.strip()


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            if enc == "utf-16" and not data[:2] in (b"\xff\xfe", b"\xfe\xff"):
                continue
            return text
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _pdf(data: bytes, stem: str) -> list[Section]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractionError("Reading PDFs needs the pypdf package. Run: pip install pypdf") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ExtractionError(f"{stem}.pdf is password protected") from exc
        out = []
        for i, page in enumerate(reader.pages, 1):
            text = _clean(page.extract_text() or "")
            if len(text) >= 20:
                out.append(Section(f"{stem}, page {i}", text))
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Could not read {stem}.pdf ({exc.__class__.__name__})") from exc
    if not out:
        raise ExtractionError(f"{stem}.pdf has no extractable text. It is probably a scanned "
                              "document; run OCR on it first and upload the result.")
    return out


def _docx(data: bytes, stem: str) -> list[Section]:
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
    except Exception as exc:
        raise ExtractionError(f"Could not read {stem}.docx; is it a valid Word file?") from exc
    paras = []
    for p in root.iter(f"{ns}p"):
        t = "".join(node.text or "" for node in p.iter(f"{ns}t")).strip()
        if t:
            paras.append(t)
    text = _clean("\n".join(paras))
    if len(text) < 20:
        raise ExtractionError(f"{stem}.docx contains no text")
    return [Section(stem, text)]


def extract(filename: str, data: bytes) -> list[Section]:
    """Return the text sections of one uploaded file."""
    path = PurePath(filename)
    ext, stem = path.suffix.lower(), path.stem or "Uploaded file"
    if len(data) > MAX_BYTES:
        raise ExtractionError(f"{filename} is larger than {MAX_BYTES // (1024 * 1024)} MB")
    if ext == ".pdf":
        return _pdf(data, stem)
    if ext == ".docx":
        return _docx(data, stem)
    if ext in (".txt", ".md"):
        text = _clean(_decode(data))
        if len(text) < 20:
            raise ExtractionError(f"{filename} contains no text")
        return [Section(stem, text)]
    raise ExtractionError(f"{filename}: unsupported type. Upload {', '.join(SUPPORTED)} files.")

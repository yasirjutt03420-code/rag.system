"""Data Loader module for loading various document formats into Document objects.
Supports: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON, Code, Web URLs, and Streamlit UploadedFile.
"""
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests

from rag.document import Document

# Optional format parsers with graceful fallbacks
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pptx
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class DataLoader:
    """Loads documents from files, byte buffers, URLs, and directory paths."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".pptx", ".ppt",
        ".xlsx", ".xls", ".csv",
        ".txt", ".md", ".json", ".py", ".html", ".htm",
        ".xml", ".yaml", ".yml", ".log", ".rst"
    }

    @classmethod
    def load_file(cls, file_input: Union[str, Path, Any], filename: Optional[str] = None) -> List[Document]:
        """Load a file given a file path or file-like object (e.g. Streamlit UploadedFile).

        Args:
            file_input: File path (str/Path) or file-like object with .read()/.getvalue()
            filename: Optional filename override if file_input is a buffer

        Returns:
            List of Document objects
        """
        # Determine filename and extension
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            ext = path.suffix.lower()
            name = path.name
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "rb") as f:
                content_bytes = f.read()
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            name = filename or getattr(file_input, "name", "document.txt")
            ext = Path(name).suffix.lower()
            if hasattr(file_input, "getvalue"):
                content_bytes = file_input.getvalue()
            elif hasattr(file_input, "read"):
                content_bytes = file_input.read()
            else:
                raise ValueError("Unsupported file_input type")

        return cls._parse_bytes(content_bytes, name, ext)

    @classmethod
    def _parse_bytes(cls, data: bytes, filename: str, ext: str) -> List[Document]:
        """Dispatch parsing based on file extension."""
        metadata_base = {
            "source": filename,
            "filename": filename,
            "file_type": ext.lstrip("."),
            "byte_size": len(data),
        }

        if ext == ".pdf":
            return cls._parse_pdf(data, metadata_base)
        elif ext in [".docx", ".doc"]:
            return cls._parse_docx(data, metadata_base)
        elif ext in [".pptx", ".ppt"]:
            return cls._parse_pptx(data, metadata_base)
        elif ext in [".xlsx", ".xls"]:
            return cls._parse_excel(data, metadata_base)
        elif ext == ".csv":
            return cls._parse_csv(data, metadata_base)
        elif ext == ".json":
            return cls._parse_json(data, metadata_base)
        else:
            # Default text handler
            return cls._parse_text(data, metadata_base)

    @classmethod
    def _parse_pdf(cls, data: bytes, metadata_base: Dict[str, Any]) -> List[Document]:
        if not PYPDF_AVAILABLE:
            text = cls._decode_text(data)
            return [Document(page_content=text, metadata=metadata_base)]

        docs = []
        try:
            reader = pypdf.PdfReader(BytesIO(data))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    meta = {**metadata_base, "page": i + 1, "total_pages": len(reader.pages)}
                    docs.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            text = cls._decode_text(data)
            docs.append(Document(page_content=f"[PDF extraction warning: {e}]\n{text}", metadata=metadata_base))

        if not docs:
            docs.append(Document(page_content="(Empty PDF document)", metadata=metadata_base))
        return docs

    @classmethod
    def _parse_docx(cls, data: bytes, metadata_base: Dict[str, Any]) -> List[Document]:
        if not DOCX_AVAILABLE:
            text = cls._decode_text(data)
            return [Document(page_content=text, metadata=metadata_base)]

        try:
            doc = docx.Document(BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Also extract tables
            table_texts = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        table_texts.append(row_text)

            full_text = "\n\n".join(paragraphs)
            if table_texts:
                full_text += "\n\n[Tables]:\n" + "\n".join(table_texts)

            return [Document(page_content=full_text, metadata=metadata_base)]
        except Exception as e:
            return [Document(page_content=f"[DOCX Parse Error: {e}]", metadata=metadata_base)]

    @classmethod
    def _parse_pptx(cls, data: bytes, metadata_base: Dict[str, Any]) -> List[Document]:
        if not PPTX_AVAILABLE:
            return [Document(page_content=cls._decode_text(data), metadata=metadata_base)]

        try:
            prs = pptx.Presentation(BytesIO(data))
            docs = []
            for i, slide in enumerate(prs.slides):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        slide_text.append(shape.text.strip())
                text = "\n".join(slide_text).strip()
                if text:
                    meta = {**metadata_base, "slide": i + 1, "total_slides": len(prs.slides)}
                    docs.append(Document(page_content=text, metadata=meta))
            return docs or [Document(page_content="(Empty Presentation)", metadata=metadata_base)]
        except Exception as e:
            return [Document(page_content=f"[PPTX Parse Error: {e}]", metadata=metadata_base)]

    @classmethod
    def _parse_excel(cls, data: bytes, metadata_base: Dict[str, Any]) -> List[Document]:
        if not PANDAS_AVAILABLE:
            return [Document(page_content=cls._decode_text(data), metadata=metadata_base)]

        try:
            excel_file = pd.ExcelFile(BytesIO(data))
            docs = []
            for sheet in excel_file.sheet_names:
                df = excel_file.parse(sheet)
                text = df.to_string(index=False)
                meta = {**metadata_base, "sheet_name": sheet}
                docs.append(Document(page_content=f"Sheet: {sheet}\n\n{text}", metadata=meta))
            return docs
        except Exception as e:
            return [Document(page_content=f"[Excel Parse Error: {e}]", metadata=metadata_base)]

    @classmethod
    def _parse_csv(cls, data: bytes, metadata_base: Dict[str, Any]) -> List[Document]:
        if PANDAS_AVAILABLE:
            try:
                df = pd.read_csv(BytesIO(data))
                text = df.to_string(index=False)
                return [Document(page_content=text, metadata=metadata_base)]
            except Exception:
                pass
        return [Document(page_content=cls._decode_text(data), metadata=metadata_base)]

    @classmethod
    def _parse_json(cls, data: bytes, metadata_base: Dict[str, Any]) -> List[Document]:
        text = cls._decode_text(data)
        try:
            obj = json.loads(text)
            formatted = json.dumps(obj, indent=2, ensure_ascii=False)
            return [Document(page_content=formatted, metadata=metadata_base)]
        except Exception:
            return [Document(page_content=text, metadata=metadata_base)]

    @classmethod
    def _parse_text(cls, data: bytes, metadata_base: Dict[str, Any]) -> List[Document]:
        text = cls._decode_text(data)
        return [Document(page_content=text, metadata=metadata_base)]

    @staticmethod
    def _decode_text(data: bytes) -> str:
        """Robustly decode byte data using common encodings."""
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @classmethod
    def load_text(cls, text: str, source_name: str = "direct_input") -> List[Document]:
        """Wrap raw text in a Document."""
        return [
            Document(
                page_content=text.strip(),
                metadata={
                    "source": source_name,
                    "filename": source_name,
                    "file_type": "text",
                    "byte_size": len(text.encode("utf-8")),
                },
            )
        ]

    @classmethod
    def load_url(cls, url: str) -> List[Document]:
        """Fetch and extract text content from a web URL."""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        # Simple HTML tag stripping
        html = resp.text
        import re
        clean_text = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r"<style.*?</style>", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r"<[^>]+>", " ", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        
        return [
            Document(
                page_content=clean_text,
                metadata={
                    "source": url,
                    "filename": url,
                    "file_type": "url",
                    "status_code": resp.status_code,
                },
            )
        ]

    @classmethod
    def load_directory(cls, dir_path: Union[str, Path], recursive: bool = True) -> List[Document]:
        """Scan directory and load all supported documents."""
        dir_p = Path(dir_path)
        if not dir_p.is_dir():
            raise NotADirectoryError(f"Not a valid directory: {dir_path}")

        pattern = "**/*" if recursive else "*"
        all_docs = []
        for file_p in dir_p.glob(pattern):
            if file_p.is_file() and file_p.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                try:
                    all_docs.extend(cls.load_file(file_p))
                except Exception as e:
                    print(f"Skipping {file_p} due to load error: {e}")
        return all_docs


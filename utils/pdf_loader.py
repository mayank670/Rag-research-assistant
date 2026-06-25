"""
PDF Loader Utility
Extracts text from PDF files and splits into chunks for embedding.
"""

import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file, returning a list of page-level dicts.
    Tries PyMuPDF first, falls back to PyPDF2.
    """
    pages = []

    # Try PyMuPDF (fitz) first — better text extraction
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "text": text,
                    "page": page_num + 1,
                    "source": os.path.basename(file_path),
                })
        doc.close()
        logger.info(f"PyMuPDF extracted {len(pages)} pages from {file_path}")
        return pages

    except ImportError:
        logger.warning("PyMuPDF not available, falling back to PyPDF2")

    # Fallback: PyPDF2
    try:
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append({
                        "text": text,
                        "page": page_num + 1,
                        "source": os.path.basename(file_path),
                    })
        logger.info(f"PyPDF2 extracted {len(pages)} pages from {file_path}")
        return pages

    except ImportError:
        raise ImportError("Install PyMuPDF or PyPDF2: pip install pymupdf PyPDF2")


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split page texts into overlapping chunks suitable for embedding.
    Each chunk carries its source filename and page number.
    """
    chunks = []

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        source = page_data["source"]

        # Slide a window across the page text
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "source": source,
                    "chunk_index": len(chunks),
                })

            if end >= len(text):
                break
            start = end - chunk_overlap  # overlap

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def process_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Full pipeline: extract pages → split into chunks."""
    pages = extract_text_from_pdf(file_path)
    if not pages:
        raise ValueError(f"No text could be extracted from {file_path}. The PDF may be scanned or image-based.")
    return chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

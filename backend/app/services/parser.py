"""
Parses PDF into text blocks containing page number, text and character offsets.
Uses PyMuPDF (fitz) because it's fast and preserves order.
"""
from typing import List, Dict
import fitz # PyMuPDF


def parse_pdf_to_text_blocks(pdf_path: str) -> List[Dict]:
    """Return list of blocks: {page:int, text:str}
    We keep text as-is and let chunker decide granularity.
    """
    doc = fitz.open(pdf_path)
    blocks = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        # normalize newlines, trim
        text = text.replace('\r\n', '\n')
        if text.strip():
            blocks.append({"page": page_idx + 1, "text": text})
    doc.close()
    return blocks
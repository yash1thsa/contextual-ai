"""
Chunk text blocks with options for letter/word-aware chunking.
This implementation is simple and deterministic. It emits chunks with
start/end indices (character offsets) relative to the block text.
"""

from typing import List, Dict
import os
import re
import logging

# --- Logging setup ---
logger = logging.getLogger("chunker")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# --- Chunking configuration ---
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))

WORD_BOUNDARY = re.compile(r"\s+")


def chunk_text_blocks(blocks: List[Dict]) -> List[Dict]:
    chunks = []
    logger.info(f"Starting chunking {len(blocks)} text blocks with CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")

    for b_idx, b in enumerate(blocks, start=1):
        text = b['text']
        page = b['page']
        length = len(text)
        start = 0
        block_chunks = 0

        logger.debug(f"Processing block {b_idx} on page {page} (length={length})")

        while start < length:
            end = min(start + CHUNK_SIZE, length)

            # Try to respect word boundaries
            if end < length:
                ws_back = text.rfind(' ', start, end)
                ws_forward = text.find(' ', end, min(length, end + 20))
                if ws_back != -1 and (end - ws_back) < (ws_forward - end if ws_forward != -1 else 9999):
                    end = ws_back
                elif ws_forward != -1:
                    end = ws_forward

            chunk_text = text[start:end].strip()
            if not chunk_text:
                logger.debug(f"Skipping empty chunk at start={start}, end={end}")
                start = end
                continue

            chunks.append({
                'page': page,
                'start': start,
                'end': end,
                'text': chunk_text
            })
            block_chunks += 1

            logger.debug(f"Created chunk {block_chunks} for block {b_idx}: start={start}, end={end}, len={len(chunk_text)}")

            # Move start forward with overlap
            start = max(end - CHUNK_OVERLAP, end) if CHUNK_OVERLAP > 0 else end

        logger.info(f"Block {b_idx} on page {page} generated {block_chunks} chunks")

    logger.info(f"Chunking complete: total {len(chunks)} chunks generated")
    return chunks

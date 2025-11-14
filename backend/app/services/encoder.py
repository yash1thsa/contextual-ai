"""
Provides `encode_chunks(list[str]) -> list[list[float]]`.

Supports multiple backends:
  - Hugging Face local model (default)
  - SBERT (local)
  - OpenAI API
  - Ollama local embedding API
"""

import os
from typing import List
import requests
import logging
from huggingface_hub import InferenceClient

# Backend selection: hf | sbert | openai | ollama
ENCODER_BACKEND = os.getenv("ENCODER_BACKEND", "hf")
logger = logging.getLogger("pdf_upload")
logger.setLevel(logging.DEBUG)


# Model names
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SBERT_MODEL = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")


def encode_chunks(texts: List[str]) -> List[List[float]]:
    """
    Encode text chunks using the selected backend.
    """
    backend = ENCODER_BACKEND.lower()
    logger.info(f"Backend found: {backend}")

    if backend == "openai":
        return _encode_openai(texts)
    elif backend == "sbert":
        return _encode_sbert(texts)
    elif backend == "ollama":
        return _encode_ollama(texts)
    elif backend == "hf":
        return _encode_hf_remote(texts)
    else:
        raise ValueError(f"Unknown ENCODER_BACKEND: {backend}")


# --------------------------------------------------------------------
# Hugging Face embedding
# --------------------------------------------------------------------
def _encode_hf_remote(texts: List[str]) -> List[List[float]]:
    from huggingface_hub import InferenceClient

    HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")

    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN environment variable not set")

    logger = logging.getLogger(__name__)
    client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")

    embeddings = []

    logger.info(f"Encoding {len(texts)} texts using HF model '{HF_EMBED_MODEL}'...")

    for text in texts:
        try:
            # Call HF embedding API
            response = client.feature_extraction(model=HF_EMBED_MODEL, inputs=text)

            # Validate structure
            if (
                response is None
                or len(response) == 0
                or response[0] is None
                or len(response[0]) == 0
            ):
                raise RuntimeError("HF returned empty embedding")

            # Use the inner vector
            vector = response[0]

            embeddings.append(vector)

        except Exception as e:
            logger.error(f"Failed to encode text '{text[:30]}...': {e}")
            raise

    logger.info("HF Remote embeddings completed.")
    return embeddings

# --------------------------------------------------------------------
# SBERT local embedding
# --------------------------------------------------------------------
def _encode_sbert(texts: List[str]):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


# --------------------------------------------------------------------
# OpenAI embeddings
# --------------------------------------------------------------------
def _encode_openai(texts: List[str]):
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=key)
    vectors = []
    for t in texts:
        resp = client.embeddings.create(model="text-embedding-3-large", input=t)
        vectors.append(resp.data[0].embedding)
    return vectors


# --------------------------------------------------------------------
# Ollama embeddings
# --------------------------------------------------------------------
def _encode_ollama(texts: List[str]):
    model = OLLAMA_MODEL
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")

    vectors = []
    for text in texts:
        payload = {"model": model, "prompt": text}
        try:
            response = requests.post(f"{endpoint}/api/embeddings", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            vectors.append(data["embedding"])
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")

    return vectors

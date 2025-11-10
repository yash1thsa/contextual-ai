"""
Provides `encode_chunks(list[str]) -> list[list[float]]`.

Supports multiple backends:
  - Hugging Face Inference API (default)
  - SBERT (local)
  - OpenAI API
  - Ollama local embedding API
"""

import os
from typing import List
import requests

# Backend selection: hf | sbert | openai | ollama
ENCODER_BACKEND = os.getenv("ENCODER_BACKEND", "hf")

# Model names
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "nomic-ai/nomic-embed-text-v2-moe")
SBERT_MODEL = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")


def encode_chunks(texts: List[str]) -> List[List[float]]:
    """
    Encode text chunks using the selected backend.
    """
    backend = ENCODER_BACKEND.lower()

    if backend == "openai":
        return _encode_openai(texts)
    elif backend == "sbert":
        return _encode_sbert(texts)
    elif backend == "ollama":
        return _encode_ollama(texts)
    elif backend == "hf":
        return _encode_hf(texts)
    else:
        raise ValueError(f"Unknown ENCODER_BACKEND: {backend}")


# --------------------------------------------------------------------
# Hugging Face Inference API (requires HF_TOKEN)
# --------------------------------------------------------------------
def _encode_hf(texts: List[str]) -> List[List[float]]:
    """
    Encodes text using Hugging Face Inference API.
    Requires `HF_TOKEN` environment variable.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set — required for Hugging Face API access.")

    model_name = HF_EMBED_MODEL
    endpoint = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    vectors = []
    for text in texts:
        payload = {"inputs": text}
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            # HF API responses vary slightly depending on model
            if isinstance(data, list) and isinstance(data[0], list):
                vectors.append(data[0])
            elif isinstance(data, dict) and "embedding" in data:
                vectors.append(data["embedding"])
            else:
                raise ValueError(f"Unexpected HF API response: {data}")
        except Exception as e:
            raise RuntimeError(f"Hugging Face embedding failed: {e}")

    return vectors


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

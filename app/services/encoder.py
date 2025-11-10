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
import torch

# Backend selection: hf | sbert | openai | ollama
ENCODER_BACKEND = os.getenv("ENCODER_BACKEND", "hf")

# Model names
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
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
        return _encode_hf_local(texts)
    else:
        raise ValueError(f"Unknown ENCODER_BACKEND: {backend}")


# --------------------------------------------------------------------
# Hugging Face LOCAL embedding
# --------------------------------------------------------------------
def _encode_hf_local(texts: List[str]) -> List[List[float]]:
    """
    Encode text locally using a Hugging Face transformer model.
    Works offline — no token or API calls needed.
    """
    from transformers import AutoTokenizer, AutoModel

    model_name = HF_EMBED_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        embeddings.append(emb)

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

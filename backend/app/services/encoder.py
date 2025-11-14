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
from huggingface_hub import InferenceClient, InferenceTimeoutError

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
    """
    Generates embeddings for a list of text chunks using the
    huggingface_hub.InferenceClient.

    The client automatically uses the HF_API_TOKEN environment variable.

    Args:
        texts: A list of string chunks to embed.

    Returns:
        A list of lists, where each inner list is a float vector embedding
        corresponding to an input text chunk.

    Raises:
        RuntimeError: If the HF_API_TOKEN environment variable is not set or the API fails.
        InferenceTimeoutError: If the API request times out.
    """
    # Retrieve API Token and Model Name from environment variables
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    if not HF_API_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN environment variable not set. Please obtain a token from Hugging Face settings.")

    client = InferenceClient(token=HF_API_TOKEN)

    try:
        embeddings = client.feature_extraction(
            model=HF_EMBED_MODEL,
            text=texts
        )

        return embeddings

    except InferenceTimeoutError:
        print("The request timed out while generating embeddings.")
        raise RuntimeError("Hugging Face API request timed out.")
    except Exception as e:
        print(f"An unexpected error occurred with the InferenceClient: {e}")
        raise RuntimeError(f"API interaction failed: {e}")


# Example Usage:
# Make sure to set your environment variable first:
# os.environ["HF_API_TOKEN"] = "hf_YOUR_TOKEN_HERE"
# os.environ["HF_EMBED_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
# texts_to_embed = ["Example sentence one.", "Example sentence two is slightly longer."]
# embeddings = encode_chunks_with_client(texts_to_embed)
# print(f"Generated {len(embeddings)} embeddings.")

#------------------------------------------------------------
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

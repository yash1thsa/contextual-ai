"""
Provides `encode_chunks(list[str]) -> list[list[float]]`.
Supports local SBERT model or OpenAI embeddings via environment.
"""
import os
from typing import List
import requests

ENCODER_BACKEND = os.getenv('ENCODER_BACKEND', 'sbert')
SBERT_MODEL = os.getenv('SBERT_MODEL', 'all-MiniLM-L6-v2')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')

def encode_chunks(texts: List[str]) -> List[List[float]]:
    if ENCODER_BACKEND == 'openai':
        return _encode_openai(texts)
    else:
        return _encode_sbert(texts)


# Local sentence-transformers encoder
def _encode_sbert(texts: List[str]):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


# OpenAI encoder (if chosen)
def _encode_openai(texts: List[str]):
    import os
    from openai import OpenAI
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set')
    client = OpenAI(api_key=key)
    vectors = []
    # batch naively; for high throughput add batching and rate control
    for t in texts:
        resp = client.embeddings.create(model='text-embedding-3-large', input=t)
        vectors.append(resp.data[0].embedding)
    return vectors


def _encode_ollama(texts: List[str]):
    model = OLLAMA_MODEL
    endpoint = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')

    vectors = []
    for text in texts:
        payload = {
        "model": model,
        "prompt": text
        }
        try:
            response = requests.post(f"{endpoint}/api/embeddings", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            vectors.append(data['embedding'])
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed for text chunk: {e}")
    return vectors
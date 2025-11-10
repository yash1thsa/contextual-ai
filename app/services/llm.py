import os
import logging
from typing import List
import requests

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger("llm")
logger.setLevel(logging.INFO)

# ---------------- ENV CONFIG ----------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()  # default: local
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "phi3:mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load local embedding model once
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
logger.info(f"Loading local SentenceTransformer model: {LOCAL_MODEL_NAME}")
_local_model = SentenceTransformer(LOCAL_MODEL_NAME)


# ---------- Local MiniLM Embedding + Similarity ----------
def _answer_local(query: str, context_chunks: List[dict]) -> str:
    """
    Finds the most semantically similar chunk(s) using a local SBERT model
    and returns a concise answer.
    """
    if not context_chunks:
        return "No context provided."

    # Encode query + all context chunks
    texts = [c["text"] for c in context_chunks]
    query_emb = _local_model.encode(query, convert_to_tensor=True)
    context_embs = _local_model.encode(texts, convert_to_tensor=True)

    # Compute similarity scores
    scores = util.cos_sim(query_emb, context_embs)[0]
    top_idx = int(scores.argmax())
    best_chunk = context_chunks[top_idx]

    answer = (
        f"Based on page {best_chunk.get('page', '?')}, "
        f"the most relevant section says:\n\n{best_chunk['text']}\n\n"
        "This section seems most related to your question."
    )
    return answer


# ---------- OpenAI ----------
def _answer_openai(prompt: str, temperature: float = 0.0) -> str:
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=800,
    )
    return resp.choices[0].message.content


# ---------- Ollama ----------
def _answer_ollama(prompt: str, temperature: float = 0.0) -> str:
    url = f"{OLLAMA_ENDPOINT}/v1/completions"
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "max_tokens": 800,
        "temperature": temperature
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Ollama server not reachable at {OLLAMA_ENDPOINT}. "
                           f"Run `ollama serve` in another terminal.")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    data = resp.json()
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0].get("text", "").strip()
    else:
        logger.warning(f"Unexpected Ollama response: {data}")
        return str(data)


# ---------- Combined ----------
def answer_with_context(query: str, context_chunks: List[dict]) -> str:
    """
    Chooses backend (local, ollama, openai) to generate or retrieve an answer.
    """
    logger.info(f"Calling LLM backend={LLM_BACKEND}")

    if LLM_BACKEND == "ollama":
        # Use a local LLM running in Ollama
        context_text = "\n\n---\n\n".join(
            [f"Page {c['page']}:\n{c['text']}" for c in context_chunks]
        )
        prompt = (
            "You are a helpful assistant that answers questions using only the provided context. "
            "If the answer is not in the context, reply 'I don't know.' "
            "Always mention the page number when referencing the document.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        )
        return _answer_ollama(prompt)

    elif LLM_BACKEND == "openai":
        # Use OpenAI GPT for contextual reasoning
        context_text = "\n\n---\n\n".join(
            [f"Page {c['page']}:\n{c['text']}" for c in context_chunks]
        )
        prompt = (
            "You are a helpful assistant that answers questions using only the provided context. "
            "If the answer is not in the context, reply 'I don't know.' "
            "Always mention the page number when referencing the document.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        )
        return _answer_openai(prompt)

    else:
        # Default: simple semantic retrieval-based answer
        return _answer_local(query, context_chunks)

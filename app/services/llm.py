# app/services/llm.py
import os
import logging
import requests
from typing import List

logger = logging.getLogger("llm")
logger.setLevel(logging.INFO)

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "mistral")  # default model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------- OpenAI ----------
def _answer_openai(prompt: str, temperature: float = 0.0) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed or import failed") from e

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
    """
    Calls Ollama's v1 completions endpoint.
    """
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
    # Ollama v1 returns {"id":..., "object":..., "choices":[{"text": "..."}]}
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0].get("text", "").strip()
    else:
        logger.warning(f"Unexpected Ollama response: {data}")
        return str(data)


# ---------- Combined Answer ----------
def answer_with_context(query: str, context_chunks: List[dict]) -> str:
    """
    Build a prompt using top-K retrieved context chunks and query the LLM backend.
    """
    context_text = "\n\n---\n\n".join(
        [f"Page {c['page']}:\n{c['text']}" for c in context_chunks]
    )

    prompt = (
        "You are a helpful assistant that answers questions using only the provided context. "
        "If the answer is not in the context, reply 'I don't know.' "
        "Always mention the page number when referencing the document.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    )

    logger.info(f"Calling LLM backend={LLM_BACKEND} model={OLLAMA_LLM_MODEL}")
    if LLM_BACKEND == "ollama":
        return _answer_ollama(prompt)
    else:
        return _answer_openai(prompt)

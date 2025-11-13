import os
import logging
import requests
from typing import List
from huggingface_hub import InferenceClient

logger = logging.getLogger("llm")
logger.setLevel(logging.INFO)

# ---------------- ENV CONFIG ----------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "phi3:mini")  # default model

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # optional, for Hugging Face models
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-2-7b-chat-hf")  # default HF model

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


# ---------- Hugging Face ----------
def _answer_huggingface(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """
    Calls Hugging Face Inference API via InferenceClient.
    Supports hosted models that allow chat/text generation.
    """
    HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")

    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN environment variable not set")

    try:
        # Initialize the client
        client = InferenceClient(
            api_key=HF_API_TOKEN,
            provider="auto"  # automatically selects best provider
        )

        # Chat completion
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Extract the response text
        answer = completion.choices[0].message["content"]
        return answer.strip()

    except Exception as e:
        logger.error(f"Hugging Face API request failed: {e}")
        raise RuntimeError(f"Hugging Face API request failed: {e}")

# ---------- Combined ----------
def answer_with_context(query: str, context_chunks: List[dict]) -> str:
    context_text = "\n\n---\n\n".join(
        [f"Page {c['page']}:\n{c['text']}" for c in context_chunks]
    )

    prompt = (
        "You are a helpful assistant that answers questions using only the provided context. "
        "If the answer is not in the context, reply 'I don't know.' "
        "Always mention the page number when referencing the document.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    )

    logger.info(f"Calling LLM backend={LLM_BACKEND}")
    if LLM_BACKEND == "ollama":
        return _answer_ollama(prompt)
    elif LLM_BACKEND == "huggingface":
        return _answer_huggingface(prompt)
    else:
        return _answer_openai(prompt)

import os
import streamlit as st
import requests
import json
from typing import List, Dict, Any

# ----------------- Streamlit Config -----------------
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Chat-to-URL", layout="centered")

# ----------------- Utility functions -----------------
def post_question(
    url: str,
    question_field: str,
    question: str,
    history: List[Dict[str, Any]],
    headers: dict,
    timeout: int = 30,
) -> dict:
    payload = {question_field: question, "history": history}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        return {"ok": False, "status_code": None, "response": f"Request error: {e}"}

    ct = resp.headers.get("content-type", "")
    try:
        if "application/json" in ct:
            return {"ok": resp.ok, "status_code": resp.status_code, "response": resp.json()}
        else:
            return {"ok": resp.ok, "status_code": resp.status_code, "response": resp.text}
    except Exception:
        return {"ok": resp.ok, "status_code": resp.status_code, "response": resp.text}

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "url" not in st.session_state:
        st.session_state.url = ""
    if "token" not in st.session_state:
        st.session_state.token = ""
    if "question_field" not in st.session_state:
        st.session_state.question_field = "question"
    if "_extra_headers" not in st.session_state:
        st.session_state._extra_headers = {}

# ----------------- Layout -----------------
init_state()
st.title("💬 Chat-to-URL — Streamlit Chat Client")

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.session_state.url = st.text_input("Target URL", st.session_state.url,
                                         placeholder="https://example.com/api/chat")
    st.session_state.token = st.text_input("Bearer token (optional)", st.session_state.token,
                                           type="password")
    st.session_state.question_field = st.text_input("Question JSON field name", st.session_state.question_field)
    st.write("Extra request headers (JSON)")
    extra_headers_text = st.text_area("Extra headers",
                                      value=json.dumps(st.session_state._extra_headers, indent=2),
                                      help="Example: {\"X-My-Header\": \"value\"}")
    try:
        st.session_state._extra_headers = json.loads(extra_headers_text) if extra_headers_text.strip() else {}
    except Exception:
        st.error("Invalid JSON in extra headers. Please enter a valid JSON object.")
        st.session_state._extra_headers = {}

    st.markdown("---")
    st.write("Conversation controls")
    cols = st.columns([1,1,2])
    if cols[0].button("Clear chat"):
        st.session_state.history = []
    if cols[1].button("Undo last"):
        if st.session_state.history:
            st.session_state.history.pop()

# Build headers
headers = {}
if st.session_state.token:
    headers["Authorization"] = f"Bearer {st.session_state.token}"
headers.update(st.session_state._extra_headers)

# Chat rendering
def render_chat():
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Bot:** {msg['text']}")

st.markdown("### Conversation")
chat_box = st.container()
with chat_box:
    render_chat()

# Input form
st.markdown("---")
with st.form(key="chat_form", clear_on_submit=True):
    question = st.text_input("Ask a question", placeholder="Type your question here...")
    submitted = st.form_submit_button("Send")
    if submitted:
        q = question.strip()
        if not st.session_state.url:
            st.error("Please fill in the target URL in the sidebar before sending.")
        elif not q:
            st.error("Enter a non-empty question.")
        else:
            st.session_state.history.append({"role": "user", "text": q})
            simple_history = [{"role": m["role"], "text": m["text"]} for m in st.session_state.history]
            with st.spinner("Sending..."):
                result = post_question(
                    url=st.session_state.url,
                    question_field=st.session_state.question_field,
                    question=q,
                    history=simple_history,
                    headers=headers,
                )

            if not result["ok"]:
                bot_text = f"ERROR (status={result['status_code']}): {result['response']}"
            else:
                resp = result["response"]
                if isinstance(resp, dict):
                    for k in ("answer", "reply", "text", "result", "message"):
                        if k in resp:
                            bot_text = resp[k]
                            break
                    else:
                        bot_text = json.dumps(resp, indent=2)
                else:
                    bot_text = str(resp)

            st.session_state.history.append({"role": "bot", "text": bot_text})

# Re-render chat
with chat_box:
    render_chat()

# ----------------- Entrypoint for Render -----------------
if __name__ == "__main__":
    import streamlit.web.bootstrap
    streamlit.web.bootstrap.run(
        "app.py",  # this file
        "",
        [],
        None,
        False,
        False,
        port=PORT
    )
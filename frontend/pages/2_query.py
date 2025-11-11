import streamlit as st
import requests
import json

st.set_page_config(page_title="Chatbot", page_icon="💬")

st.title("💬 Chat-to-URL Chatbot")

# ----------------- Initialize session state -----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "max_results" not in st.session_state:
    st.session_state.max_results = 5  # default, can be set from sidebar

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    st.session_state.max_results = st.slider(
        "Max results per query", 1, 20, st.session_state.max_results
    )

# ----------------- Render chat history -----------------
chat_container = st.container()
with chat_container:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Bot:** {msg['text']}")

# ----------------- Chat input -----------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question...", placeholder="Type your question here...")
    submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        # Add user message to history
        st.session_state.history.append({"role": "user", "text": user_input})

        # Prepare request payload
        payload = {
            "question": user_input,
            "max_results": st.session_state.max_results,
            "history": st.session_state.history  # optional: send past messages
        }

        try:
            # Send request to backend
            resp = requests.post("http://backend:8000/query", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # Attempt to extract answer from common fields
                bot_text = data.get("answer") or data.get("reply") or data.get("text") or str(data)
            else:
                bot_text = f"❌ Query failed: {resp.text}"

        except Exception as e:
            bot_text = f"❌ Connection error: {e}"

        # Add bot message to history
        st.session_state.history.append({"role": "bot", "text": bot_text})

# Re-render chat after new message
with chat_container:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Bot:** {msg['text']}")

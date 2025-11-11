import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = "landing"

# Sidebar
with st.sidebar:
    st.button("Landing", on_click=lambda: st.session_state.update({"page": "landing"}))
    st.button("Upload PDFs", on_click=lambda: st.session_state.update({"page": "upload"}))
    st.button("Chatbot", on_click=lambda: st.session_state.update({"page": "chat"}))

# Main content
if st.session_state.page == "landing":
    st.title("Welcome")
elif st.session_state.page == "upload":
    st.title("Upload PDFs")
elif st.session_state.page == "chat":
    st.title("Chatbot")

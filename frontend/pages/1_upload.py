import streamlit as st
import requests
import os

st.set_page_config(page_title="Upload PDFs", page_icon="📄")

st.title("📄 Upload PDFs")

# Sidebar: optional max_results setting (shared with Chat page)
if "max_results" not in st.session_state:
    st.session_state.max_results = 5

with st.sidebar:
    st.header("Settings")
    st.session_state.max_results = st.slider(
        "Max results per query", 1, 20, st.session_state.max_results
    )

# File uploader
uploaded_files = st.file_uploader(
    "Select PDF files to upload", type=["pdf"], accept_multiple_files=True
)

# Upload button
if st.button("Upload PDFs"):

    if not uploaded_files:
        st.warning("Please select at least one PDF file.")
    else:
        for pdf_file in uploaded_files:
            try:
                # Send file to backend
                files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
                CONTEXTUAL_API = os.getenv("CONTEXTUAL_API", "https://contextual-ai-cw79.onrender.com/")
                resp = requests.post(f"{CONTEXTUAL_API}/upload/pdf", files=files, timeout=60)

                if resp.status_code == 200:
                    st.success(f"✅ Successfully uploaded {pdf_file.name}")
                else:
                    st.error(f"❌ Failed to upload {pdf_file.name}: {resp.text}")

            except Exception as e:
                st.error(f"❌ Error uploading {pdf_file.name}: {e}")

import streamlit as st
import requests

st.title("📤 File Upload Client")

# Input for your upload target
url = st.text_input("Enter the target upload URL")

uploaded_file = st.file_uploader("Choose a file", type=None)

if uploaded_file is not None:
    st.write(f"Selected file: {uploaded_file.name}")

    if st.button("Upload"):
        if not url:
            st.error("Please enter a target URL.")
        else:
            try:
                # Send multipart/form-data POST
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(url, files=files)

                if response.status_code == 200:
                    st.success("✅ File uploaded successfully!")
                    st.json(response.json() if response.headers.get("content-type") == "application/json" else response.text)
                else:
                    st.error(f"❌ Upload failed with status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"An error occurred: {e}")

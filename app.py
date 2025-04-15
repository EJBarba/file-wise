import streamlit as st
import PyPDF2
import requests

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 Ask Questions About Your PDF")

# --- Function to extract text from uploaded PDF ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# --- Hugging Face Inference API Call ---
def query_huggingface(context, question, hf_token):
    API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        st.error("❌ Failed to parse Hugging Face API response.")
        st.text("Raw response:")
        st.text(response.text)
        return {}

# --- File uploader ---
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# --- Main Logic ---
if pdf_file:
    context = extract_text_from_pdf(pdf_file)
    
    if not context.strip():
        st.error("❌ No text could be extracted from this PDF.")
    else:
        st.success("✅ PDF text extracted. You can now ask a question.")

        question = st.text_input("❓ Ask a question about the PDF:")

        if st.button("Get Answer") and question:
            try:
                hf_token = st.secrets["hf_token"]
                result = query_huggingface(context, question, hf_token)

                if "answer" in result and result["answer"].strip():
                    st.markdown(f"💬 **Answer:** {result['answer']}")
                else:
                    st.warning("⚠️ No clear answer found.")
            except KeyError:
                st.error("❌ Hugging Face API token not found in secrets.toml.")
else:
    st.info("👆 Please upload a PDF file to begin.")

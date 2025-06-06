import streamlit as st  # type: ignore
from PyPDF2 import PdfReader  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
import google.generativeai as genai  # type: ignore
# from langchain.vectorstores import FAISS  # type: ignore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain.chains.question_answering import load_qa_chain  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
import docx  # python-docx for .docx files
import chardet  # for encoding detection in .txt, .asc
import PyPDF2

# Streamlit page configuration
st.set_page_config(page_title="Document Genie", layout="wide")

# Sidebar Instructions
st.sidebar.markdown("""
## Document Genie: Instant Insights from Your Documents

This chatbot uses Google's Generative AI (Gemini-PRO) to process and analyze uploaded PDFs for quick insights.

### How to Use:
1. Upload your PDF documents.
2. Submit and ask questions for precise answers based on the content.
""")

# API key setup
api_key = "AIzaSyDlY83MSLwjB-4xj1mgHlzjH-_Kqbj6Dxw"

def extract_text_from_file(uploaded_file):
    """Extract text from supported uploaded files."""
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text

        elif file_name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])
            return text

        elif file_name.endswith(('.txt', '.asc', '.rtf')):
            raw_data = uploaded_file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            text = raw_data.decode(encoding)
            return text

        else:
            st.warning(f"Unsupported file format: {file_name}")
            return None

    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return None

def get_text_from_uploaded_files(uploaded_files):
    """Extract and combine text from all uploaded files."""
    combined_text = ""
    for uploaded_file in uploaded_files:
        text = extract_text_from_file(uploaded_file)
        if text:
            combined_text += text + "\n"
    return combined_text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # smaller chunks
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    """Create and save a FAISS vector store from text chunks."""
    # Filter out empty and very large chunks
    filtered_chunks = [chunk for chunk in text_chunks if chunk.strip() != "" and len(chunk) < 6000]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(filtered_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create a conversational chain using a custom prompt."""
    prompt_template = """
    Answer the question as accurately as possible based on the provided context. If the answer is not available, respond with:
    "Answer is not available in the context."

    Context: {context}
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        temperature=0.3,
        google_api_key=api_key,
        api_version="v1"  # <-- Important addition
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, api_key):
    """Process user input and generate a response."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: \n", response["output_text"])
    except Exception as e:
        st.error(f"Error processing your request: {e}")

def main():
    """Main Streamlit app logic."""
    st.header("Dr. Genie 💁")

    # User question input
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question and api_key:
        with st.spinner("Generating answer..."):
            user_input(user_question, api_key)

    # Sidebar file upload and processing
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not api_key:
                st.error("Please enter your Google API Key.")
            elif not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing documents..."):
                    raw_text = get_text_from_uploaded_files(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks, api_key)
                        st.success("Processing complete! You can now ask questions.")
                    else:
                        st.error("No readable text found in the uploaded files.")

if __name__ == "__main__":
    main()
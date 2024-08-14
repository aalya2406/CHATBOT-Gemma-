import os
import streamlit as st  # type: ignore
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import time
import fitz  # PyMuPDF

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if keys are properly loaded
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment")

# Set environment variable
os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

class SimpleDocument:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    pdf_text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            pdf_text += page.get_text()
    return pdf_text

def vector_embedding(file_path):
    if not file_path:
        st.error("No file selected.")
        return

    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
    
    # Extract and process the selected PDF file
    pdf_text = extract_text_from_pdf(file_path)
    if not pdf_text:
        st.error("No text extracted from the PDF.")
        return

    # Wrap the text in SimpleDocument objects with empty metadata
    st.session_state.docs = [SimpleDocument(pdf_text, metadata={'source': file_path})]
    
    # Split documents and check if any documents were split
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    if not st.session_state.final_documents:
        st.error("No documents were split from the input.")
        return
    
    # Check if the documents are correctly split and not empty
    if any(len(doc.page_content) == 0 for doc in st.session_state.final_documents):
        st.error("Some split documents are empty.")
        return

    try:
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings
    except Exception as e:
        st.error(f"An error occurred while creating the FAISS index: {e}")

# Path to the folder containing the PDFs
pdf_folder_path = "./EB policy Copy"

# List PDF files in the folder
pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

# User selects a PDF file
selected_file = st.selectbox("Select a PDF file:", pdf_files)

if selected_file:
    file_path = os.path.join(pdf_folder_path, selected_file)
    if st.button("Documents Embedding"):
        vector_embedding(file_path)
        st.write("Vector Store DB Is Ready")

prompt1 = st.text_input("Enter Your Question")

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            if "context" in response:
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No context available in the response.")
    else:
        st.write("No vector store found. Please select and process a PDF file.")

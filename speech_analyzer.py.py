import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint

# Global variables
llm_hub = None
embeddings = None
vector_store = None
qa_chain = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def init_llm():
    global llm_hub, embeddings

    logger.info("Initializing HuggingFaceEndpoint LLM with Llama 4 Scout...")

    # Updated to use Llama 4 Scout via Hugging Face Hub
    MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    # Initialize Hugging Face Endpoint using your API Token
    llm_hub = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token="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"  # Replace with your actual HF token
    )
    logger.debug("HuggingFaceEndpoint LLM initialized successfully.")

    # Initialize local Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )
    logger.debug("Embeddings initialized with device: %s", DEVICE)

def process_document(file_path):
    global vector_store, qa_chain
    logger.info("Processing document: %s", file_path)

    # Load PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Create Chroma vector store
    vector_store = Chroma.from_documents(texts, embeddings)
    logger.debug("Vector store created successfully.")

    # Create Retrieval QA chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    logger.info("QA chain initialized successfully.")

def ask_question(query):
    if not qa_chain:
        return {"error": "QA chain not initialized. Please upload a document first."}
    
    logger.info("Asking query: %s", query)
    response = qa_chain.invoke({"query": query})
    return response

if __name__ == "__main__":
    init_llm()
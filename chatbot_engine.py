import os
import pickle
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain

# Load documents from file paths
def load_documents(doc_paths):
    documents = []
    for path in doc_paths:
        print(f"Loading: {path}")
        loader = TextLoader(path, encoding='utf-8')
        documents.extend(loader.load())
    return documents

# Create or load a FAISS vectorstore
def get_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    return vectorstore

# Create LangChain's Q&A chat chain
def get_qa_chain(vectorstore):
    llm = ChatOllama(model="tinyllama")  # Use a small, RAM-friendly model
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

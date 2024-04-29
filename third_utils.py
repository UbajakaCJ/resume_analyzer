import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
MODEL = "gpt-3.5-turbo"
open_api_key = "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"

# loading PDF, DOCX, and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None
    
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(chunks, embeddings)

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
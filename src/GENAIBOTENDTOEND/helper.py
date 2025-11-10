from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
from huggingface_hub import login
from pinecone import ServerlessSpec


load_dotenv()


def load_pdf_files(data):
    loader = DirectoryLoader(data,glob="*.pdf",
    loader_cls = PyPDFLoader
    )
    documents = loader.load()
    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list containing only
    'source' in metadata and the original page_content.
    """
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source")}
        )
        for doc in docs
    ]



def download_embeddings():
    """
    Download and return HuggingFace Embeddings
    """
    embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

)
    return embeddings



# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk
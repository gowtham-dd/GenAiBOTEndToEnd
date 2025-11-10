from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
from huggingface_hub import login
from pinecone import ServerlessSpec
load_dotenv()
from src.GENAIBOTENDTOEND.helper import load_pdf_files,filter_to_minimal_docs,download_embeddings,text_split
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['GOOGLE_API_KEY']= GOOGLE_API_KEY


extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)

embeddings = download_embeddings()


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(name = index_name,
                    dimension=384,
                    metric="cosine",
                    spec = ServerlessSpec(cloud="aws",region="us-east-1"))
    
index= pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents = texts_chunk,
    embedding=embeddings,
    index_name= index_name
)
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

CHROMA_DIR="chroma_db"
Collection="transcripts"
model="all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"})

def get_vector_store(transcripts:str)->Chroma:
    splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks=splitter.split_text(transcripts)
    
    docs=[Document(page_content=chunk) for chunk in chunks]
    
    embeddings=get_embeddings()
    
    vector_store=Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR, collection_name=Collection)
    
    return vector_store

def load_vector_store()->Chroma:
    embeddings=get_embeddings()
    
    vector_store=Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, collection_name=Collection)
    
    return vector_store

def retrieve_vector_store(vector_store:Chroma, k:int=4):
    
    return vector_store.as_retriever(
        search_kwargs={"k": k},
        search_type="similarity"
    )


    
    
    
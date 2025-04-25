import os
from typing import Optional, Union
import streamlit as st
import qdrant_client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant, FAISS
from qdrant_client.http.models import VectorParams, Distance
from langchain.schema import Document

def get_qdrant_vectorstore(
    docs: list[Document],
    collection_name: str
) -> Optional[Qdrant]:
    """Create / connect Qdrant and index all Documents (text & tables)."""
    embeddings = _initialize_embeddings()
    if not embeddings:
        return None
    vector_size = _get_vector_size(embeddings)
    if not vector_size:
        return None

    client = _initialize_qdrant_client()
    if not client:
        return None

    if not _ensure_collection_exists(client, collection_name, vector_size):
        return None

    try:
        vs = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )
        if docs:
            # Add IDs to documents if they don't have them
            for i, doc in enumerate(docs):
                if 'id' not in doc.metadata:
                    doc.metadata['id'] = str(i)
            
            vs.add_documents(docs)
            st.sidebar.success(f"Indexed {len(docs)} documents (text+tables).")
        return vs
    except Exception as e:
        st.error(f"Failed to init vectorstore: {e}")
        return None

def _initialize_embeddings():
    """Initialize OpenAI embeddings."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set it in the .env file.")
        return None
    
    return OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        chunk_size=1000
    )

def _get_vector_size(embeddings):
    """Get vector size from embeddings."""
    try:
        return len(embeddings.embed_query("test"))
    except Exception as e:
        st.error(f"Failed to get embedding dimension from OpenAI: {e}")
        return None

def _initialize_qdrant_client():
    """Initialize Qdrant client based on environment settings."""
    qdrant_host_or_url = os.getenv("QDRANT_HOST")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_host_or_url:
        st.sidebar.warning("Using in-memory Qdrant. Data will be lost on restart.")
        return qdrant_client.QdrantClient(":memory:")
    
    try:
        timeout = 120.0
        if qdrant_host_or_url.startswith("http"):
            return qdrant_client.QdrantClient(
                url=qdrant_host_or_url,
                api_key=qdrant_api_key,
                timeout=timeout,
                prefer_grpc=True
            )
        else:
            st.sidebar.warning("QDRANT_HOST does not contain http/https. Assuming it's a hostname.")
            return qdrant_client.QdrantClient(
                host=qdrant_host_or_url,
                api_key=qdrant_api_key,
                timeout=timeout,
                prefer_grpc=True
            )
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        return None

def _ensure_collection_exists(client, collection_name, vector_size):
    """Ensure Qdrant collection exists with correct configuration."""
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            return True
        except Exception as e:
            st.error(f"Failed to create Qdrant collection '{collection_name}': {e}")
            return False

def _add_texts_to_vectorstore(vectorstore, texts, batch_size=100):
    """Add texts to vectorstore in batches with progress bar."""
    try:
        progress_bar = st.sidebar.progress(0)
        total_batches = (len(texts)-1)//batch_size + 1
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            vectorstore.add_texts(texts=batch_texts)
            progress = (i//batch_size + 1) / total_batches
            progress_bar.progress(progress)
        
        progress_bar.empty()
        st.sidebar.success(f"Added {len(texts)} chunks to vectorstore.")
    except Exception as e:
        st.error(f"Failed to add texts to vectorstore: {e}")

def create_vector_store(text_chunks):
    """Create a FAISS vector store from text chunks."""
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None 
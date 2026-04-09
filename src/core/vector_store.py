import logging
from typing import Optional

import streamlit as st
import qdrant_client
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client.http.models import VectorParams, Distance

from config.settings import OPENAI_API_KEY, QDRANT_HOST, QDRANT_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_qdrant_vectorstore(
    docs: list[Document],
    collection_name: str,
) -> Optional[QdrantVectorStore]:
    embeddings = _init_embeddings()
    if not embeddings:
        return None

    vector_size = _get_vector_size(embeddings)
    if not vector_size:
        return None

    client = _init_qdrant_client()
    if not client:
        return None

    if not _ensure_collection(client, collection_name, vector_size):
        return None

    try:
        vs = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        if docs:
            indexable = _prepare_docs_for_indexing(docs)
            vs.add_documents(indexable)
            st.sidebar.success(f"Indexed {len(indexable)} documents (text + tables + images).")
        return vs

    except Exception as e:
        logger.error("Failed to initialise vectorstore: %s", e)
        st.error(f"Failed to init vectorstore: {e}")
        return None


def _init_embeddings() -> Optional[OpenAIEmbeddings]:
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Set OPENAI_API_KEY in .env.")
        return None
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def _get_vector_size(embeddings: OpenAIEmbeddings) -> Optional[int]:
    try:
        return len(embeddings.embed_query("dimension probe"))
    except Exception as e:
        st.error(f"Failed to get embedding dimension: {e}")
        return None


def _init_qdrant_client() -> Optional[qdrant_client.QdrantClient]:
    if not QDRANT_HOST:
        st.sidebar.warning("No QDRANT_HOST set — using in-memory Qdrant (data lost on restart).")
        return qdrant_client.QdrantClient(":memory:")

    try:
        kwargs = {
            "api_key": QDRANT_API_KEY,
            "timeout": 120.0,
            "prefer_grpc": False,
        }
        if QDRANT_HOST.startswith("http"):
            kwargs["url"] = QDRANT_HOST
        else:
            kwargs["host"] = QDRANT_HOST

        client = qdrant_client.QdrantClient(**kwargs)
        client.get_collections()
        return client

    except Exception as e:
        logger.error("Qdrant connection failed: %s", e)
        st.error(f"Failed to connect to Qdrant: {e}")
        return None


def _ensure_collection(
    client: qdrant_client.QdrantClient,
    collection_name: str,
    vector_size: int,
) -> bool:
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except qdrant_client.http.exceptions.UnexpectedResponse:
        pass
    except Exception as e:
        logger.error("Error checking collection: %s", e)

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return True
    except Exception as e:
        st.error(f"Failed to create collection '{collection_name}': {e}")
        return False


def _prepare_docs_for_indexing(docs: list[Document]) -> list[Document]:
    if "image_cache" not in st.session_state:
        st.session_state.image_cache = {}

    prepared: list[Document] = []
    for i, doc in enumerate(docs):
        meta = dict(doc.metadata)
        meta["id"] = str(i)
        if "image_bytes" in meta:
            key = f"{meta.get('source', '')}_{meta.get('page', '')}_{meta.get('image_index', '')}"
            st.session_state.image_cache[key] = {
                "bytes": meta.pop("image_bytes"),
                "ext": meta.get("image_ext", "png"),
            }
            meta["image_cache_key"] = key
        prepared.append(Document(page_content=doc.page_content, metadata=meta))

    return prepared

import streamlit as st
from langchain_core.messages import AIMessage

from core.pdf_processor import get_pdf_documents, get_text_chunks_from_documents
from core.image_processor import describe_images
from core.vector_store import get_qdrant_vectorstore
from core.rag_chain import get_context_retriever_chain, get_conversational_rag_chain
from utils.helpers import generate_collection_name


def render_sidebar():
    """Render the sidebar with PDF upload and processing controls."""
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your research papers here and click 'Process'",
            accept_multiple_files=True,
            type="pdf",
        )

        if st.button("Process Documents"):
            _process_documents(pdf_docs)

        st.divider()
        _render_status()

        if st.button("Clear Chat History"):
            _clear_chat()


def _process_documents(pdf_docs):
    if not pdf_docs:
        st.warning("Please upload at least one PDF document.")
        return

    with st.spinner("Extracting text, tables, and images..."):
        docs = get_pdf_documents(pdf_docs)

    with st.spinner("Generating image descriptions (multimodal LLM)..."):
        docs = describe_images(docs)

    with st.spinner("Chunking documents..."):
        chunked = get_text_chunks_from_documents(docs)

    with st.spinner("Indexing into vector store..."):
        collection = generate_collection_name()
        st.session_state.vectorstore = get_qdrant_vectorstore(chunked, collection)

    with st.spinner("Building RAG chain..."):
        retriever = get_context_retriever_chain(st.session_state.vectorstore)
        st.session_state.history_aware_retriever = retriever
        st.session_state.conversation_rag_chain = get_conversational_rag_chain(retriever)

    st.success("Done! Ask questions about your papers.")


def _render_status():
    if st.session_state.vectorstore:
        st.sidebar.success("Documents processed — ready to chat.")
    else:
        st.sidebar.info("Upload PDFs and click 'Process' to start.")


def _clear_chat():
    st.session_state.chat_history = [
        AIMessage(content="Hello! Upload some PDFs and ask me questions about them.")
    ]
    st.rerun()

import streamlit as st
from core.pdf_processor import get_pdf_documents, get_text_chunks_from_documents
from core.vector_store import get_qdrant_vectorstore
from core.rag_chain import get_context_retriever_chain, get_conversational_rag_chain
from utils.helpers import get_document_id, generate_collection_name
from langchain_core.messages import AIMessage

def render_sidebar():
    """Render the sidebar with PDF upload functionality."""
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your research papers here and click 'Process'",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process Documents"):
            process_documents(pdf_docs)

        st.divider()
        render_sidebar_status()
        
        if st.button("Clear Chat History"):
            clear_chat_history()

def process_documents(pdf_docs):
    """Process uploaded PDF documents."""
    if not pdf_docs:
        st.warning("Please upload at least one PDF document.")
        return

    with st.spinner("Processing PDFs..."):
        # Get Documents (text + tables) from PDFs
        docs = get_pdf_documents(pdf_docs)
        
        # Split text Documents into chunks
        chunked_docs = get_text_chunks_from_documents(docs)
        
        # Generate collection name using helper function
        collection_name = generate_collection_name()
        
        # Create/update vector store with the generated collection name
        st.session_state.vectorstore = get_qdrant_vectorstore(
            chunked_docs,  # Now passing Document objects instead of text chunks
            collection_name=collection_name
        )
        
        # Initialize RAG chains
        st.session_state.history_aware_retriever = get_context_retriever_chain(
            st.session_state.vectorstore
        )

        st.session_state.conversation_rag_chain = get_conversational_rag_chain(
            st.session_state.history_aware_retriever
        )

def render_sidebar_status():
    """Display processing status in sidebar."""
    if st.session_state.vectorstore:
        st.sidebar.success("Successfully processed documents! Go ahead and ask questions.")
    else:
        st.sidebar.info("Upload PDFs and click 'Process' to start.")

def clear_chat_history():
    """Clear the chat history and reset the UI."""
    st.session_state.chat_history = [
        AIMessage(content="Hello! Upload some PDFs and ask me questions about them.")
    ]
    st.rerun() 
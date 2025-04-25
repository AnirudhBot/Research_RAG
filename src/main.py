import streamlit as st
from config.settings import PAGE_TITLE, PAGE_LAYOUT
from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat_interface
from utils.helpers import generate_collection_name
from langchain_core.messages import AIMessage

def initialize_session_state():
    """Initialize all session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! Upload some PDFs and ask me questions about them.")
        ]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "processed_doc_ids" not in st.session_state:
        st.session_state.processed_doc_ids = set()
    if "retriever_chain" not in st.session_state:
        st.session_state.retriever_chain = None
    if "conversation_rag_chain" not in st.session_state:
        st.session_state.conversation_rag_chain = None
    if "qdrant_collection_name" not in st.session_state:
        st.session_state.qdrant_collection_name = generate_collection_name()

def main():
    """Main application entry point."""
    # Configure the page
    st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
    st.title("📚 Research Paper Assistant")
    st.caption("Upload research papers (PDF) and ask questions about them.")

    # Initialize session state
    initialize_session_state()

    # Render UI components
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main() 
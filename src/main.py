import streamlit as st
from config.settings import PAGE_TITLE, PAGE_LAYOUT
from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat_interface
from langchain_core.messages import AIMessage


def _init_session_state():
    defaults = {
        "chat_history": [AIMessage(content="Hello! Upload some PDFs and ask me questions about them.")],
        "vectorstore": None,
        "history_aware_retriever": None,
        "conversation_rag_chain": None,
        "image_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
    st.title("Research Paper Assistant")
    st.caption("Upload research papers (PDF) and ask questions about them — text, tables, and figures.")

    _init_session_state()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()

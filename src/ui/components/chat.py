import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# newly added imports for table rendering
import pandas as pd
import io

def render_chat_interface():
    """Render the chat interface."""
    display_chat_history()
    handle_user_input()

def display_chat_history():
    """Display chat messages."""
    for message in st.session_state.chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(role):
            st.markdown(message.content)

def handle_user_input():
    """Process user input and generate responses, rendering tables if returned."""
    if query := st.chat_input("Ask about your documents..."):
        if not st.session_state.conversation_rag_chain:
            st.error("Please upload and process documents first.")
            return

        # Display user message
        st.chat_message("Human").markdown(query)
        st.session_state.chat_history.append(HumanMessage(content=query))

        with st.spinner("Thinking..."):
            try:
                # invoke RAG chain
                result = st.session_state.conversation_rag_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": query
                })

                # extract answer & source_documents
                answer = result.get("answer") or result.get("text", "")
                source_docs = result.get("source_documents", [])

                # display the LLM answer
                st.chat_message("AI").markdown(answer)
                st.session_state.chat_history.append(AIMessage(content=answer))

                # render any tables that were retrieved
                for doc in source_docs:
                    if doc.metadata.get("type") == "table":
                        src = doc.metadata.get("source", "unknown")
                        pg  = doc.metadata.get("page", "?")
                        st.markdown(f"**Table from {src} (page {pg})**")
                        df = pd.read_csv(io.StringIO(doc.page_content))
                        st.dataframe(df)

            except Exception as e:
                st.error(f"Error generating response: {e}")


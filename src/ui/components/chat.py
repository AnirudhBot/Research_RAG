import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


def render_chat_interface():
    """Render the chat interface."""
    _display_chat_history()
    _handle_user_input()


def _display_chat_history():
    for message in st.session_state.chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(role):
            st.markdown(message.content)


def _handle_user_input():
    query = st.chat_input("Ask about your documents...")
    if not query:
        return

    if not st.session_state.conversation_rag_chain:
        st.error("Please upload and process documents first.")
        return

    st.chat_message("Human").markdown(query)
    st.session_state.chat_history.append(HumanMessage(content=query))

    with st.spinner("Thinking..."):
        try:
            result = st.session_state.conversation_rag_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": query,
            })

            answer = result.get("answer", "")
            context_docs = result.get("context", [])

            # Display the LLM answer (may already contain markdown tables)
            st.chat_message("AI").markdown(answer)
            st.session_state.chat_history.append(AIMessage(content=answer))

            # Show retrieved images inline if they were part of the context
            _render_source_images(context_docs)

        except Exception as e:
            st.error(f"Error generating response: {e}")


def _render_source_images(context_docs: list):
    """Display any images that were retrieved as supporting context."""
    image_cache = st.session_state.get("image_cache", {})

    for doc in context_docs:
        if doc.metadata.get("content_type") != "image":
            continue

        cache_key = doc.metadata.get("image_cache_key")
        if not cache_key or cache_key not in image_cache:
            continue

        cached = image_cache[cache_key]
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")

        with st.expander(f"Figure from {source} (page {page})", expanded=False):
            st.image(cached["bytes"], use_container_width=True)

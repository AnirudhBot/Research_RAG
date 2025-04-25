import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_context_retriever_chain(vectorstore):
    """Create a retriever chain that considers chat history."""
    # Check if vectorstore is None
    if vectorstore is None:
        return None
        
    llm = ChatOpenAI(model=os.getenv("AI_MODEL"), temperature=0.2)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    context_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the chat history and latest question, create a search-optimized question.
        Maintain the core intent but make it self-contained and clear."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    return create_history_aware_retriever(llm, retriever, context_prompt)

def get_conversational_rag_chain(retriever):
    """Create the main RAG chain for answering questions, returning sources."""
    if retriever is None:
        return None

    llm = ChatOpenAI(model=os.getenv("AI_MODEL"), temperature=0.2)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful research assistant. Use the provided context to answer questions.
        Be detailed and technical in your responses. If the context doesn't contain enough information,
        clearly state what you cannot answer. Base your answers solely on the provided context.
        
        When answering:
        1. Be comprehensive and technical in your explanations
        2. Include specific details from the documents
        3. If multiple documents are relevant, synthesize the information
        4. Clearly state if certain information is not available in the context
        
        Context:
        {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create the retrieval chain
    return create_retrieval_chain(retriever, document_chain) 
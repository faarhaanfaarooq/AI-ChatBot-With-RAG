# Phase1 Imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st

# Phase2 Imports
import os
from dotenv import load_dotenv
from rag_pipeline import load_vectorstore, build_rag  
load_dotenv()

# Phase3 Imports

st.set_page_config(page_title="RAG Chatbot", layout="centered")

# Custom CSS for bubble styling
st.markdown("""
<style>
.chat-container {
    display: flex;
    margin: 10px 0;
    width: 100%;
}

/* User messages */
.chat-bubble-user {
    background-color: #005c4b; /* WhatsApp green */
    color: white;
    border-radius: 15px;
    padding: 8px 12px;
    display: inline-block;
    max-width: 70%;
    word-wrap: break-word;
    margin-left: auto; /* Pushes user bubble to the right */
}

/* Assistant messages */
.chat-bubble-assistant {
    background-color: #202c33; /* Dark grey */
    color: white;
    border-radius: 15px;
    padding: 8px 12px;
    display: inline-block;
    max-width: 70%;
    word-wrap: break-word;
    margin-right: auto; /* Pushes assistant bubble to the left */
}
</style>
""", unsafe_allow_html=True)

st.title("RAG Chatbot!")

# Session state variable for the old messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='chat-container'><div class='chat-bubble-user'>{msg['content']}</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-container'><div class='chat-bubble-assistant'>{msg['content']}</div></div>",
            unsafe_allow_html=True
        )


# Input box
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(
        f"<div class='chat-container'><div class='chat-bubble-user'>{user_input}</div></div>",
        unsafe_allow_html=True
    )

    try:
        # Load vectorstore + build RAG chain
        vectorstore = load_vectorstore()
        rag_chain = build_rag(vectorstore)

        # Run the chain
        result = rag_chain.invoke({"input": user_input})
        response = result["answer"]

        # Save + display response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(
            f"<div class='chat-container'><div class='chat-bubble-assistant'>{response}</div></div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error: [{str(e)}]")

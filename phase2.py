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
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
            f"<div class='chat-container'><div class='chat-bubble-user'>{msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='chat-container'><div class='chat-bubble-assistant'>{msg['content']}</div></div>", unsafe_allow_html=True)


@st.cache_resource
def get_vectorstore():
    pdf_name = "./DSA pat.pdf" #Enter your pdf
    loaders = [PyPDFLoader(pdf_name)]
    # Creatr Chunks
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore


# Input box
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(
        f"<div class='chat-container'><div class='chat-bubble-user'>{user_input}</div></div>", unsafe_allow_html=True)

    # Giving Prompt
    system_prompt = ChatPromptTemplate.from_template("""
    You are a PDF QA assistant. 
    You must ONLY use the provided context to answer. 
    If the answer is not in the context, reply: "I don’t know."
    Do not use outside knowledge. 
    Keep answers short and factual.

    Context:
    {context}

    Question: {input}
    """)


    # Initializing Model and API Key
    model = "openai/gpt-oss-120b"
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_chat = ChatGroq(
        api_key=groq_api_key,
        model=model
    )

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the document")

        #response = result["result"]
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Use the system prompt you already defined
        doc_chain = create_stuff_documents_chain(groq_chat, system_prompt)

        # Connect retriever with LLM chain
        chain = create_retrieval_chain(retriever, doc_chain)

        # Run the chain
        result = chain.invoke({"input": user_input})
        response = result["answer"]
        
        
    # response = "I am your assistant"
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.markdown(
            f"<div class='chat-container'><div class='chat-bubble-assistant'>{response}</div></div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error: [{str(e)}]")

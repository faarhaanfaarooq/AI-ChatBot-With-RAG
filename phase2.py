# Phase1 Imports
import streamlit as st

# Phase2 Imports
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

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

# Input box
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(
        f"<div class='chat-container'><div class='chat-bubble-user'>{user_input}</div></div>", unsafe_allow_html=True)

    # Giving Prompt
    system_prompt = ChatPromptTemplate.from_template("""You are an expert AI bot who knows everthing and can chat like chatgpt
                                              Answer the following question:{user_prompt}.
                                              Start answers directly. No small talk please.
                                              """)

    # Initializing Model and API Key
    model = "openai/gpt-oss-120b"
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_chat = ChatGroq(
        api_key=groq_api_key,
        model=model
    )

    # Initializing Parser for text output
    parser = StrOutputParser()

    chain = system_prompt | groq_chat | parser

    response = chain.invoke({"user_prompt": user_input})
    # response = "I am your assistant"
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
    st.markdown(
        f"<div class='chat-container'><div class='chat-bubble-assistant'>{response}</div></div>", unsafe_allow_html=True)

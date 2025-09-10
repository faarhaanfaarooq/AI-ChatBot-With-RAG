import streamlit as st

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Inject custom CSS for bubble styling
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
        st.markdown(f"<div class='chat-container'><div class='chat-bubble-user'>{msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-container'><div class='chat-bubble-assistant'>{msg['content']}</div></div>", unsafe_allow_html=True)

# Input box
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='chat-container'><div class='chat-bubble-user'>{prompt}</div></div>", unsafe_allow_html=True)

    response = "I am your assistant 🤖"
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f"<div class='chat-container'><div class='chat-bubble-assistant'>{response}</div></div>", unsafe_allow_html=True)

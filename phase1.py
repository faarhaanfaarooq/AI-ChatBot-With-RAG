import streamlit as st 
st.title("RAG Chatbot!")


# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)




prompt = st.chat_input("Pass your prompt  here!")

if prompt:
    st.chat_message("user").markdown(prompt)
import os
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from config import pdf, embeddings_model, LLM_Model

def load_vectorstore():
    """Load and index the PDF into a vectorstore."""
    loader = PyPDFLoader(pdf)
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceBgeEmbeddings(model_name=embeddings_model),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders([loader])
    return index.vectorstore

def build_rag(vectorstore):
    """Build the retrieval-augmented generation chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

    groq_chat = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_Model
    )
    doc_chain = create_stuff_documents_chain(groq_chat, system_prompt)
    return create_retrieval_chain(retriever, doc_chain)

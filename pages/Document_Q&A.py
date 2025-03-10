import os

import time

import streamlit as st

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

prompt=ChatPromptTemplate.from_template(
    """
        answer the question based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Question:{input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        st.session_state.loader = PyPDFDirectoryLoader("C:/DataScience/Gen_AI/Q&A/assets")
        
        st.session_state.docs = st.session_state.loader.load()
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title("RAG Document Q&A with groq and gemini")

user_prompt=st.text_input("enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("vector Database")

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)

    retriever=st.session_state.vectors.as_retriever()

    retriever_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retriever_chain.invoke({"input":user_prompt})
    print(f"respose_time:{time.process_time()-start}")

    st.write(response["answer"])

    with st.expander("Document similarity search"):
        for i,doc in enumerate(response["context"]):
            st.write("------------------------------------")
            st.write(doc.page_content)
            st.write("------------------------------------")
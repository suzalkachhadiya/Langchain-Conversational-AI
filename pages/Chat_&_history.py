import streamlit as st

import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
# groq_api_key=os.getenv("GROQ_API_KEY")

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

st.title("conversational RAG with PDF uploads and chat history")
st.write("upload pdf's and chat with their content")

api_key=st.text_input("enter your groq API key:",type="password")

if api_key:
    llm=ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192"
    )

    session_id=st.text_input("session id",value="default_session")

    if "store" not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("choose a PDF file",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temp_pdf=f"./assets/temp.pdf"
            with open(temp_pdf,"wb") as f:
                f.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temp_pdf)
            docs=loader.load()
            documents.extend(docs)
        
        text_splitter =  RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits  = text_splitter.split_documents(documents)
        vectorstores=Chroma.from_documents(documents=splits,embedding=embeddings,persist_directory=PERSIST_DIRECTORY)
        retriever = vectorstores.as_retriever()

        contextualize_q_system_prompt=(
            "given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history. Do not answer the question,"
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        system_prompt = (
            "you are an assistant for question-answering tasks."
            "use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer, say that you,"
            "don't know. use three sentences maximum and keep the"
            "answer concise"
            "\n\n"
            "{context}"
        )

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        ques_ans_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,ques_ans_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            # if session_id not in st.session_state.store:
            #     st.session_state.store[session_id]=ChatMessageHistory()
            #     return st.session_state.store[session_id]
            
            if session_id not in st.session_state.store:
            # Initialize and add a placeholder to avoid NoneType error
                chat_history = ChatMessageHistory()
                st.session_state.store[session_id] = chat_history
            else:
                chat_history = st.session_state.store[session_id]
        
            return chat_history
            
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("your question:")
        if user_input:
            session_history=get_session_history(session_id)
            session_history.add_user_message(user_input)
            st.write("Session History Check:", session_history)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )

            st.write(st.session_state.store)
            st.success("assistant:"+response["answer"])
            st.write("chat history:",session_history)

else:
    st.write("please enter your Groq api key")
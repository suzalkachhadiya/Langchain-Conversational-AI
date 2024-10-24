import streamlit as st
import google.generativeai as genai

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
load_dotenv()

# lansmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q & A chatbot with gemini"

# prompt template
prompt=ChatPromptTemplate(
    [
        ("system","you are a helpful assistant. please response to the user queries"),
        ("user","question:{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    genai.configure(api_key=api_key)
    
    llm=ChatGoogleGenerativeAI(model=llm)
    
    output_parser=StrOutputParser()

    chain=prompt | llm | output_parser

    answer=chain.invoke({"question":question})

    return answer

st.title("Enhanced Q&A Chatbot with :blue[Gemini]")

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("enter your Gemini API key:",type="password")

llm=st.sidebar.selectbox("select gemini model",["gemini-1.5-flash","gemini-1.5-pro"])

temperature=st.sidebar.slider("temperature",min_value=0.0, max_value=1.0,value=0.8)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50, max_value=300,value=150)

st.write("Go ahead and ask questions")

user_input=st.text_input("You:")

if user_input and api_key:
    response=generate_response(
        user_input, 
        api_key, 
        llm, 
        temperature, 
        max_tokens)
    st.write(response)

else:
    st.write("please provide the question")
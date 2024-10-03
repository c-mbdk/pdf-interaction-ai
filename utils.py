from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
embeddings = HuggingFaceEmbeddings()

def get_pdf_text(pdf_doc):
    doc_text = ""
    with pymupdf.open(pdf_doc) as doc:
        for page in doc:
            doc_text += page.get_text()
    return doc_text

def split_doc_text(doc_text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
    length_function = len,
)

    doc_chunks = text_splitter.split_text(doc_text)

    return doc_chunks

def get_vectorstore(doc_chunks):

    vectorstore = FAISS.from_texts(
        texts=doc_chunks, embedding=embeddings
    )

    vectorstore.save_local("faiss_db")

    return vectorstore

def get_conversation_chain(tools, question):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question with as much detail as possible from the provided context. If the answer is not in the provided context, respond with 'The answer is not available in the context'. Don't provide the wrong answer.""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": question})
    st.write("Answer: ", response['output'])


def process_user_input(user_question):
    new_db = FAISS.load_local(
        "faiss_db", embeddings, allow_dangerous_deserialization=True
    )
    
    retriever = new_db.as_retriever()

    retrieval_chain = create_retriever_tool(
        retriever, 
        "pdf_extractor", 
        "This tool is to provide answers to queries based on the pdf"
    )

    get_conversation_chain(retrieval_chain, user_question)


def process_file(file):
    doc_text = get_pdf_text(file)
    doc_chunks = split_doc_text(doc_text)
    get_vectorstore(doc_chunks)

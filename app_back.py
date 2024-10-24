from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from ctransformers import AutoModelForCausalLM
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

# Initialize API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Loading the index
index_name = "pharmaproduct"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# Initialize the language model
llm = AutoModelForCausalLM.from_pretrained("model/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama")

# Set up the retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the prompt template
system_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{context}"),
    ]
)

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit UI setup
st.title("Test")

with st.form("user_inputs"):
    user_input = st.text_input("Input", max_chars=20) 
    button = st.form_submit_button("Submit")
    
    if button and user_input:
        user_input_str = str(user_input)
        try:
            with st.spinner("Loading..."):
                st.write(f"User Input: {user_input_str}")
                
                # Make sure to pass the input correctly
                response = rag_chain.invoke({"input": user_input_str})
                
                # Adjust based on the structure of the response
                st.write(response["answer"])  # Ensure "answer" key exists in response
        except Exception as e:
            st.error(f"Error: {str(e)}")

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain import embeddings
from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
from ctransformers import AutoModelForCausalLM
#from langchain_community.llms import CTransformers
#from transformers import AutoTokenizer
#from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import streamlit as st
import os
#from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#tokenizer=AutoTokenizer.from_pretrained("model/llama-2-7b-chat.ggmlv3.q4_0.bin")

load_dotenv()
#tokenizer = YourTokenizer.from_pretrained("model_name", clean_up_tokenization_spaces=True)

#Initializing the Pinecone
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
#OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

#Loading the index
index_name = "pharmaproduct"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

#PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens':512, 'temperature':0.5})


#config={'max_new_tokens': 512, 'temperature': 0.8}
#llm = AutoModelForCausalLM.from_pretrained("model/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama")

#retriever = docsearch.as_retriever(search_kwargs={'k': 2})

#qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever = retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


#llm = OpenAI(temperature=0.4, max_tokens=500)
#prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}"),])
prompt = PromptTemplate(input_variables=["input"], template=system_prompt)

#print(f"User Input: {user_input_str}, Type: {type(user_input_str)}")
#print(f"Prompt Template: {system_prompt}, Type: {type(system_prompt)}")

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.title("Pharma Clusters")

with st.form("user_inputs"):

    user_input=st.text_input("input",max_chars=100) 
    
    button=st.form_submit_button("submit")
    
    if button and user_input:
        user_input_str = str(user_input)
        try:
            with st.spinner("loading..."):

                # Retrieve context based on user input
                #context_results = retriever.get_relevant_documents(user_input_str)
                #context = " ".join([doc.page_content for doc in context_results])  # Adjust according to your document structure

                st.write(f"User Input: {(user_input_str)}")
                response = rag_chain.invoke({"input": user_input_str})

                st.write(response["answer"])
                st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    

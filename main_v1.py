import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import os 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA

load_dotenv()

# Page Title & Heading content
st.set_page_config(page_title="Talk with PDF using RAG System")
st.title("RAG System - Chat with your PDF",)
st.caption("Powered by Devansh Rajani")
st.write("### Upload PDF and ask any questions..")

# Upload PDF file from streamlit
uploaded_pdf = st.file_uploader(label="",type=['pdf'])
if st.button("Process PDF"):
    if uploaded_pdf is not None:
        with st.spinner("Processing your PDF Please Wait..."):
            # Extract content from PDF
            text =""
            for page in PdfReader(uploaded_pdf).pages:
                text += page.extract_text()

            # Make chunks of that extracted text
            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

            # Make embedding vectors from chunks using Azure Embedding model
            embedding_model = AzureOpenAIEmbeddings(
                openai_api_key=os.getenv("AZURE_API_KEY"),
                azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_API_BASE"),
                azure_deployment=os.getenv("EMBEDDING_AZURE_OPENAI_API_NAME"),
                chunk_size=10
            )

            # Store embedding vectors using FAISS
            vectorstore = FAISS.from_texts(chunks, embedding_model)

            # RAG QA chain 
            llm = AzureChatOpenAI(
                openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                model_name="gpt-4o",
                temperature=0.7,
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )

            st.session_state.qa_chain = qa_chain
            st.success("PDF Processed successfully!")
    else:
        st.error("Please enter PDF file!!")

if "qa_chain" in st.session_state:
    user_query = st.text_input("Enter any query: ")
    if user_query:
        with st.spinner("Generating answer..."):
            result = st.session_state.qa_chain.run(user_query)
            st.markdown("### Answer:")
            st.markdown(result.replace("\n", "<br>"), unsafe_allow_html=True)

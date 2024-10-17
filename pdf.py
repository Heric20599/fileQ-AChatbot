from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

def main():
    # Load environment variables (for OpenAI API Key)
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Set up Streamlit page
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split text into manageable chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings using OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Take user question as input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            # Perform similarity search to find relevant document chunks
            docs = knowledge_base.similarity_search(user_question)
            
            # Initialize OpenAI LLM
            llm = ChatOpenAI(api_key=openai_api_key)
            
            # Load the question-answering chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Use invoke instead of run
            response = chain.invoke({
                "input_documents": docs, 
                "question": user_question
            })
            
            # Display the answer to the user
            st.write(response['output_text'])  # Assuming 'output_text' is the key for the answer

if __name__ == '__main__':
    main()

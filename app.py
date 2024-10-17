from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd

# Set the page configuration before any other Streamlit commands
st.set_page_config(page_title="Ask your CSV")

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("OPENAI_API_KEY is not set. Please check your .env file.")
        exit(1)
    else:
        st.success("OPENAI_API_KEY is set")

    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # Initialize the language model
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Create the agent using the pandas DataFrame
        agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            allow_dangerous_code=True,  # Enable dangerous code execution
            verbose=True
        )

        # Ask a question to the CSV file
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                try:
                    # Get the answer from the agent
                    response = agent_executor.run(user_question)
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

import streamlit as st
import os
from dotenv import load_dotenv
from chat import embedding_func, fields_definition, create_vector_store, create_llm

def index_search():
    st.title("Index Search tool")

    # Initialize necessary components
    embeddings = embedding_func()
    fields, sc = fields_definition(embeddings)
    vector_store = create_vector_store(embeddings, fields, sc)
    index_retriever = vector_store.as_retriever(fetch_k=3, fetch_metadata=True)
    # Unique keys for widgets
    input_key = "input_query"
    submit_button_key = "submit_button"
    exit_button_key = "exit_button"

    # Input field for user query
    query = st.text_input("Enter your query:", key=input_key)

    # Display submit button and exit button side by side
    
    if st.button("Submit", key=submit_button_key):
        if query:
            # Process the query and store message/response in history
            # response = chat.invoke({"query": query})
            try:
                response = index_retriever.invoke(query)
                # Create a list to hold the data for the table
                table_data = []

                # Extracting metadata from each document
                for document in response:
                    metadata = document.metadata
                    page_content = document.page_content[:100]

                    # Append metadata and page content to table data
                    table_data.append([metadata, page_content])

                # Display table
                st.table(table_data)
           
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    index_search()

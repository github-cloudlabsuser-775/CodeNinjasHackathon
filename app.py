import streamlit as st
import os
from lda import topics_from_pdf
from kgraph import crate_embedding_vector, build_knowledge_graph, draw_graph
from dotenv import load_dotenv
from chat import embedding_func, fields_definition, create_vector_store, create_llm

def main():
    st.title("Document Query Application")

    # Initialize necessary components
    embeddings = embedding_func()
    fields, sc = fields_definition(embeddings)
    vector_store = create_vector_store(embeddings, fields, sc)
    chat = create_llm(vector_store)

    # Get or create message history from session state
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []

    # Unique keys for widgets
    input_key = "input_query"
    submit_button_key = "submit_button"
    exit_button_key = "exit_button"

    # Input field for user query
    query = st.text_input("Enter your query:", key=input_key)

    # Display submit button and exit button side by side
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Submit", key=submit_button_key):
            if query:
                # Process the query and store message/response in history
                # response = chat.invoke({"query": query})
                try:
                    response = chat.invoke({"query": query})
                    # Accessing the list of documents
                    documents = response['source_documents']

                    # Extracting metadata from each document
                    response_metadata = set()
                    for document in documents:
                        response_metadata.add(document.metadata['source'])

                    message = f'Query: {response["query"]} : , Response: {response["result"]}, Source Documents: {response_metadata}'
                    st.session_state.message_history.append((query, message))
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    with col2:
        if st.button("Exit", key=exit_button_key):
            st.stop()

    # Display message history
    st.subheader("Message History")
    for i, (msg, resp) in enumerate(reversed(st.session_state.message_history[-5:]), 1):
        st.write(f"{i}. **User:** {msg}")
        st.write(f"   **Bot:** {resp}")
        st.write("")

def topic_extraction_tool():
    st.title("Topic Extraction Tool")

    file_path = "./pdfdocs/Bayesian_neuralnetworks.pdf"
    file_name = os.path.basename(file_path)
    
    num_topics = st.slider("Number of Topics", min_value=1, max_value=10, value=3)
    words_per_topic = st.slider("Words per Topic", min_value=5, max_value=50)

    if st.button("Submit"):
        # Call method from main_script
        # You may need to adjust the parameters based on the method signature
        summary = topics_from_pdf(file_path, num_topics, words_per_topic)

        # Display the summary with file name as title
        st.write(f"### {file_name}")
        st.write(summary)

def document_similarity_index():
    st.title("Document Similarity Index - Cosine Similarity")

    embedding_vector = {}
    dir = "./TextFiles"
    embedding_vector = crate_embedding_vector(dir, embedding_vector)
    knowledge_graph = build_knowledge_graph(embedding_vector)

    # Display similarity matrix
    st.write("Cosine Similarity Matrix:")
   # Display the graph image
    st.image(draw_graph(knowledge_graph), use_column_width=True)

if __name__ == "__main__":
    page = st.sidebar.selectbox(
        "Select a page:",
        ["Document Query Application", "Topic Extraction Tool", "Document Similarity Index - Cosine Similarity"]
    )
    if page == "Document Query Application":
        main()
    elif page == "Topic Extraction Tool":
        topic_extraction_tool()
    elif page == "Document Similarity Index - Cosine Similarity":
        document_similarity_index()

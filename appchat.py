import streamlit as st
from dotenv import load_dotenv
from chat import embedding_func, fields_definition, create_vector_store, create_llm

# Get configuration settings 
load_dotenv()

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
                response = chat.invoke({"query": query})
                st.session_state.message_history.append((query, response))

    with col2:
        if st.button("Exit", key=exit_button_key):
            st.stop()

    # Display message history
    st.subheader("Message History")
    for i, (msg, resp) in enumerate(reversed(st.session_state.message_history[-5:]), 1):
        st.write(f"{i}. **User:** {msg}")
        st.write(f"   **Bot:** {resp}")
        st.write("")

if __name__ == "__main__":
    main()
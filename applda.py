import streamlit as st
import os
from lda import topics_from_pdf

def main():
    st.title("Topic Extraction tool")

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

if __name__ == "__main__":
    main()

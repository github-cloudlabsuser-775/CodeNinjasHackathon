
import os
import json
from dotenv import load_dotenv

# Add OpenAI import
from openai import AzureOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.text_splitter import CharacterTextSplitter
from azure.search.documents.indexes.models import (
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)
from pypdf import PdfReader
from tqdm import tqdm
import os
import pandas as pd
import nest_asyncio
nest_asyncio.apply()
import warnings
warnings.filterwarnings("ignore") 
from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()
# Get configuration settings 
load_dotenv()
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_openai_api_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
azure_oai_text_deployment = os.getenv("AZURE_OAI_TEXT_DEPLOYMENT")
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_KEY")
azure_search_index = os.getenv("AZURE_SEARCH_INDEX")

def split_doc(filename_):
    print(f'Reading - {filename_}')
    loader = TextLoader(filename_, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs

def add_metadata(data,time):
    for chunk in data:
        chunk.metadata['last_update'] = time
    return data

def extract_combine_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        combined_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            combined_text += page_text + "\n\n"  # Add "\n\n" after each page
    return combined_text
# Adding same data with different last_update 
from datetime import datetime, timedelta

q2_time = (datetime.utcnow() - timedelta(days=90)).strftime(
    "%Y-%m-%dT%H:%M:%S-00:00"
)
q1_time = (datetime.utcnow() - timedelta(days=180)).strftime(
    "%Y-%m-%dT%H:%M:%S-00:00"
)
# documents[0]
# # Initialize the Azure OpenAI client
# client = AzureOpenAI(
#     base_url=f"{azure_oai_endpoint}/openai/deployments/{azure_oai_text_deployment}/extensions",
#     api_key=azure_oai_key,
#     api_version="2023-09-01-preview")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_oai_text_deployment,
    api_key=azure_openai_api_key,
    azure_endpoint=azure_oai_endpoint
)
embedding_function=embeddings.embed_query
fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embedding_function("Text")),
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field for filtering on document source
    SimpleField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
    # Additional data field for last doc update
    SimpleField(
        name="last_update",
        type=SearchFieldDataType.DateTimeOffset,
        searchable=True,
        filterable=True,
    ),
]
# Adding a custom scoring profile with a freshness function
sc_name = "scoring_profile"
sc = ScoringProfile(
    name=sc_name,
    text_weights=TextWeights(weights={"content": 5}),
    function_aggregation="sum",
    functions=[
        FreshnessScoringFunction(
            field_name="last_update",
            boost=100,
            parameters=FreshnessScoringParameters(boosting_duration="P2D"),
            interpolation="linear",
        )
    ],
)
def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding="utf-8") as file:
        file.write(text)

if __name__ == "__main__":
    # msft_q1 = split_doc('MSFT_q1_2024.txt')
    # msft_q2 = split_doc('MSFT_q2_2024.txt')
    # documents = msft_q1 + msft_q2
    # file = "./the-metamorphosis.pdf"
    # documents = load_pdf(file)
    # documentsp = split_doc(documents)
    # documents = documentms + documentsp

    file_path = "./Capgemini_2024-03-29_Document.pdf"
    output_txt_path = "./Capgemini_2024-03-29_Document.txt"

    # # Extract and combine text from all pages
    combined_text = extract_combine_text_from_pdf(file_path)
    save_text_to_file(combined_text, output_txt_path)
    # # Split the combined text using existing method
    documents = split_doc(output_txt_path)
    
    print(len(documents))
    # vector_store_address
    index_name = "codeninjas-index"

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=azure_search_endpoint,
        azure_search_key=azure_search_key,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        fields=fields,
        scoring_profiles=[sc],
        default_scoring_profile=sc_name,
    )

    vector_store.add_documents(documents=documents)
    azureai_retriever = vector_store.as_retriever()
    # azureai_retriever.invoke("How is Windows OEM revenue growth?")
    llm = AzureChatOpenAI(azure_endpoint=azure_oai_endpoint,
                        api_key=azure_openai_api_key, 
                        api_version="2023-09-01-preview",
                        azure_deployment=azure_oai_deployment)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=azureai_retriever,
        metadata={"application_type": "question_answering"}
    )
    query = "what is Caggemmini ooperting margin % increase"
    print(qa.invoke({"query": query}))
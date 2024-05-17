import os
import json
from dotenv import load_dotenv
# Add OpenAI import
import openai
from openai import AzureOpenAI
from tqdm import tqdm
import pandas as pd
import nest_asyncio
nest_asyncio.apply()
import warnings
warnings.filterwarnings("ignore") 
from azure.identity import DefaultAzureCredential
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
credential = DefaultAzureCredential()

text1 = "King and Queen are happy"
text2 = "King Ram and Queen Sita are married couples and have 2 kids"
text3 = "Guidewire is a policy center application and used to quote and create policies"
text4 = "Biiling is part Guidewire application and deals with customer invoices, and Payments features"
# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_oai_endpoint,
    api_key=azure_oai_key,
    api_version="2023-09-01-preview")
    
embeddings1 = client.embeddings.create(
    input=text1,
    model=azure_oai_text_deployment).data[0].embedding

embeddings2 = client.embeddings.create(
    input=text2,
    model=azure_oai_text_deployment).data[0].embedding



embedding_vector = []
embedding_vector.append(embeddings1)
embedding_vector.append(embeddings2)
embedding_vector.append(client.embeddings.create(
    input=text3,
    model=azure_oai_text_deployment).data[0].embedding
    )
embedding_vector.append(client.embeddings.create(
    input=text4,
    model=azure_oai_text_deployment).data[0].embedding
    )
# print(embedding_vector)
def similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

def build_knowledge_graph(embedding_vector):
    
    # Construct a graph
    G = nx.Graph()
    
    # Add nodes and edges based on embeddings
    for i, embedding_i in enumerate(embedding_vector):
        for j, embedding_j in enumerate(embedding_vector):
            if i != j:
                similarity_score = similarity(embedding_i, embedding_j)
                
                # Threshold for considering an edge
                if similarity_score > 0.8:
                    G.add_edge(i, j, weight=similarity_score)
    
    return G

def draw_graph(G):
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# # Example text
# text = """
# Your example text goes here. You can replace this with any text you want to analyze.
# """

# Build the knowledge graph
knowledge_graph = build_knowledge_graph(embedding_vector)

# Draw the graph
draw_graph(knowledge_graph)



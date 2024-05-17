import os
import json
import matplotlib.pyplot as plt
from io import BytesIO
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

def read_file(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        combined_text = ""
        token_count = 0
        for line in file:
            tokens = line.rstrip('\n').split()
            for token in tokens:
                token_count += 1
                if token_count > 2000:
                    break
                combined_text += token + ' '
    return combined_text.rstrip()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_oai_endpoint,
    api_key=azure_oai_key,
    api_version="2023-09-01-preview")
    
def similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

def crate_embedding_vector(dir, embedding_vector):
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir, file)
        print(f'Processing file: {file} Text deplyment: {azure_oai_text_deployment}')
        text = read_file(file_path)
        embedding_vector[file] = client.embeddings.create(
            input=text,
            model=azure_oai_text_deployment).data[0].embedding
        
    for i, embedding_i_label in enumerate(embedding_vector.items()):
        embedding_i, label_i = embedding_i_label[1], embedding_i_label[0]
        print(f'Embedding for {label_i[:5]}: {embedding_i[:5]}')
    return embedding_vector
    

def build_knowledge_graph(embedding_vector):
    G = nx.Graph()
    embeddings = list(embedding_vector.values())
    for i, embedding_i_label in enumerate(embedding_vector.items()):
        embedding_i, label_i = embedding_i_label[1], embedding_i_label[0]
        for j, embedding_j_label in enumerate(embedding_vector.items()):
            embedding_j, label_j = embedding_j_label[1], embedding_j_label[0]
            if i != j:
                similarity_score = similarity(embedding_i, embedding_j)
                if similarity_score > 0.7:
                    G.add_edge(label_i, label_j, weight=similarity_score)
                else:
                    G.add_edge(label_i, label_j, weight=0)
    return G

# def draw_graph(G):
#     pos = nx.spring_layout(G, seed=42)
#     nx.draw(G, pos, with_labels=True, font_weight='bold')
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.show()
def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, font_weight='bold')

    # Get edge attributes
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # Iterate over edges, highlight edges with similarity > 0.7
    for u, v, attrs in G.edges(data=True):
        similarity = attrs['weight']
        if similarity > 0.75:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='g', width=2.0)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # plt.show()
    # Instead of plt.show(), save the figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf


if __name__ == "__main__":
    
    embedding_vector = {}
    dir = ".\TextFiles"
    embedding_vector = crate_embedding_vector(dir, embedding_vector)
    knowledge_graph = build_knowledge_graph(embedding_vector)

    # # Draw the graph
    draw_graph(knowledge_graph)

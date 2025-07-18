import pyTigerGraph as tg
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import json

# === Step 1: Connect to TigerGraph and read vertex data ===

HOST = "https://<your-tigergraph-host>"
GRAPHNAME = "<your-graph-name>"
USERNAME = "tigergraph"
PASSWORD = "<your-password>"
SECRET = "<your-secret>"  # Generated from TigerGraph UI

conn = tg.TigerGraphConnection(host=HOST, graphname=GRAPHNAME)
conn.getToken(SECRET)

# Read a limited number of vertices
vertex_type = "YourVertexType"
vertex_limit = 100

vertices = conn.getVertices(vertex_type, count=vertex_limit)

# === Step 2: Build a graph and apply Node2Vec ===

# Create NetworkX graph from TigerGraph vertices
G = nx.Graph()

# Add nodes and collect attribute vectors
node_attrs = {}

for vertex in vertices:
    v_id = vertex["v_id"]
    attrs = vertex["attributes"]
    G.add_node(v_id)
    node_attrs[v_id] = list(attrs.values())  # Use attribute values

# Create dummy edges (optional if edges not needed or fetched)
# If your graph has edges, fetch and add them here using conn.getEdges()

# Use node2vec to compute embeddings
node2vec = Node2Vec(
    G, dimensions=64, walk_length=10, num_walks=80, workers=2
)

model = node2vec.fit(window=5, min_count=1, batch_words=4)

# === Step 3: Update vertex embedding ("emd") back to TigerGraph ===

for node_id in G.nodes():
    embedding = model.wv[str(node_id)].tolist()
    update_data = {
        "attributes": {
            "emd": embedding
        }
    }
    conn.updateVertex(vertex_type, node_id, update_data)

print("✅ Embeddings updated in TigerGraph.")

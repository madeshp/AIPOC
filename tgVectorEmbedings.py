import pyTigerGraph as tg
import networkx as nx
from node2vec import Node2Vec
import numpy as np

# --- CONFIGURATION ---
TG_HOST = "http://YOUR_TG_HOST"  # e.g., "http://127.0.0.1"
TG_GRAPHNAME = "YOUR_GRAPH_NAME"
TG_USERNAME = "YOUR_USERNAME"
TG_PASSWORD = "YOUR_PASSWORD"
TG_SECRET = "YOUR_SECRET"  # For token generation
VERTEX_TYPE = "YOUR_VERTEX_TYPE"  # e.g., "Person"
LIMIT = 100  # Number of vertices to process
EMBEDDING_DIM = 64  # Size of embedding vector

# --- CONNECT TO TIGERGRAPH ---
conn = tg.TigerGraphConnection(host=TG_HOST, graphname=TG_GRAPHNAME, username=TG_USERNAME, password=TG_PASSWORD)
conn.getToken(TG_SECRET)

# --- READ VERTICES ---
vertices = conn.getVertices(VERTEX_TYPE, select=None, limit=LIMIT)

# --- BUILD GRAPH FOR NODE2VEC ---
G = nx.Graph()
for v in vertices:
    G.add_node(v["v_id"])
# Optionally, add edges if you want to use Node2Vec meaningfully
# For demo, we skip edges (or you can fetch edges from TigerGraph)

# --- GENERATE EMBEDDINGS ---
node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=10, num_walks=20, workers=1)
model = node2vec.fit(window=5, min_count=1)

# --- UPDATE EMBEDDINGS BACK TO TIGERGRAPH ---
for v in vertices:
    vid = v["v_id"]
    if str(vid) in model.wv:
        emb = model.wv[str(vid)].tolist()
        # Update the 'emd' attribute
        conn.updateVertex(VERTEX_TYPE, vid, set_attributes={"emd": emb})
    else:
        print(f"No embedding for vertex {vid}")

print("Embeddings updated successfully.")

# --- NOTES ---
# - Replace all YOUR_* placeholders with your actual TigerGraph details.
# - For meaningful embeddings, you should add edges to the NetworkX graph (fetch from TigerGraph if needed).
# - The 'emd' attribute in TigerGraph should be a list/array type with EMBEDDING_DIM size.

import pyTigerGraph as tg
import numpy as np
from sklearn.random_projection import SparseRandomProjection

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

# --- EXTRACT ATTRIBUTES AND PREPARE DATA ---
vertex_ids = []
attr_matrix = []
for v in vertices:
    vid = v["v_id"]
    attrs = v.get("attributes", {})
    # Convert all attribute values to floats (handle categorical/text as needed)
    attr_values = []
    for k, val in attrs.items():
        # Try to convert to float, else use 0.0 (or handle encoding for categorical/text)
        try:
            attr_values.append(float(val))
        except (ValueError, TypeError):
            attr_values.append(0.0)
    vertex_ids.append(vid)
    attr_matrix.append(attr_values)

# Pad attribute vectors to same length
max_len = max(len(row) for row in attr_matrix)
for row in attr_matrix:
    row += [0.0] * (max_len - len(row))
attr_matrix = np.array(attr_matrix)

# --- RANDOM FAST PROJECTION ---
projector = SparseRandomProjection(n_components=EMBEDDING_DIM, random_state=42)
embeddings = projector.fit_transform(attr_matrix)

# --- UPDATE EMBEDDINGS BACK TO TIGERGRAPH ---
for vid, emb in zip(vertex_ids, embeddings):
    conn.updateVertex(VERTEX_TYPE, vid, set_attributes={"emd": emb.tolist()})

print("Random Fast Projection embeddings updated successfully.")

# --- NOTES ---
# - Replace all YOUR_* placeholders with your actual TigerGraph details.
# - This script assumes all attributes are numeric or can be coerced to float. For categorical/text, consider encoding.
# - The 'emd' attribute in TigerGraph should be a list/array type with EMBEDDING_DIM size.

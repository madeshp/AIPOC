import pyTigerGraph as tg
import numpy as np
from sklearn.random_projection import SparseRandomProjection
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_to_tigergraph(host, graphname, username, password, secret):
    """
    Connect to TigerGraph database
    """
    try:
        conn = tg.TigerGraphConnection(
            host=host,
            graphname=graphname,
            username=username,
            password=password
        )
        token = conn.getToken(secret)
        logger.info("Successfully connected to TigerGraph")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to TigerGraph: {e}")
        return None

def read_vertices_with_limit(conn, vertex_type, limit):
    """
    Read vertices from TigerGraph with specified limit
    """
    try:
        vertices = conn.getVertices(vertex_type, select=None, limit=limit)
        logger.info(f"Read {len(vertices)} vertices of type {vertex_type}")
        return vertices
    except Exception as e:
        logger.error(f"Failed to read vertices: {e}")
        return []

def convert_attributes_to_vectors(vertices, embedding_dim=64):
    """
    Convert vertex attributes to multi-dimensional vectors using Random Fast Projection
    """
    vertex_ids = []
    attribute_matrix = []
    
    # Extract attributes from all vertices
    for vertex in vertices:
        vid = vertex["v_id"]
        attributes = vertex.get("attributes", {})
        
        # Convert attribute values to numerical format
        attr_values = []
        for key, value in attributes.items():
            try:
                # Try to convert to float
                if isinstance(value, (int, float)):
                    attr_values.append(float(value))
                elif isinstance(value, str):
                    # For string values, use hash % 1000 to get numerical value
                    attr_values.append(float(hash(value) % 1000))
                elif isinstance(value, bool):
                    attr_values.append(float(value))
                elif isinstance(value, list):
                    # For lists, take the mean of numerical values
                    numeric_vals = [float(v) for v in value if isinstance(v, (int, float))]
                    attr_values.append(np.mean(numeric_vals) if numeric_vals else 0.0)
                else:
                    attr_values.append(0.0)
            except:
                attr_values.append(0.0)
        
        vertex_ids.append(vid)
        attribute_matrix.append(attr_values)
    
    # Pad all attribute vectors to same length
    if attribute_matrix:
        max_length = max(len(row) for row in attribute_matrix)
        for row in attribute_matrix:
            row.extend([0.0] * (max_length - len(row)))
        
        # Convert to numpy array
        attribute_matrix = np.array(attribute_matrix, dtype=np.float32)
        
        # Apply Random Fast Projection
        if attribute_matrix.shape[1] > 0:
            projector = SparseRandomProjection(
                n_components=embedding_dim,
                random_state=42
            )
            embeddings = projector.fit_transform(attribute_matrix)
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return vertex_ids, embeddings
    
    logger.error("Failed to generate embeddings")
    return [], []

def update_embeddings_to_tigergraph(conn, vertex_type, vertex_ids, embeddings):
    """
    Update embedding vectors back to TigerGraph vertex attribute 'emd1'
    """
    success_count = 0
    
    for vid, embedding in zip(vertex_ids, embeddings):
        try:
            # Convert embedding to list format
            if hasattr(embedding, 'toarray'):
                # Handle sparse matrix
                embedding_list = embedding.toarray().flatten().tolist()
            else:
                # Handle dense array
                embedding_list = embedding.tolist()
            
            # Update vertex with embedding in 'emd1' attribute
            result = conn.updateVertex(
                vertex_type,
                vid,
                set_attributes={"emd1": embedding_list}
            )
            
            if result:
                success_count += 1
                logger.info(f"Updated vertex {vid} with embedding")
            
        except Exception as e:
            logger.error(f"Failed to update vertex {vid}: {e}")
    
    logger.info(f"Successfully updated {success_count}/{len(vertex_ids)} vertices")
    return success_count

def main():
    """
    Main function to execute the complete pipeline
    """
    # Configuration - Replace with your actual values
    TG_HOST = "http://YOUR_TG_HOST"  # e.g., "http://127.0.0.1:9000"
    TG_GRAPHNAME = "YOUR_GRAPH_NAME"
    TG_USERNAME = "YOUR_USERNAME"
    TG_PASSWORD = "YOUR_PASSWORD"
    TG_SECRET = "YOUR_SECRET"
    VERTEX_TYPE = "YOUR_VERTEX_TYPE"  # e.g., "Person"
    LIMIT = 100
    EMBEDDING_DIM = 64
    
    # Step 1: Connect to TigerGraph
    logger.info("Step 1: Connecting to TigerGraph...")
    conn = connect_to_tigergraph(TG_HOST, TG_GRAPHNAME, TG_USERNAME, TG_PASSWORD, TG_SECRET)
    if not conn:
        return False
    
    # Step 2: Read vertices with limit
    logger.info("Step 2: Reading vertices...")
    vertices = read_vertices_with_limit(conn, VERTEX_TYPE, LIMIT)
    if not vertices:
        logger.error("No vertices found")
        return False
    
    # Step 3: Convert attributes to vectors using Random Fast Projection
    logger.info("Step 3: Converting attributes to vectors using Random Fast Projection...")
    vertex_ids, embeddings = convert_attributes_to_vectors(vertices, EMBEDDING_DIM)
    if not vertex_ids:
        logger.error("Failed to generate embeddings")
        return False
    
    # Step 4: Update vectors back to TigerGraph 'emd1' attribute
    logger.info("Step 4: Updating embeddings back to TigerGraph...")
    success_count = update_embeddings_to_tigergraph(conn, VERTEX_TYPE, vertex_ids, embeddings)
    
    if success_count > 0:
        logger.info(f"✅ Successfully completed! Updated {success_count} vertices with embeddings")
        return True
    else:
        logger.error("❌ Failed to update any vertices")
        return False

# Execute the pipeline
if __name__ == "__main__":
    success = main()
    if success:
        print("Random Fast Projection embedding pipeline completed successfully!")
    else:
        print("Pipeline execution failed!")

# --- SCHEMA REQUIREMENT ---
"""
Make sure your TigerGraph schema has the 'emd1' attribute:

ALTER VERTEX YourVertexType ADD ATTRIBUTE (emd1 LIST<DOUBLE>);

Or when creating the vertex:
CREATE VERTEX YourVertexType (
    PRIMARY_ID id STRING,
    emd1 LIST<DOUBLE>,
    -- other attributes...
) WITH primary_id_as_attribute="true"
"""

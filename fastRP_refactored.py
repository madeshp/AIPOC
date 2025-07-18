import pyTigerGraph as tg
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import json
from typing import Dict, List, Any, Tuple
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TigerGraphEmbeddings:
    def __init__(self, host: str, graphname: str, username: str, password: str, secret: str):
        """Initialize TigerGraph connection and embedding generator."""
        self.host = host
        self.graphname = graphname
        self.username = username
        self.password = password
        self.secret = secret
        self.conn = None
        self.projector = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def connect(self) -> bool:
        """Establish connection to TigerGraph."""
        try:
            self.conn = tg.TigerGraphConnection(
                host=self.host, 
                graphname=self.graphname, 
                username=self.username, 
                password=self.password
            )
            token = self.conn.getToken(self.secret)
            logger.info(f"Connected to TigerGraph successfully. Token: {token[:20]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TigerGraph: {e}")
            return False
    
    def fetch_vertices(self, vertex_type: str, limit: int = 100) -> List[Dict]:
        """Fetch vertices from TigerGraph."""
        try:
            vertices = self.conn.getVertices(vertex_type, select=None, limit=limit)
            logger.info(f"Fetched {len(vertices)} vertices of type {vertex_type}")
            return vertices
        except Exception as e:
            logger.error(f"Failed to fetch vertices: {e}")
            return []
    
    def preprocess_attributes(self, vertices: List[Dict]) -> Tuple[List[str], np.ndarray]:
        """
        Extract and preprocess vertex attributes.
        Handles numeric, categorical, and text attributes.
        """
        vertex_ids = []
        attr_matrix = []
        all_attr_names = set()
        
        # First pass: collect all attribute names
        for v in vertices:
            attrs = v.get("attributes", {})
            all_attr_names.update(attrs.keys())
        
        all_attr_names = sorted(list(all_attr_names))
        logger.info(f"Found attributes: {all_attr_names}")
        
        # Second pass: extract and encode attributes
        for v in vertices:
            vid = v["v_id"]
            attrs = v.get("attributes", {})
            
            attr_values = []
            for attr_name in all_attr_names:
                val = attrs.get(attr_name, None)
                processed_val = self._process_attribute_value(attr_name, val)
                attr_values.append(processed_val)
            
            vertex_ids.append(vid)
            attr_matrix.append(attr_values)
        
        attr_matrix = np.array(attr_matrix, dtype=np.float32)
        
        # Handle missing values (NaN)
        attr_matrix = np.nan_to_num(attr_matrix, nan=0.0)
        
        # Standardize features
        attr_matrix = self.scaler.fit_transform(attr_matrix)
        
        logger.info(f"Preprocessed attribute matrix shape: {attr_matrix.shape}")
        return vertex_ids, attr_matrix
    
    def _process_attribute_value(self, attr_name: str, value: Any) -> float:
        """Process individual attribute value based on its type."""
        if value is None:
            return 0.0
        
        # Try numeric conversion first
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        
        # Handle categorical/text data with label encoding
        if isinstance(value, str):
            if attr_name not in self.label_encoders:
                self.label_encoders[attr_name] = LabelEncoder()
                # Initialize with this value
                self.label_encoders[attr_name].fit([value])
                return 0.0
            else:
                try:
                    return float(self.label_encoders[attr_name].transform([value])[0])
                except ValueError:
                    # New category not seen during fit
                    # Add it to the encoder
                    current_classes = list(self.label_encoders[attr_name].classes_)
                    current_classes.append(value)
                    self.label_encoders[attr_name].classes_ = np.array(current_classes)
                    return float(len(current_classes) - 1)
        
        # Handle boolean
        if isinstance(value, bool):
            return float(value)
        
        # Handle lists/arrays (take mean or first element)
        if isinstance(value, (list, tuple)):
            if len(value) > 0:
                try:
                    return float(np.mean([float(x) for x in value if x is not None]))
                except (ValueError, TypeError):
                    return 0.0
            return 0.0
        
        # Default fallback
        return 0.0
    
    def generate_embeddings(self, attr_matrix: np.ndarray, embedding_dim: int = 64) -> np.ndarray:
        """Generate embeddings using Random Projection."""
        try:
            # Use Johnson-Lindenstrauss lemma for dimension estimation
            n_samples = attr_matrix.shape[0]
            min_dim = int(np.log(n_samples) * 4)  # Rough estimate
            actual_dim = min(embedding_dim, attr_matrix.shape[1])
            
            if actual_dim < min_dim:
                logger.warning(f"Embedding dimension {actual_dim} might be too small for {n_samples} samples")
            
            self.projector = SparseRandomProjection(
                n_components=actual_dim,
                density='auto',
                random_state=42
            )
            
            embeddings = self.projector.fit_transform(attr_matrix)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return np.array([])
    
    def update_embeddings(self, vertex_type: str, vertex_ids: List[str], 
                         embeddings: np.ndarray, embedding_attr: str = "embedding") -> bool:
        """Update embeddings back to TigerGraph."""
        success_count = 0
        total_count = len(vertex_ids)
        
        for i, (vid, emb) in enumerate(zip(vertex_ids, embeddings)):
            try:
                # Convert sparse matrix to dense if needed
                if hasattr(emb, 'toarray'):
                    emb_list = emb.toarray().flatten().tolist()
                else:
                    emb_list = emb.tolist()
                
                # Update vertex with embedding
                result = self.conn.updateVertex(
                    vertex_type, 
                    vid, 
                    set_attributes={embedding_attr: emb_list}
                )
                
                if result:
                    success_count += 1
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Updated {i + 1}/{total_count} vertices")
                    
            except Exception as e:
                logger.error(f"Failed to update vertex {vid}: {e}")
                continue
        
        logger.info(f"Successfully updated {success_count}/{total_count} vertices")
        return success_count == total_count
    
    def save_model_artifacts(self, filepath: str):
        """Save model artifacts for later use."""
        try:
            artifacts = {
                'projector': self.projector,
                'scaler': self.scaler,
                'label_encoders': {k: v.classes_.tolist() for k, v in self.label_encoders.items()}
            }
            
            with open(filepath, 'w') as f:
                json.dump(artifacts, f, indent=2, default=str)
            
            logger.info(f"Model artifacts saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model artifacts: {e}")
    
    def run_pipeline(self, vertex_type: str, limit: int = 100, 
                    embedding_dim: int = 64, embedding_attr: str = "embedding") -> bool:
        """Run the complete embedding pipeline."""
        logger.info("Starting TigerGraph embedding pipeline...")
        
        # Connect to TigerGraph
        if not self.connect():
            return False
        
        # Fetch vertices
        vertices = self.fetch_vertices(vertex_type, limit)
        if not vertices:
            logger.error("No vertices found or failed to fetch")
            return False
        
        # Preprocess attributes
        vertex_ids, attr_matrix = self.preprocess_attributes(vertices)
        if attr_matrix.size == 0:
            logger.error("Failed to preprocess attributes")
            return False
        
        # Generate embeddings
        embeddings = self.generate_embeddings(attr_matrix, embedding_dim)
        if embeddings.size == 0:
            logger.error("Failed to generate embeddings")
            return False
        
        # Update embeddings in TigerGraph
        success = self.update_embeddings(vertex_type, vertex_ids, embeddings, embedding_attr)
        
        if success:
            logger.info("Pipeline completed successfully!")
            return True
        else:
            logger.error("Pipeline completed with errors")
            return False


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Configuration
    config = {
        "host": "http://YOUR_TG_HOST",  # e.g., "http://127.0.0.1:9000"
        "graphname": "YOUR_GRAPH_NAME",
        "username": "YOUR_USERNAME", 
        "password": "YOUR_PASSWORD",
        "secret": "YOUR_SECRET"
    }
    
    # Parameters
    vertex_type = "YOUR_VERTEX_TYPE"  # e.g., "Person"
    limit = 100
    embedding_dim = 64
    embedding_attr = "embedding"  # Attribute name to store embeddings
    
    # Create and run pipeline
    embedder = TigerGraphEmbeddings(**config)
    
    try:
        success = embedder.run_pipeline(
            vertex_type=vertex_type,
            limit=limit,
            embedding_dim=embedding_dim,
            embedding_attr=embedding_attr
        )
        
        if success:
            # Optionally save model artifacts
            embedder.save_model_artifacts("embedding_model.json")
            print("✅ Embedding pipeline completed successfully!")
        else:
            print("❌ Embedding pipeline failed!")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"❌ Pipeline execution failed: {e}")

# --- SCHEMA REQUIREMENTS ---
"""
Make sure your TigerGraph schema includes an attribute to store embeddings:

CREATE VERTEX YourVertexType (
    PRIMARY_ID id STRING, 
    embedding LIST<DOUBLE>,  -- or LIST<FLOAT>
    -- other attributes...
) WITH primary_id_as_attribute="true"

Or use ALTER VERTEX to add the embedding attribute:
ALTER VERTEX YourVertexType ADD ATTRIBUTE (embedding LIST<DOUBLE>);
"""

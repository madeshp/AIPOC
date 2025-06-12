import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    TIGERGRAPH_HOST = os.getenv('TIGERGRAPH_HOST', 'http://localhost:9000')
    TIGERGRAPH_GRAPH = os.getenv('TIGERGRAPH_GRAPH', 'FraudDetection')
    TIGERGRAPH_USERNAME = os.getenv('TIGERGRAPH_USERNAME', 'tigergraph')
    TIGERGRAPH_PASSWORD = os.getenv('TIGERGRAPH_PASSWORD')
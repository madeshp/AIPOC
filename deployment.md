
# GraphRAG Fraud Detection Chatbot Deployment Guide

## Prerequisites

1. Python 3.8 or higher
2. TigerGraph database (local or cloud instance)
3. OpenAI API key

## Installation Steps

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv graphrag_env
source graphrag_env/bin/activate  # On Windows: graphrag_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
TIGERGRAPH_HOST=http://localhost:9000
TIGERGRAPH_GRAPH=FraudDetection
TIGERGRAPH_USERNAME=tigergraph
TIGERGRAPH_PASSWORD=your_password_here
```

### 3. Setup TigerGraph

1. Install TigerGraph (if running locally):
   ```bash
   # Download and install from https://tigergraph.com/download/
   ```

2. Create the fraud detection schema:
   ```bash
   gsql tigergraph_schema.gsql
   ```

3. Install fraud detection queries:
   ```bash
   gsql fraud_detection_queries.gsql
   ```

### 4. Run the Application

```bash
streamlit run chatbot_app.py
```

The application will be available at `http://localhost:8501`
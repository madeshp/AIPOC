import streamlit as st
import openai
import pyTigerGraph as tg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re
from typing import List, Dict, Any
import hashlib

# Page configuration
st.set_page_config(
    page_title="GraphRAG Fraud Detection Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
    }
    
    .safe-indicator {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class TigerGraphConnector:
    """Handle TigerGraph database connections and queries"""
    
    def __init__(self, host: str, graph_name: str, username: str, password: str):
        self.conn = tg.TigerGraphConnection(
            host=host,
            graphname=graph_name,
            username=username,
            password=password
        )
        
    def get_transaction_data(self, user_id: str = None, limit: int = 100) -> List[Dict]:
        """Retrieve transaction data from TigerGraph"""
        # Example GSQL query - adjust based on your schema
        query = f"""
        INTERPRET QUERY () FOR GRAPH MyGraph {{
            ListAccum<EDGE> @@edgeList;
            Start = {{Transaction.*}};
            
            Result = SELECT t FROM Start:t
                     WHERE t.amount > 0
                     LIMIT {limit};
            
            PRINT Result;
        }}
        """
        
        try:
            results = self.conn.runInterpretedQuery(query)
            return results
        except Exception as e:
            st.error(f"TigerGraph query error: {str(e)}")
            return []
    
    def detect_fraud_patterns(self, transaction_data: Dict) -> Dict:
        """Detect fraud patterns using graph queries"""
        # Example fraud detection patterns
        patterns = {
            "rapid_transactions": self._check_rapid_transactions(transaction_data),
            "unusual_amounts": self._check_unusual_amounts(transaction_data),
            "suspicious_connections": self._check_suspicious_connections(transaction_data)
        }
        return patterns
    
    def _check_rapid_transactions(self, data: Dict) -> bool:
        """Check for rapid successive transactions"""
        # Implement your fraud detection logic
        return False
    
    def _check_unusual_amounts(self, data: Dict) -> bool:
        """Check for unusual transaction amounts"""
        return False
    
    def _check_suspicious_connections(self, data: Dict) -> bool:
        """Check for suspicious network connections"""
        return False

class GraphRAGProcessor:
    """Handle GraphRAG processing for enhanced responses"""
    
    def __init__(self, openai_api_key: str, tigergraph_conn: TigerGraphConnector):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.tg_conn = tigergraph_conn
        
    def process_query(self, query: str, context_data: List[Dict]) -> str:
        """Process user query with GraphRAG approach"""
        
        # Extract relevant graph data based on query
        graph_context = self._extract_graph_context(query, context_data)
        
        # Create enhanced prompt with graph data
        enhanced_prompt = self._create_enhanced_prompt(query, graph_context)
        
        # Generate response using OpenAI
        response = self._generate_response(enhanced_prompt)
        
        return response
    
    def _extract_graph_context(self, query: str, data: List[Dict]) -> str:
        """Extract relevant context from graph data"""
        # Simple keyword-based extraction - enhance as needed
        keywords = ["fraud", "transaction", "suspicious", "pattern", "amount", "user"]
        
        relevant_data = []
        for item in data[:5]:  # Limit context size
            if any(keyword in str(item).lower() for keyword in keywords):
                relevant_data.append(item)
        
        return json.dumps(relevant_data, indent=2)
    
    def _create_enhanced_prompt(self, query: str, graph_context: str) -> str:
        """Create enhanced prompt with graph context"""
        prompt = f"""
        You are a fraud detection expert analyzing financial transaction data from a graph database.
        
        User Query: {query}
        
        Graph Database Context:
        {graph_context}
        
        Please provide a comprehensive analysis focusing on:
        1. Fraud detection insights
        2. Pattern recognition
        3. Risk assessment
        4. Recommendations
        
        Be specific and reference the graph data when relevant.
        """
        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a fraud detection specialist with expertise in graph databases and financial analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'tg_connector' not in st.session_state:
        st.session_state.tg_connector = None
    if 'graphrag_processor' not in st.session_state:
        st.session_state.graphrag_processor = None

def create_sample_data():
    """Create sample fraud detection data for demo"""
    return [
        {
            "transaction_id": "TXN001",
            "user_id": "USER123",
            "amount": 15000,
            "timestamp": "2024-06-12 10:30:00",
            "merchant": "Online Store A",
            "fraud_score": 0.85,
            "risk_level": "HIGH"
        },
        {
            "transaction_id": "TXN002",
            "user_id": "USER123",
            "amount": 50,
            "timestamp": "2024-06-12 10:32:00",
            "merchant": "Coffee Shop",
            "fraud_score": 0.15,
            "risk_level": "LOW"
        },
        {
            "transaction_id": "TXN003",
            "user_id": "USER456",
            "amount": 2500,
            "timestamp": "2024-06-12 11:15:00",
            "merchant": "Electronics Store",
            "fraud_score": 0.45,
            "risk_level": "MEDIUM"
        }
    ]

def display_fraud_metrics(data: List[Dict]):
    """Display fraud detection metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(data)
    high_risk = sum(1 for item in data if item.get('risk_level') == 'HIGH')
    avg_fraud_score = sum(item.get('fraud_score', 0) for item in data) / len(data) if data else 0
    total_amount = sum(item.get('amount', 0) for item in data)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_transactions}</h3>
            <p>Total Transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{high_risk}</h3>
            <p>High Risk Transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_fraud_score:.2f}</h3>
            <p>Avg Fraud Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${total_amount:,}</h3>
            <p>Total Amount</p>
        </div>
        """, unsafe_allow_html=True)

def create_fraud_visualization(data: List[Dict]):
    """Create fraud detection visualizations"""
    df = pd.DataFrame(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Fraud Score Distribution")
        fig_hist = px.histogram(
            df, 
            x='fraud_score', 
            title='Distribution of Fraud Scores',
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Level Breakdown")
        risk_counts = df['risk_level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Risk Level Distribution',
            color_discrete_sequence=['#ff6b6b', '#feca57', '#48dbfb']
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 GraphRAG Fraud Detection Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Hardcoded OpenAI API Key (replace with your actual key)
        openai_key = "your-openai-api-key-here"  # Replace with your actual API key
        
        # Hardcoded TigerGraph Configuration
        tg_host = "http://localhost:9000"
        tg_graph = "FraudDetection"
        tg_username = "tigergraph"
        tg_password = "tigergraph"
        
        # Connection button
        if st.button("🔗 Connect to GraphDB"):
            try:
                st.session_state.tg_connector = TigerGraphConnector(
                    tg_host, tg_graph, tg_username, tg_password
                )
                st.success("Connected to GraphDB!")
                
                # Initialize GraphRAG
                st.session_state.graphrag_processor = GraphRAGProcessor(
                    openai_key, st.session_state.tg_connector
                )
                st.success("GraphRAG initialized!")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
        
        # Show connection status
        if st.session_state.tg_connector:
            st.success("✅ GraphDB Connected")
        else:
            st.info("ℹ️ Click 'Connect to GraphDB' to start")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat container
        chat_container = st.container()
        
        # Display chat messages
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Fraud Detective AI:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask about fraud patterns, transactions, or risk analysis...")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            if st.session_state.graphrag_processor:
                # Use real GraphRAG processing
                sample_data = create_sample_data()  # Replace with real TigerGraph data
                response = st.session_state.graphrag_processor.process_query(user_input, sample_data)
            else:
                # Fallback response
                response = "Please click 'Connect to GraphDB' in the sidebar to enable full GraphRAG functionality. Currently showing demo mode."
            
            # Add bot response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        st.header("📈 Real-time Analytics")
        
        # Sample data for demo
        sample_data = create_sample_data()
        
        # Display metrics
        display_fraud_metrics(sample_data)
        
        # Fraud alert
        high_risk_transactions = [t for t in sample_data if t['risk_level'] == 'HIGH']
        if high_risk_transactions:
            st.markdown(f"""
            <div class="fraud-alert">
                🚨 FRAUD ALERT: {len(high_risk_transactions)} high-risk transaction(s) detected!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="safe-indicator">
                ✅ No high-risk transactions detected
            </div>
            """, unsafe_allow_html=True)
        
        # Recent transactions
        st.subheader("📋 Recent Transactions")
        for transaction in sample_data[:3]:
            risk_color = {
                'HIGH': '#ff6b6b',
                'MEDIUM': '#feca57',
                'LOW': '#2ed573'
            }.get(transaction['risk_level'], '#gray')
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {risk_color}, {risk_color}90); 
                        padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem; color: white;">
                <strong>ID:</strong> {transaction['transaction_id']}<br>
                <strong>Amount:</strong> ${transaction['amount']:,}<br>
                <strong>Risk:</strong> {transaction['risk_level']} ({transaction['fraud_score']:.2f})
            </div>
            """, unsafe_allow_html=True)
    
    # Visualizations section
    st.header("📊 Fraud Analytics Dashboard")
    create_fraud_visualization(create_sample_data())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🔍 GraphRAG Fraud Detection Chatbot | Powered by TigerGraph, OpenAI & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
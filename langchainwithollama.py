# config.py
"""Configuration settings for the LangChain-Ollama integration"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class OllamaConfig:
    """Configuration for Ollama settings"""
    base_url: str = "http://localhost:11434"
    model: str = "llama2"
    temperature: float = 0.7
    top_p: float = 0.9
    num_predict: int = 256
    timeout: int = 60

@dataclass
class EmbeddingConfig:
    """Configuration for embedding settings"""
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class VectorStoreConfig:
    """Configuration for vector store settings"""
    provider: str = "chroma"  # chroma, faiss, pinecone
    persist_directory: str = "./vector_store"
    collection_name: str = "documents"
    
    # Pinecone specific
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None

@dataclass
class AppConfig:
    """Main application configuration"""
    ollama: OllamaConfig = OllamaConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables"""
        return cls(
            ollama=OllamaConfig(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama2"),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            ),
            embedding=EmbeddingConfig(
                model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434"),
            ),
            vector_store=VectorStoreConfig(
                provider=os.getenv("VECTOR_STORE_PROVIDER", "chroma"),
                persist_directory=os.getenv("VECTOR_STORE_DIR", "./vector_store"),
            )
        )

# llm_provider.py
"""LLM provider implementations"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManagerForLLMRun
from config import OllamaConfig

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def get_llm(self) -> LLM:
        """Return the LLM instance"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass

class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self._llm = None
    
    def get_llm(self) -> LLM:
        """Get Ollama LLM instance"""
        if self._llm is None:
            self._llm = Ollama(
                base_url=self.config.base_url,
                model=self.config.model,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_predict=self.config.num_predict,
                timeout=self.config.timeout
            )
        return self._llm
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class LLMFactory:
    """Factory class for creating LLM providers"""
    
    @staticmethod
    def create_provider(provider_type: str, config: Any) -> BaseLLMProvider:
        """Create LLM provider based on type"""
        if provider_type.lower() == "ollama":
            return OllamaProvider(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")

# embedding_provider.py
"""Embedding provider implementations"""

from abc import ABC, abstractmethod
from typing import List
from langchain.embeddings.base import Embeddings
from langchain.embeddings import OllamaEmbeddings
from config import EmbeddingConfig

class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Return the embeddings instance"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the embedding provider is available"""
        pass

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._embeddings = None
    
    def get_embeddings(self) -> Embeddings:
        """Get Ollama embeddings instance"""
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                base_url=self.config.base_url,
                model=self.config.model
            )
        return self._embeddings
    
    def is_available(self) -> bool:
        """Check if Ollama embeddings are available"""
        try:
            import requests
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class EmbeddingFactory:
    """Factory class for creating embedding providers"""
    
    @staticmethod
    def create_provider(provider_type: str, config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """Create embedding provider based on type"""
        if provider_type.lower() == "ollama":
            return OllamaEmbeddingProvider(config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider_type}")

# vector_store_provider.py
"""Vector store provider implementations"""

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from config import VectorStoreConfig

class BaseVectorStoreProvider(ABC):
    """Abstract base class for vector store providers"""
    
    @abstractmethod
    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """Return the vector store instance"""
        pass
    
    @abstractmethod
    def create_from_documents(self, documents: List[Document], embeddings: Embeddings) -> VectorStore:
        """Create vector store from documents"""
        pass

class ChromaProvider(BaseVectorStoreProvider):
    """Chroma vector store provider implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
    
    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """Get Chroma vector store instance"""
        return Chroma(
            persist_directory=self.config.persist_directory,
            embedding_function=embeddings,
            collection_name=self.config.collection_name
        )
    
    def create_from_documents(self, documents: List[Document], embeddings: Embeddings) -> VectorStore:
        """Create Chroma vector store from documents"""
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name
        )

class FAISSProvider(BaseVectorStoreProvider):
    """FAISS vector store provider implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
    
    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """Get FAISS vector store instance"""
        try:
            return FAISS.load_local(self.config.persist_directory, embeddings)
        except:
            # Create empty FAISS index if none exists
            return FAISS.from_texts([""], embeddings)
    
    def create_from_documents(self, documents: List[Document], embeddings: Embeddings) -> VectorStore:
        """Create FAISS vector store from documents"""
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(self.config.persist_directory)
        return vector_store

class VectorStoreFactory:
    """Factory class for creating vector store providers"""
    
    @staticmethod
    def create_provider(provider_type: str, config: VectorStoreConfig) -> BaseVectorStoreProvider:
        """Create vector store provider based on type"""
        if provider_type.lower() == "chroma":
            return ChromaProvider(config)
        elif provider_type.lower() == "faiss":
            return FAISSProvider(config)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider_type}")

# document_processor.py
"""Document processing utilities"""

from typing import List, Optional
from langchain.schema import Document
from langchain.document_loaders import (
    TextLoader, PDFLoader, WebBaseLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import EmbeddingConfig

class DocumentProcessor:
    """Document processing and chunking utilities"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
    
    def load_from_file(self, file_path: str) -> List[Document]:
        """Load documents from a file"""
        if file_path.endswith('.pdf'):
            loader = PDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_from_directory(self, directory_path: str, glob_pattern: str = "**/*") -> List[Document]:
        """Load documents from a directory"""
        loader = DirectoryLoader(directory_path, glob=glob_pattern)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_from_web(self, urls: List[str]) -> List[Document]:
        """Load documents from web URLs"""
        loader = WebBaseLoader(urls)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_from_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Load documents from raw text"""
        document = Document(page_content=text, metadata=metadata or {})
        return self.text_splitter.split_documents([document])

# rag_system.py
"""RAG system implementation"""

from typing import List, Optional, Dict, Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from config import AppConfig
from llm_provider import LLMFactory
from embedding_provider import EmbeddingFactory
from vector_store_provider import VectorStoreFactory
from document_processor import DocumentProcessor

class RAGSystem:
    """Retrieval-Augmented Generation system"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Initialize providers
        self.llm_provider = LLMFactory.create_provider("ollama", config.ollama)
        self.embedding_provider = EmbeddingFactory.create_provider("ollama", config.embedding)
        self.vector_store_provider = VectorStoreFactory.create_provider(
            config.vector_store.provider, config.vector_store
        )
        
        # Initialize components
        self.document_processor = DocumentProcessor(config.embedding)
        self.llm = self.llm_provider.get_llm()
        self.embeddings = self.embedding_provider.get_embeddings()
        self.vector_store = None
        self.qa_chain = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        if self.vector_store is None:
            self.vector_store = self.vector_store_provider.create_from_documents(
                documents, self.embeddings
            )
        else:
            self.vector_store.add_documents(documents)
        
        # Rebuild QA chain
        self._build_qa_chain()
    
    def add_documents_from_file(self, file_path: str) -> None:
        """Add documents from a file"""
        documents = self.document_processor.load_from_file(file_path)
        self.add_documents(documents)
    
    def add_documents_from_directory(self, directory_path: str, glob_pattern: str = "**/*") -> None:
        """Add documents from a directory"""
        documents = self.document_processor.load_from_directory(directory_path, glob_pattern)
        self.add_documents(documents)
    
    def add_documents_from_web(self, urls: List[str]) -> None:
        """Add documents from web URLs"""
        documents = self.document_processor.load_from_web(urls)
        self.add_documents(documents)
    
    def add_documents_from_text(self, text: str, metadata: Optional[dict] = None) -> None:
        """Add documents from raw text"""
        documents = self.document_processor.load_from_text(text, metadata)
        self.add_documents(documents)
    
    def _build_qa_chain(self) -> None:
        """Build the QA chain"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add documents first.")
        
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Add documents first.")
        
        result = self.qa_chain({"query": question})
        return result["result"]
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add documents first.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def is_ready(self) -> bool:
        """Check if the RAG system is ready"""
        return (
            self.llm_provider.is_available() and
            self.embedding_provider.is_available() and
            self.vector_store is not None
        )

# main.py
"""Main application entry point"""

import argparse
from config import AppConfig
from rag_system import RAGSystem

def main():
    """Main application function"""
    parser = argparse.ArgumentParser(description="LangChain-Ollama RAG System")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--add-documents", help="Add documents from directory/file")
    parser.add_argument("--query", help="Query the system")
    
    args = parser.parse_args()
    
    # Load configuration
    config = AppConfig.from_env()
    
    # Initialize RAG system
    rag_system = RAGSystem(config)
    
    # Add documents if specified
    if args.add_documents:
        print(f"Adding documents from: {args.add_documents}")
        try:
            rag_system.add_documents_from_directory(args.add_documents)
            print("Documents added successfully!")
        except Exception as e:
            print(f"Error adding documents: {e}")
            return
    
    # Handle query
    if args.query:
        if not rag_system.is_ready():
            print("RAG system not ready. Please add documents first.")
            return
        
        print(f"Query: {args.query}")
        try:
            answer = rag_system.query(args.query)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error processing query: {e}")
    
    # Interactive mode
    if args.interactive:
        print("Entering interactive mode. Type 'quit' to exit.")
        while True:
            try:
                query = input("\nEnter your question: ")
                if query.lower() in ['quit', 'exit']:
                    break
                
                if not rag_system.is_ready():
                    print("RAG system not ready. Please add documents first.")
                    continue
                
                answer = rag_system.query(query)
                print(f"Answer: {answer}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()

# requirements.txt
"""
langchain>=0.1.0
ollama>=0.1.0
chromadb>=0.4.0
faiss-cpu>=1.7.0
PyPDF2>=3.0.0
requests>=2.25.0
python-dotenv>=1.0.0
"""

# example_usage.py
"""Example usage of the RAG system"""

from config import AppConfig
from rag_system import RAGSystem

def example_usage():
    """Example usage of the RAG system"""
    
    # Initialize with default configuration
    config = AppConfig()
    rag_system = RAGSystem(config)
    
    # Add documents from various sources
    print("Adding documents...")
    
    # From text
    rag_system.add_documents_from_text(
        "Python is a high-level programming language known for its simplicity and readability.",
        metadata={"source": "programming_facts"}
    )
    
    # From file (if exists)
    # rag_system.add_documents_from_file("path/to/document.pdf")
    
    # From directory (if exists)
    # rag_system.add_documents_from_directory("path/to/documents/")
    
    # From web URLs
    # rag_system.add_documents_from_web(["https://example.com/article"])
    
    # Query the system
    print("\nQuerying the system...")
    if rag_system.is_ready():
        answer = rag_system.query("What is Python?")
        print(f"Answer: {answer}")
        
        # Similarity search
        similar_docs = rag_system.similarity_search("Python programming")
        print(f"Found {len(similar_docs)} similar documents")
    else:
        print("System not ready")

if __name__ == "__main__":
    example_usage()

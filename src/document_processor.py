#!/usr/bin/env python3
"""
LangChain document processing pipeline.
This module handles loading, splitting, and embedding documents for RAG.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class DocumentProcessor:
    """Handles document loading, processing, and vector store management."""
    
    def __init__(self, knowledge_base_path: str, persist_directory: str = "data/chroma_db"):
        load_dotenv()
        
        self.knowledge_base_path = knowledge_base_path
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vector_store = None
        self.documents = []
        
        # Initialize embeddings
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """Load documents from the knowledge base directory."""
        print(f"📁 Loading documents from {self.knowledge_base_path}")
        
        try:
            # Load markdown and text files
            loader = DirectoryLoader(
                self.knowledge_base_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()
            
            # Also load .txt files if any
            txt_loader = DirectoryLoader(
                self.knowledge_base_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            txt_documents = txt_loader.load()
            documents.extend(txt_documents)
            
            print(f"✅ Loaded {len(documents)} documents")
            self.documents = documents
            return documents
            
        except Exception as e:
            print(f"❌ Error loading documents: {str(e)}")
            return []
    
    def split_documents(self, documents: Optional[List[Document]] = None) -> List[Document]:
        """Split documents into smaller chunks."""
        if documents is None:
            documents = self.documents
        
        if not documents:
            print("⚠️  No documents to split")
            return []
        
        print(f"✂️  Splitting {len(documents)} documents into chunks")
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            print(f"✅ Created {len(chunks)} document chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ Error splitting documents: {str(e)}")
            return []
    
    def create_vector_store(self, documents: Optional[List[Document]] = None) -> Optional[Chroma]:
        """Create or load vector store with document embeddings."""
        if documents is None:
            # Try to load existing vector store
            if os.path.exists(self.persist_directory):
                print(f"📦 Loading existing vector store from {self.persist_directory}")
                try:
                    self.vector_store = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    return self.vector_store
                except Exception as e:
                    print(f"⚠️  Error loading existing vector store: {str(e)}")
                    print("Creating new vector store...")
            
            # Load and process documents if not provided
            documents = self.load_documents()
            documents = self.split_documents(documents)
        
        if not documents:
            print("❌ No documents available for vector store creation")
            return None
        
        print(f"🔄 Creating vector store with {len(documents)} document chunks")
        
        try:
            # Create new vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist the vector store
            self.vector_store.persist()
            print(f"✅ Vector store created and saved to {self.persist_directory}")
            return self.vector_store
            
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            return None
    
    def search_documents(self, query: str, k: int = 4) -> List[Document]:
        """Search for relevant documents using similarity search."""
        if not self.vector_store:
            print("⚠️  Vector store not initialized. Creating it now...")
            self.create_vector_store()
        
        if not self.vector_store:
            print("❌ Unable to perform search - no vector store available")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            print(f"🔍 Found {len(results)} relevant document chunks for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"❌ Error searching documents: {str(e)}")
            return []
    
    def get_retriever(self, k: int = 4):
        """Get a retriever object for use in RAG chains."""
        if not self.vector_store:
            self.create_vector_store()
        
        if not self.vector_store:
            return None
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})

def test_document_processor():
    """Test the document processing pipeline."""
    print("🧪 Testing Document Processor")
    print("=" * 50)
    
    # Initialize processor
    knowledge_base_path = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base')
    processor = DocumentProcessor(knowledge_base_path)
    
    # Test document loading
    documents = processor.load_documents()
    if not documents:
        print("❌ No documents loaded")
        return False
    
    # Test document splitting
    chunks = processor.split_documents(documents)
    if not chunks:
        print("❌ No document chunks created")
        return False
    
    # Test vector store creation
    vector_store = processor.create_vector_store(chunks)
    if not vector_store:
        print("❌ Vector store creation failed")
        return False
    
    # Test search functionality
    test_queries = [
        "vacation policy",
        "API rate limiting",
        "pricing tiers",
        "password reset"
    ]
    
    print("\n🔍 Testing search functionality:")
    for query in test_queries:
        results = processor.search_documents(query, k=2)
        if results:
            print(f"\nQuery: '{query}'")
            for i, doc in enumerate(results, 1):
                print(f"  {i}. Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"     Preview: {doc.page_content[:100]}...")
        else:
            print(f"❌ No results for query: '{query}'")
    
    print("\n✅ Document processor test completed successfully!")
    return True

if __name__ == "__main__":
    test_document_processor()
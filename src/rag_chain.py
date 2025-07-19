#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) chain implementation using LangChain.
This module combines document retrieval with LLM generation.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from document_processor import DocumentProcessor

class RAGChain:
    """Retrieval-Augmented Generation chain for question answering."""
    
    def __init__(self, knowledge_base_path: str, model_name: str = "gpt-3.5-turbo"):
        load_dotenv()
        
        self.knowledge_base_path = knowledge_base_path
        self.model_name = model_name
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor(knowledge_base_path)
        
        # Initialize LLM
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=0.7
        )
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context.
Use the following pieces of context to answer the user's question. If you don't know the answer based on the context, say so.

Context:
{context}

Guidelines:
- Be concise and accurate
- Cite specific information from the context when possible
- If the context doesn't contain enough information, say "I don't have enough information in the provided context to answer that question"
- Be helpful and professional"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Initialize the retriever
        self.retriever = None
        self._setup_retriever()
        
        # Create the RAG chain
        self.chain = None
        self._setup_chain()
    
    def _setup_retriever(self):
        """Set up the document retriever."""
        try:
            self.retriever = self.doc_processor.get_retriever(k=4)
            if self.retriever:
                print("✅ Document retriever initialized successfully")
            else:
                print("❌ Failed to initialize document retriever")
        except Exception as e:
            print(f"❌ Error setting up retriever: {str(e)}")
    
    def _setup_chain(self):
        """Set up the complete RAG chain."""
        if not self.retriever:
            print("❌ Cannot setup chain without retriever")
            return
        
        try:
            # Create the chain that combines retrieval and generation
            self.chain = (
                {
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough(),
                    "chat_history": lambda _: self.memory.chat_memory.messages
                }
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
            print("✅ RAG chain initialized successfully")
        except Exception as e:
            print(f"❌ Error setting up RAG chain: {str(e)}")
    
    def _format_docs(self, docs):
        """Format retrieved documents for the prompt."""
        if not docs:
            return "No relevant context found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Document {i} ({source}):\n{doc.page_content}")
        
        return "\n\n".join(formatted)
    
    def ask(self, question: str) -> Dict[str, str]:
        """Ask a question using the RAG chain."""
        if not self.chain:
            return {
                "answer": "❌ RAG chain not properly initialized",
                "sources": [],
                "error": "Chain initialization failed"
            }
        
        try:
            # Get relevant documents for context
            relevant_docs = self.doc_processor.search_documents(question, k=4)
            
            # Generate answer using the chain
            answer = self.chain.invoke(question)
            
            # Update conversation memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            
            # Extract sources
            sources = [doc.metadata.get('source', 'Unknown') for doc in relevant_docs]
            
            return {
                "answer": answer,
                "sources": sources,
                "context_docs": len(relevant_docs)
            }
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "sources": [],
                "error": error_msg
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        history = []
        messages = self.memory.chat_memory.messages
        
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"type": "ai", "content": message.content})
        
        return history
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
        print("🧹 Conversation memory cleared")
    
    def reload_documents(self):
        """Reload documents and recreate the vector store."""
        print("🔄 Reloading documents...")
        try:
            # Remove existing vector store
            if os.path.exists(self.doc_processor.persist_directory):
                import shutil
                shutil.rmtree(self.doc_processor.persist_directory)
            
            # Recreate vector store
            self.doc_processor.create_vector_store()
            
            # Reinitialize retriever and chain
            self._setup_retriever()
            self._setup_chain()
            
            print("✅ Documents reloaded successfully")
        except Exception as e:
            print(f"❌ Error reloading documents: {str(e)}")

def test_rag_chain():
    """Test the RAG chain functionality."""
    print("🧪 Testing RAG Chain")
    print("=" * 50)
    
    # Initialize RAG chain
    knowledge_base_path = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base')
    rag = RAGChain(knowledge_base_path)
    
    if not rag.chain:
        print("❌ RAG chain initialization failed")
        return False
    
    # Test questions
    test_questions = [
        "What is the company's vacation policy?",
        "How many requests per hour are allowed for the API?",
        "What are the different pricing tiers?",
        "How do I reset my password?",
        "What programming languages are supported?",  # This should return "not enough info"
    ]
    
    print("\n💬 Testing RAG Chain with various questions:")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        result = rag.ask(question)
        
        print(f"   Answer: {result['answer'][:200]}..." if len(result['answer']) > 200 else f"   Answer: {result['answer']}")
        if result.get('sources'):
            print(f"   Sources: {', '.join(result['sources'])}")
        if result.get('error'):
            print(f"   Error: {result['error']}")
    
    # Test conversation memory
    print(f"\n📝 Conversation History ({len(rag.get_conversation_history())} messages):")
    history = rag.get_conversation_history()
    for msg in history[-4:]:  # Show last 4 messages
        role = "User" if msg["type"] == "human" else "Assistant"
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"   {role}: {content}")
    
    print("\n✅ RAG chain test completed successfully!")
    return True

if __name__ == "__main__":
    test_rag_chain()
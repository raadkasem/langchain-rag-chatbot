#!/usr/bin/env python3
"""
LangChain Agent that combines RAG capabilities with structured data tools.
This is the main chatbot agent that can answer questions from both knowledge base and structured data.
"""

import os
from typing import Dict, List, Any
from dotenv import load_dotenv

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import Tool

from rag_chain import RAGChain
from data_tools import DataTools


class ChatbotAgent:
    """Main chatbot agent that combines RAG and structured data capabilities."""

    def __init__(self, knowledge_base_path: str, data_directory: str = "data"):
        load_dotenv()

        self.knowledge_base_path = knowledge_base_path
        self.data_directory = data_directory

        # Initialize components
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.llm = ChatOpenAI(
            openai_api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.7
        )

        # Initialize RAG chain for knowledge base queries
        self.rag_chain = RAGChain(knowledge_base_path)

        # Initialize data tools for structured data queries
        self.data_tools = DataTools(data_directory)

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Create agent with all tools
        self.agent_executor = None
        self._setup_agent()

    def _create_rag_tool(self) -> Tool:
        """Create a tool that wraps the RAG chain for knowledge base queries."""

        def rag_query(query: str) -> str:
            """Query the knowledge base using RAG."""
            try:
                result = self.rag_chain.ask(query)
                answer = result.get("answer", "No answer generated")
                sources = result.get("sources", [])

                response = answer
                if sources:
                    response += f"\n\nSources: {', '.join(sources)}"

                return response
            except Exception as e:
                return f"Error querying knowledge base: {str(e)}"

        return Tool(
            name="query_knowledge_base",
            description="Search the company knowledge base for information about policies, procedures, technical documentation, FAQ, and general company information. Use this for questions about company policies, technical guides, product information, etc.",
            func=rag_query,
        )

    def _setup_agent(self):
        """Set up the agent with all available tools."""
        try:
            # Get all tools
            tools = []

            # Add RAG tool for knowledge base
            rag_tool = self._create_rag_tool()
            tools.append(rag_tool)

            # Add structured data tools
            data_tools = self.data_tools.get_langchain_tools()
            tools.extend(data_tools)

            # Create system prompt
            system_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a helpful AI assistant for a company. You have access to various tools to help answer questions:

1. Knowledge Base Tool: Use this for questions about company policies, procedures, technical documentation, FAQ, and general company information.

2. Employee Search: Use this to find information about specific employees or departments.

3. Customer Search: Use this to find information about customers, companies, or industries.

4. Department Statistics: Use this to get statistics about employee departments and salaries.

5. Revenue Statistics: Use this to get information about customer revenue and subscription tiers.

6. Data Query: Use this for specific data queries like counts, averages, or lists.

Guidelines:
- Always use the most appropriate tool for each question
- If a question could be answered by multiple tools, choose the most specific one
- Provide clear, helpful answers based on the tool results
- If you can't find relevant information, say so clearly
- Be professional and concise in your responses""",
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # Create the agent
            agent = create_openai_functions_agent(
                llm=self.llm, tools=tools, prompt=system_prompt
            )

            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate",
            )

            print(f"✅ Agent initialized with {len(tools)} tools")

        except Exception as e:
            print(f"❌ Error setting up agent: {str(e)}")

    def chat(self, message: str) -> Dict[str, Any]:
        """Process a chat message and return a response."""
        if not self.agent_executor:
            return {
                "response": "❌ Agent not properly initialized",
                "error": "Agent setup failed",
            }

        try:
            # Execute the agent
            result = self.agent_executor.invoke({"input": message})

            return {
                "response": result.get("output", "No response generated"),
                "intermediate_steps": result.get("intermediate_steps", []),
            }

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "response": "Sorry, I encountered an error while processing your message. Please try again.",
                "error": error_msg,
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
        # Also clear RAG chain memory
        self.rag_chain.clear_memory()
        print("🧹 Conversation memory cleared")

    def get_available_capabilities(self) -> List[str]:
        """Get a list of the agent's capabilities."""
        return [
            "🔍 Search company knowledge base (policies, procedures, technical docs, FAQ)",
            "👥 Search and find employee information",
            "🏢 Search and find customer information",
            "📊 Get department and salary statistics",
            "💰 Get customer revenue statistics",
            "🔢 Execute data queries (counts, averages, lists)",
            "💬 Maintain conversation context",
            "🧠 Remember previous questions in the conversation",
        ]


def test_chatbot_agent():
    """Test the complete chatbot agent."""
    print("🤖 Testing Chatbot Agent")
    print("=" * 50)

    # Initialize agent
    knowledge_base_path = os.path.join(
        os.path.dirname(__file__), "..", "knowledge_base"
    )
    agent = ChatbotAgent(knowledge_base_path)

    if not agent.agent_executor:
        print("❌ Agent initialization failed")
        return False

    # Show capabilities
    print("\n🎯 Agent Capabilities:")
    for capability in agent.get_available_capabilities():
        print(f"  {capability}")

    # Test different types of questions
    test_questions = [
        "What is the company's vacation policy?",
        "Who works in the Engineering department?",
        "Which customers are on the Enterprise tier?",
        "What's the average salary by department?",
        "How many employees do we have?",
        "What are our API rate limits?",
        "Can you find information about TechStart Inc?",
        "What's our total monthly revenue from customers?",
    ]

    print(f"\n💬 Testing with {len(test_questions)} questions:")

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 40)

        result = agent.chat(question)
        response = result.get("response", "No response")

        # Truncate long responses for readability
        if len(response) > 300:
            response = response[:300] + "..."

        print(f"Answer: {response}")

        if result.get("error"):
            print(f"Error: {result['error']}")

    # Test conversation memory
    print(f"\n📚 Conversation History:")
    history = agent.get_conversation_history()
    print(f"Total messages: {len(history)}")

    print("\n✅ Chatbot agent test completed!")
    return True


if __name__ == "__main__":
    test_chatbot_agent()

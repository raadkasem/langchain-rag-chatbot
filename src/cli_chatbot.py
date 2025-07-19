#!/usr/bin/env python3
"""
Command-line interface for the LangChain chatbot.
This provides an interactive chat interface for testing and using the chatbot.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

from chatbot_agent import ChatbotAgent


class CLIChatbot:
    """Command-line interface for the chatbot agent."""

    def __init__(self):
        self.agent: Optional[ChatbotAgent] = None
        self.running = False

    def initialize_agent(self) -> bool:
        """Initialize the chatbot agent."""
        try:
            print("🚀 Initializing chatbot agent...")

            # Check for API key
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ OPENAI_API_KEY not found!")
                print("Please copy .env.example to .env and add your OpenAI API key.")
                return False

            # Initialize agent
            knowledge_base_path = os.path.join(
                os.path.dirname(__file__), "..", "knowledge_base"
            )
            self.agent = ChatbotAgent(knowledge_base_path)

            if not self.agent.agent_executor:
                print("❌ Failed to initialize agent")
                return False

            print("✅ Chatbot agent initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ Error initializing agent: {str(e)}")
            return False

    def show_welcome_message(self):
        """Display welcome message and instructions."""
        print("\n" + "=" * 60)
        print("🤖 LANGCHAIN CHATBOT - Interactive Mode")
        print("=" * 60)
        print("\n👋 Welcome! I'm your AI assistant with access to:")

        if self.agent:
            capabilities = self.agent.get_available_capabilities()
            for capability in capabilities:
                print(f"  {capability}")

        print("\n📝 Commands:")
        print("  • Type your question and press Enter")
        print("  • '/help' - Show this help message")
        print("  • '/clear' - Clear conversation history")
        print("  • '/history' - Show conversation history")
        print("  • '/quit' or 'exit' - Exit the chatbot")
        print("\n💡 Example questions:")
        print("  • What is the company's vacation policy?")
        print("  • Who works in the Engineering department?")
        print("  • What customers are on the Enterprise tier?")
        print("  • What's the average salary by department?")
        print("\n" + "-" * 60)

    def show_help(self):
        """Show help information."""
        print("\n📚 HELP - Available Commands:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear conversation history")
        print("  /history  - Show conversation history")
        print("  /quit     - Exit the chatbot")
        print("  exit      - Exit the chatbot")

        print("\n🎯 What I can help you with:")
        if self.agent:
            capabilities = self.agent.get_available_capabilities()
            for capability in capabilities:
                print(f"  {capability}")

        print("\n💡 Tips:")
        print("  • Ask specific questions for better results")
        print("  • I can search both documents and data")
        print("  • I remember our conversation context")
        print("  • Try asking follow-up questions!")

    def show_history(self):
        """Show conversation history."""
        if not self.agent:
            print("❌ Agent not initialized")
            return

        history = self.agent.get_conversation_history()

        if not history:
            print("📝 No conversation history yet. Start by asking a question!")
            return

        print(f"\n📚 Conversation History ({len(history)} messages):")
        print("-" * 40)

        for i, message in enumerate(history, 1):
            role = "You" if message["type"] == "human" else "Assistant"
            content = message["content"]

            # Truncate long messages for display
            if len(content) > 150:
                content = content[:150] + "..."

            print(f"{i}. {role}: {content}")

        print("-" * 40)

    def clear_history(self):
        """Clear conversation history."""
        if not self.agent:
            print("❌ Agent not initialized")
            return

        self.agent.clear_memory()
        print("🧹 Conversation history cleared!")

    def process_command(self, user_input: str) -> bool:
        """Process special commands. Returns True if it was a command, False otherwise."""
        command = user_input.strip().lower()

        if command in ["/quit", "exit", "/exit"]:
            print("\n👋 Goodbye! Thanks for using the chatbot!")
            return True

        elif command in ["/help", "help"]:
            self.show_help()
            return True

        elif command in ["/clear", "clear"]:
            self.clear_history()
            return True

        elif command in ["/history", "history"]:
            self.show_history()
            return True

        return False

    def chat_loop(self):
        """Main chat interaction loop."""
        if not self.agent:
            print("❌ Agent not initialized. Cannot start chat.")
            return

        self.running = True

        try:
            while self.running:
                # Get user input
                try:
                    user_input = input("\n💬 You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Check for commands
                if self.process_command(user_input):
                    if user_input.lower() in ["/quit", "exit", "/exit"]:
                        break
                    continue

                # Process as regular chat message
                print("\n🤖 Assistant: ", end="", flush=True)

                try:
                    result = self.agent.chat(user_input)
                    response = result.get(
                        "response", "Sorry, I could not generate a response."
                    )

                    print(response)

                    if result.get("error"):
                        print(f"\n⚠️  Note: {result['error']}")

                except Exception as e:
                    print(f"Sorry, I encountered an error: {str(e)}")
                    print("Please try rephrasing your question.")

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")

        finally:
            self.running = False

    def run(self):
        """Run the CLI chatbot."""
        print("🤖 LangChain Chatbot CLI")
        print("Starting up...")

        # Initialize agent
        if not self.initialize_agent():
            print("❌ Failed to start chatbot. Please check your configuration.")
            sys.exit(1)

        # Show welcome message
        self.show_welcome_message()

        # Start chat loop
        try:
            self.chat_loop()
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Please check your configuration and try again.")


def main():
    """Main entry point for the CLI chatbot."""
    chatbot = CLIChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()

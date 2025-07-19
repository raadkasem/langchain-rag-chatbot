import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    
    print("🤖 LangChain Chatbot Project")
    print("=" * 40)
    print("\nAvailable options:")
    print("1. Run Interactive Chatbot (CLI)")
    print("2. Test Basic LLM Integration")
    print("3. Test Document Processing")
    print("4. Test RAG Chain")
    print("5. Test Data Tools")
    print("6. Test Complete Agent")
    print("7. Setup Database")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (0-7): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            
            elif choice == "1":
                print("\n🚀 Starting Interactive Chatbot...")
                from cli_chatbot import main as cli_main
                cli_main()
                break
            
            elif choice == "2":
                print("\n🧪 Testing Basic LLM Integration...")
                from basic_llm_test import test_basic_llm, test_different_prompts
                if test_basic_llm():
                    test_different_prompts()
            
            elif choice == "3":
                print("\n🧪 Testing Document Processing...")
                from document_processor import test_document_processor
                test_document_processor()
            
            elif choice == "4":
                print("\n🧪 Testing RAG Chain...")
                from rag_chain import test_rag_chain
                test_rag_chain()
            
            elif choice == "5":
                print("\n🧪 Testing Data Tools...")
                from data_tools import test_data_tools
                test_data_tools()
            
            elif choice == "6":
                print("\n🧪 Testing Complete Agent...")
                from chatbot_agent import test_chatbot_agent
                test_chatbot_agent()
            
            elif choice == "7":
                print("\n🔧 Setting up Database...")
                from create_database import create_database
                create_database()
            
            else:
                print("❌ Invalid option. Please choose 0-7.")
        
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
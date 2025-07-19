# LangChain Chatbot Project

A comprehensive chatbot implementation using LangChain that combines Retrieval-Augmented Generation (RAG) with structured data querying capabilities.

**By:** Raad Kasem  
**Contact:** mail@raadkasem.dev  
**Website:** [raadkasem.dev](https://raadkasem.dev)

## 🎯 Features

- **RAG (Retrieval-Augmented Generation)**: Query company knowledge base documents
- **Structured Data Access**: Search employees, customers, and get analytics
- **Conversational Memory**: Maintains context throughout conversations
- **Multiple Data Sources**: Works with both unstructured (documents) and structured (CSV/SQLite) data
- **Interactive CLI**: Easy-to-use command-line interface
- **Modular Architecture**: Well-organized, testable components

## 📁 Project Structure

```
langchain-rag-chatbot/
├── src/                          # Source code
│   ├── basic_llm_test.py         # Basic LLM integration testing
│   ├── document_processor.py     # Document loading and processing
│   ├── rag_chain.py             # RAG chain implementation
│   ├── data_tools.py            # Structured data query tools
│   ├── chatbot_agent.py         # Main agent combining all capabilities
│   ├── cli_chatbot.py           # Command-line interface
│   └── create_database.py       # Database setup utility
├── knowledge_base/               # Document knowledge base
│   ├── company_policies.md
│   ├── technical_documentation.md
│   ├── product_information.md
│   ├── faq.md
│   └── onboarding_guide.md
├── data/                        # Structured data
│   ├── employees.csv
│   ├── customers.csv
│   ├── company.db              # SQLite database (auto-generated)
│   └── chroma_db/              # Vector store (auto-generated)
├── .venv/                      # Python virtual environment
├── .env.example                 # Environment configuration template
├── requirements.txt             # Python dependencies
└── run_chatbot.py              # Main entry point
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository (if needed)
# cd langchain-rag-chatbot

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Chatbot

```bash
# Method 1: Use the main menu
python run_chatbot.py

# Method 2: Run chatbot directly  
python src/cli_chatbot.py
```

## 🧪 Testing Components

The project includes individual test functions for each component:

```bash
# Test basic LLM integration
python src/basic_llm_test.py

# Test document processing
python src/document_processor.py

# Test RAG chain
python src/rag_chain.py

# Test data tools
python src/data_tools.py

# Test complete agent
python src/chatbot_agent.py
```

## 💬 Example Conversations

### Knowledge Base Queries
```
You: What is the company's vacation policy?
Assistant: According to the company policy, all full-time employees receive 15 days of paid vacation per year. Vacation requests must be submitted at least 2 weeks in advance through the HR portal.
```

### Employee Data Queries
```
You: Who works in the Engineering department?
Assistant: Found 4 employee(s) matching 'engineering':
• John Smith - Senior Developer in Engineering
• Mike Davis - Team Lead in Engineering  
• Lisa Garcia - Junior Developer in Engineering
• Chris Taylor - DevOps Engineer in Engineering
```

### Analytics Queries
```
You: What's the average salary by department?
Assistant: Department Statistics:
• Engineering: 4 employees, Average Salary: $88,750
• Marketing: 2 employees, Average Salary: $62,500
• HR: 1 employees, Average Salary: $60,000
```

## 🛠️ Architecture Overview

### Components

1. **Document Processor** (`document_processor.py`)
   - Loads documents from knowledge base
   - Splits text into chunks
   - Creates embeddings and vector store
   - Provides similarity search

2. **RAG Chain** (`rag_chain.py`) 
   - Combines retrieval with generation
   - Maintains conversation memory
   - Provides contextual answers

3. **Data Tools** (`data_tools.py`)
   - Employee and customer search
   - Department and revenue statistics
   - SQL-like data queries

4. **Chatbot Agent** (`chatbot_agent.py`)
   - Orchestrates all components
   - Routes questions to appropriate tools
   - Maintains unified conversation flow

5. **CLI Interface** (`cli_chatbot.py`)
   - Interactive command-line interface
   - Command processing
   - User-friendly interactions

### Data Flow

```
User Question → Agent → Tool Selection → Data/Document Retrieval → LLM Processing → Response
```

## 🔧 Customization

### Adding New Documents
1. Add markdown or text files to `knowledge_base/`
2. Restart the chatbot to reload documents
3. Or use the `/clear` command and ask questions about new content

### Adding New Data Sources
1. Add CSV files to `data/` directory
2. Update `create_database.py` to include new tables
3. Extend `data_tools.py` with new search functions
4. Add new tools to the agent in `chatbot_agent.py`

### Modifying the Agent Behavior
- Edit system prompts in `chatbot_agent.py`
- Adjust tool descriptions for better routing
- Modify conversation memory settings

## 📦 Dependencies

- **openai**: OpenAI API integration
- **langchain**: LangChain framework
- **pandas**: Data manipulation
- **chromadb**: Vector database
- **python-dotenv**: Environment variable management

## 🚫 Limitations

- Requires OpenAI API key (costs apply)
- Knowledge base updates require restart
- Limited to structured data schemas defined in code
- Single-user conversation (no session management)

## 🔍 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Ensure `.env` file exists with valid API key
   - Check that the virtual environment is activated

2. **"No documents loaded"**
   - Verify `knowledge_base/` directory contains `.md` or `.txt` files
   - Check file permissions

3. **"Database not available"**
   - Run `python src/create_database.py` to recreate database
   - Ensure `data/` directory exists

4. **Vector store errors**
   - Delete `data/chroma_db/` directory to force recreation
   - Restart the chatbot

## 🎓 Learning Objectives Achieved

This project demonstrates:

✅ **Environment Setup**: Python virtual environment and dependencies  
✅ **LLM Integration**: Basic OpenAI API usage and experimentation  
✅ **Document Processing**: LangChain loaders, splitters, and embeddings  
✅ **Vector Stores**: ChromaDB for similarity search  
✅ **RAG Implementation**: Retrieval-augmented generation pipeline  
✅ **Custom Tools**: Structured data query capabilities  
✅ **Agent Architecture**: Multi-tool LangChain agent  
✅ **User Interface**: Interactive CLI for testing  
✅ **Best Practices**: Modular code, error handling, testing  

## 📚 TO DO:

- Add web interface using Streamlit or FastAPI
- Implement user authentication and session management
- Add support for file uploads and dynamic knowledge base updates
- Integrate with external APIs and databases
- Add more sophisticated SQL query generation
- Implement caching and performance optimizations
# PDF RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot for answering questions about PDF documents using LangChain, Groq AI, and embeddings.

## Features

- Processes PDF documents from a folder
- Uses Llama3-8B via Groq API for responses
- Utilizes HuggingFace or OpenAI embeddings for semantic search
- Includes LLM fallback when documents don't have relevant information
- Process visualization to see how RAG works internally
- Streamlit interface for chatting with your documents
- Page references in responses for attribution

## Requirements

- Python 3.8+
- Groq API key ([sign up here](https://console.groq.com/))
- HuggingFace API token or OpenAI API key

## Setup

### Automated Setup (Recommended)

We provide a setup script to automate the installation process:

```bash
python setup.py
```

This will:
1. Create a virtual environment
2. Install all required packages
3. Create a template .env file
4. Create the pdf directory if it doesn't exist

### Manual Setup

1. Clone the repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
pip install -U sentence-transformers langchain-huggingface
```

3. Create a `.env` file with your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
# Uncomment and set this if HuggingFace embeddings don't work
# OPENAI_API_KEY=your_openai_api_key_here
```

4. Add your PDF files to the `pdf` folder

## Usage

### Running the Streamlit App

Run the Streamlit app with:

```bash
streamlit run app.py
```

This will start the web interface where you can:
- Ask questions about your PDFs
- Choose between HuggingFace or OpenAI embeddings in the sidebar
- Enable process visualization to see how the RAG system works

### Using the RAG System Programmatically

You can also use the RAG system programmatically:

```python
from rag_system import RAGSystem

# Initialize the RAG system
rag = RAGSystem()
rag.load_documents()
rag.create_vectorstore()
rag.setup_rag_chain()

# Query the system
answer = rag.query("What are the key points in the document?")
print(answer)
```

If you need to use OpenAI embeddings instead:

```python
from fallback_openai_embeddings import RAGSystem as OpenAIRAGSystem

# Initialize with OpenAI embeddings
rag = OpenAIRAGSystem()
# ... rest of the code is the same
```

## How It Works

1. **Document Loading**: The system loads PDF documents from the specified folder
2. **Chunking**: Documents are split into manageable chunks for processing
3. **Embedding**: Each chunk is embedded using either HuggingFace or OpenAI embeddings
4. **Indexing**: Embeddings are stored in a ChromaDB vector database
5. **Retrieval**: When a question is asked, relevant chunks are retrieved
6. **Generation**: Groq's Llama3 LLM generates an answer based on the retrieved context
7. **Fallback**: If no relevant documents are found, the system uses the LLM's general knowledge

## Process Visualization

You can enable process visualization in the Streamlit app by checking "Show RAG Process Details" in the sidebar. This feature shows:
- Process log with timestamps
- Loaded document information
- Vector store statistics
- Retrieved documents for each query with relevance scores

See `README_PROCESS_TRACKING.md` for more details.

## Troubleshooting

### HuggingFace Embeddings Issues

If you encounter errors with HuggingFace embeddings:

1. Make sure you have the correct packages installed:
   ```bash
   pip install -U sentence-transformers langchain-huggingface
   ```

2. Check if your system can run the models locally:
   ```bash
   python test_embeddings.py
   ```

3. If issues persist, switch to OpenAI embeddings:
   - Uncomment and add your OpenAI API key in the `.env` file
   - Select "OpenAI API" in the embedding model dropdown in the Streamlit app

### PDF Loading Issues

- Make sure your PDFs are not corrupted
- Ensure they are placed in the 'pdf' folder
- Check if the PDFs are password-protected

### ChromaDB Issues

If you encounter "no such table: collections" error:
- Try deleting the '.chroma' directory if it exists and restart the app
- Make sure you don't have multiple instances of the app running

## License

MIT 
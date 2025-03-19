import streamlit as st
from dotenv import load_dotenv
import os
import time
import warnings
from typing import List, Dict, Union, Tuple, Optional
import traceback
import pandas as pd

# Suppress PyTorch/Streamlit warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables for process tracking
if "process_log" not in st.session_state:
    st.session_state.process_log = []

if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []

if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

if "vectorstore_stats" not in st.session_state:
    st.session_state.vectorstore_stats = {}

if "show_process" not in st.session_state:
    st.session_state.show_process = False

if "used_fallback" not in st.session_state:
    st.session_state.used_fallback = False

# Sidebar with information
with st.sidebar:
    st.title("📚 PDF RAG Chatbot")
    st.markdown("""
    This chatbot uses Retrieval Augmented Generation (RAG) to answer questions about your PDF documents.
    
    **Features:**
    - Processes PDF documents from the 'pdf' folder
    - Uses Llama3-8B via Groq for responses
    - Utilizes embeddings for semantic search
    - Includes source references in answers
    - Falls back to LLM knowledge when PDFs don't have relevant info
    
    **PDFs loaded:**
    """)
    
    # List the PDFs that have been loaded
    pdf_folder = "pdf"
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    for pdf in pdf_files:
        st.write(f"- {pdf}")
    
    # Add embedding model selection
    st.markdown("---")
    embedding_option = st.radio(
        "Choose Embedding Model:",
        ["HuggingFace", "OpenAI API"],
        index=0
    )
    
    # Add toggle for process display
    st.markdown("---")
    st.write("**Process Visibility:**")
    show_process = st.checkbox("Show RAG Process Details", value=st.session_state.show_process)
    st.session_state.show_process = show_process
    
    st.markdown("---")
    st.markdown("Made with ❤️ using LangChain & Groq LLM")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom RAG system wrapper for tracking processes
class ProcessTrackedRAG:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.log_process("Initialized RAG system")
    
    def log_process(self, message, level="info"):
        """Log a process message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.process_log.append({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })
    
    def load_documents(self):
        """Track document loading process"""
        self.log_process("Starting document loading")
        result = self.rag.load_documents()
        
        # Store document information for display
        st.session_state.loaded_docs = []
        for i, doc in enumerate(self.rag.docs[:20]):  # Limit to first 20 docs for display
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 0)
            filename = os.path.basename(source)
            
            # Store truncated content for display
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            st.session_state.loaded_docs.append({
                "id": i,
                "filename": filename,
                "page": page + 1,
                "chars": len(doc.page_content),
                "content_preview": content_preview
            })
        
        total_docs = len(self.rag.docs)
        self.log_process(f"Loaded and processed {total_docs} document chunks")
        return result
    
    def create_vectorstore(self):
        """Track vectorstore creation process"""
        self.log_process("Creating vectorstore and computing embeddings")
        result = self.rag.create_vectorstore()
        
        # Store vectorstore statistics
        collection_name = self.rag.vectorstore._collection.name
        doc_count = len(self.rag.docs)
        embedding_dim = self.rag.embed_model.embed_query("test").shape[0] if hasattr(self.rag.embed_model.embed_query("test"), "shape") else len(self.rag.embed_model.embed_query("test"))
        
        st.session_state.vectorstore_stats = {
            "collection_name": collection_name,
            "document_count": doc_count,
            "embedding_dimension": embedding_dim,
            "retriever_k": self.rag.retriever.search_kwargs.get("k", 4)
        }
        
        self.log_process("Vectorstore created successfully")
        return result
    
    def setup_rag_chain(self):
        """Track RAG chain setup process"""
        self.log_process("Setting up RAG chain with LLM fallback")
        result = self.rag.setup_rag_chain()
        self.log_process("RAG chain setup complete")
        return result
    
    def query(self, question):
        """Track query process and capture retrieved documents"""
        self.log_process(f"Processing query: '{question}'")
        
        # Reset fallback flag
        st.session_state.used_fallback = False
        
        # Get the retrieved documents before generating the answer
        retriever = self.rag.retriever
        try:
            # Use invoke() method instead of the deprecated get_relevant_documents()
            retrieved_docs = retriever.invoke(question)
        except AttributeError:
            # Fallback to the old method if invoke isn't available
            retrieved_docs = retriever.get_relevant_documents(question)
        
        # Check if we have relevant documents
        has_relevant_context = False
        if retrieved_docs and len(retrieved_docs) > 0:
            total_content = ''.join([doc.page_content for doc in retrieved_docs])
            has_relevant_context = len(total_content) > 100
        
        # Store retrieved documents for display
        st.session_state.retrieved_docs = []
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 0)
            filename = os.path.basename(source)
            
            # Calculate a simple relevance score for visualization (just for UI purposes)
            relevance = 1.0 - (i * 0.2)  # Simple decreasing score
            
            st.session_state.retrieved_docs.append({
                "rank": i + 1,
                "filename": filename,
                "page": page + 1,
                "relevance": relevance,
                "content": doc.page_content
            })
        
        self.log_process(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        # Check if we're going to use fallback
        if not has_relevant_context:
            self.log_process("Insufficient context in documents, using LLM fallback", level="warning")
            st.session_state.used_fallback = True
            
        # Generate response
        self.log_process(f"Generating response with {'direct LLM' if not has_relevant_context else 'RAG pipeline'}")
        response = self.rag.query(question)
        self.log_process("Response generated successfully")
        
        return response

# Initialize RAG system
@st.cache_resource
def load_rag_system(embedding_option: str) -> Tuple[Optional[object], Optional[str]]:
    """Load the RAG system with the selected embedding model
    
    Args:
        embedding_option: The embedding option to use ("HuggingFace" or "OpenAI API")
        
    Returns:
        A tuple of (rag_system, error_message)
    """
    try:
        # Clear process log when loading a new system
        st.session_state.process_log = []
        
        if embedding_option == "OpenAI API":
            # Import fallback OpenAI implementation
            from fallback_openai_embeddings import RAGSystem as OpenAIRAGSystem
            base_rag = OpenAIRAGSystem()
            st.info("Using OpenAI embeddings")
        else:
            # Use the default HuggingFace implementation
            from rag_system import RAGSystem
            base_rag = RAGSystem()
            st.info("Using HuggingFace embeddings")
        
        # Wrap the RAG system with process tracking
        rag = ProcessTrackedRAG(base_rag)
        
        # Initialize the RAG pipeline
        rag.load_documents()
        rag.create_vectorstore()
        rag.setup_rag_chain()
        
        return rag, None
    except Exception as e:
        error_detail = traceback.format_exc()
        return None, f"Error: {str(e)}\n\nDetails: {error_detail}"

# Load the RAG system with progress
with st.spinner("Loading documents and setting up RAG system..."):
    rag_system, error = load_rag_system(embedding_option)
    
    if error:
        st.error(f"Failed to initialize RAG system")
        with st.expander("See error details"):
            st.code(error)
        
        # If the error is related to HuggingFace and we didn't already try OpenAI
        if "HuggingFace" in error and embedding_option != "OpenAI API":
            st.warning("Trying to fall back to OpenAI embeddings...")
            rag_system, fallback_error = load_rag_system("OpenAI API")
            
            if fallback_error:
                st.error("Fallback also failed")
                with st.expander("See fallback error details"):
                    st.code(fallback_error)
            else:
                st.success("System loaded successfully with OpenAI embeddings!")
    else:
        st.success("System loaded successfully!")

# Display chat messages
st.title("Chat with your PDFs 💬")
st.caption("Ask questions about your PDFs. The AI will include references to specific pages in its answers or use its own knowledge when needed.")

# Display RAG process details if enabled
if st.session_state.show_process:
    st.markdown("---")
    st.header("🔍 RAG Process Details")
    
    # Display process log in a collapsible section
    with st.expander("Process Log", expanded=False):
        log_df = pd.DataFrame(st.session_state.process_log)
        if not log_df.empty:
            st.dataframe(log_df[["timestamp", "message", "level"]], use_container_width=True)
        else:
            st.write("No process logs available yet.")
    
    # Display loaded documents in a collapsible section
    with st.expander("Loaded Documents", expanded=False):
        if st.session_state.loaded_docs:
            docs_df = pd.DataFrame(st.session_state.loaded_docs)
            st.dataframe(docs_df, use_container_width=True)
        else:
            st.write("No document information available.")
    
    # Display vectorstore statistics
    with st.expander("Vectorstore Information", expanded=False):
        if st.session_state.vectorstore_stats:
            stats = st.session_state.vectorstore_stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Document Chunks", stats.get("document_count", 0))
            with col2:
                st.metric("Embedding Dimension", stats.get("embedding_dimension", 0))
            with col3:
                st.metric("Retriever k", stats.get("retriever_k", 4))
            
            st.write(f"**Collection Name**: {stats.get('collection_name', 'unknown')}")
        else:
            st.write("No vectorstore information available.")
    
    # Display retrieved documents for the last query
    with st.expander("Last Query Retrieved Documents", expanded=True):
        if st.session_state.retrieved_docs:
            st.subheader("Documents Retrieved for Last Query")
            
            # Show whether fallback was used
            if st.session_state.used_fallback:
                st.warning("⚠️ Insufficient relevant context found in documents. LLM general knowledge was used as fallback.")
            
            # Show relevance bars for documents
            for doc in st.session_state.retrieved_docs:
                with st.container():
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.markdown(f"**Rank {doc['rank']}**")
                        st.progress(doc['relevance'])
                    with col2:
                        st.markdown(f"**Source**: {doc['filename']}, Page {doc['page']}")
                        st.text_area(
                            "Content", 
                            doc['content'], 
                            height=100, 
                            key=f"doc_{doc['rank']}_{doc['page']}"
                        )
        else:
            st.write("No documents have been retrieved yet. Ask a question to see retrieved documents.")

st.markdown("---")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show fallback indicator if this was a fallback response
        if message.get("used_fallback"):
            st.caption("ℹ️ Response generated using LLM's knowledge, not from your documents")

# Accept user input
if rag_system and (prompt := st.chat_input("Ask a question about your PDFs...")):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get response with error handling
        try:
            with st.spinner("Thinking..."):
                response = rag_system.query(prompt)
            
            # Simulate typing effect
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.01)  # Small delay to simulate typing
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history with fallback indicator
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "used_fallback": st.session_state.used_fallback
            })
            
            # Show fallback indicator if this was a fallback response
            if st.session_state.used_fallback:
                st.caption("ℹ️ Response generated using LLM's knowledge, not from your documents")
            
            # Refresh the page to update the process display if it's enabled
            if st.session_state.show_process:
                st.rerun()
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error_message}"})

# Add a bit more information at the bottom
st.markdown("---")
st.caption("Note: This chatbot's knowledge is limited to the content of the loaded PDFs and general LLM knowledge when needed.")

# Add troubleshooting information
with st.expander("Troubleshooting"):
    st.markdown("""
    ### Common issues:
    
    1. **HuggingFace embeddings don't work**:
       - Ensure you have the latest langchain-huggingface installed: `pip install -U langchain-huggingface`
       - Ensure you have sentence-transformers installed: `pip install sentence-transformers`
       - Try using OpenAI embeddings as a fallback
    
    2. **GROQ API issues**:
       - Check that your API key is valid
       - Ensure you have internet connectivity
       - Try a different LLM provider if needed
    
    3. **PDF loading issues**:
       - Make sure your PDFs are not corrupted
       - Ensure they are in the 'pdf' folder
       
    4. **PyTorch warnings in console**:
       - These are harmless warnings from Streamlit trying to monitor PyTorch internals
       - They don't affect the functionality of the application
       
    5. **"no such table: collections" error**:
       - This means ChromaDB hasn't been initialized properly
       - Try deleting the '.chroma' directory if it exists and restart the app
       - Make sure you don't have multiple instances of the app running
    """) 
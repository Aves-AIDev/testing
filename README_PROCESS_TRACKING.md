# RAG Process Visualization

This feature allows you to visualize and understand the Retrieval Augmented Generation (RAG) process step by step. By enabling the "Show RAG Process Details" checkbox in the sidebar, you can view detailed information about each stage of the RAG pipeline.

## Features

### 1. Process Log

The process log displays a chronological record of all the steps taken by the RAG system, including:
- Document loading
- Vector store creation
- Embedding computation
- Document retrieval
- Response generation

Each log entry includes a timestamp, making it easy to track the time taken by each step.

### 2. Loaded Documents

This section shows sample information about the documents loaded from your PDFs:
- Filename
- Page number
- Character count
- Content preview

This helps you understand what content is available for the RAG system to retrieve from.

### 3. Vector Store Information

The vector store information panel displays key metrics about your embeddings:
- Total number of document chunks
- Embedding dimension size
- Retriever "k" value (number of documents retrieved per query)
- Collection name

These statistics give you insights into the size and configuration of your vector database.

### 4. Retrieved Documents

For each query, this section shows the documents retrieved by the system, ranked by relevance:
- Visual relevance indicators
- Source file and page number
- Complete document content

This helps you understand why the system generated a particular response and what information it considered most relevant.

## How to Use

1. Enable the process visualization by checking "Show RAG Process Details" in the sidebar
2. Ask a question in the chat interface
3. Explore the detailed process information after receiving a response
4. Review which documents were retrieved and how they influenced the answer

This feature is particularly useful for:
- Debugging retrieval issues
- Understanding why certain answers are generated
- Assessing the quality of your document processing
- Optimizing your RAG pipeline parameters

## Technical Details

The process tracking is implemented by wrapping the core RAG system in a `ProcessTrackedRAG` class that:
- Logs each step with timestamps
- Captures intermediate results
- Stores statistics about the vector store
- Intercepts document retrieval to capture and rank results

This is all done without modifying the underlying RAG system, making it easy to enable or disable as needed. 
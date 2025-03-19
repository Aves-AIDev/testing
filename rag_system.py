import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, pdf_folder: str = "pdf"):
        """Initialize the RAG system with documents from the given folder"""
        self.pdf_folder = pdf_folder
        self.docs = []
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.direct_llm_chain = None
        
        try:
            # Using HuggingFaceEmbeddings from langchain_huggingface package
            # Be careful with the parameters - some might not be supported
            self.embed_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            print("Successfully initialized HuggingFace embeddings")
        except Exception as e:
            print(f"Error initializing HuggingFace embeddings: {e}")
            # If HuggingFace embeddings fail, we'll handle the fallback in the app
            raise
        
        # Initialize LLM using Groq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        self.llm = ChatGroq(model="llama3-8b-8192")
        
    def load_documents(self):
        """Load and split all PDF documents from the folder"""
        print(f"Loading documents from {self.pdf_folder}...")
        
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_folder}")
        
        # Load each PDF
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            loader = PyPDFLoader(os.path.join(self.pdf_folder, pdf_file))
            self.docs.extend(loader.load())
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.docs = text_splitter.split_documents(self.docs)
        print(f"Documents processed: {len(self.docs)}")
        return self.docs
    
    def create_vectorstore(self):
        """Create a vectorstore from the documents"""
        if not self.docs:
            self.load_documents()
            
        print("Creating vectorstore...")
        self.vectorstore = Chroma.from_documents(
            documents=self.docs, 
            embedding=self.embed_model,
            collection_name="pdf_rag"
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 documents
        )
        print("Vectorstore created successfully")
        return self.vectorstore
    
    def setup_rag_chain(self):
        """Set up the RAG pipeline"""
        if not self.retriever:
            self.create_vectorstore()
            
        # Define the RAG prompt template
        RAG_SYSTEM_PROMPT = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the human's questions.
        If you don't know the answer based on the context, say that the information is not in the provided documents.
        
        For each fact or piece of information you include in your answer, cite the source by mentioning the filename 
        and the page number in square brackets like [filename, page X] at the end of the relevant sentence or paragraph.
        
        Context:
        ```
        {context}
        ```
        """
        
        # Define the direct LLM prompt template for fallback
        DIRECT_SYSTEM_PROMPT = """
        You are a helpful assistant answering questions about AI, technology, and general knowledge.
        
        The user asked a question that couldn't be answered using the information in their documents.
        Please provide a helpful general answer based on your knowledge.
        
        Make it clear in your response that you're providing information from your general knowledge,
        rather than from their specific documents.
        
        Question: {input}
        """
        
        RAG_HUMAN_PROMPT = "{input}"
        DIRECT_HUMAN_PROMPT = ""  # Empty since we include the question in the system prompt
        
        RAG_PROMPT = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT)
        ])
        
        DIRECT_PROMPT = ChatPromptTemplate.from_messages([
            ("system", DIRECT_SYSTEM_PROMPT)
        ])
        
        # Function to format documents
        def format_docs(docs: List[Document]) -> str:
            formatted_docs = []
            for doc in docs:
                # Extract filename and page number from metadata
                source = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page', 0)
                
                # Get just the filename without path
                filename = os.path.basename(source)
                
                # Format the document with source information
                formatted_doc = f"[{filename}, page {page+1}]\n{doc.page_content}"
                formatted_docs.append(formatted_doc)
            
            return "\n\n".join(formatted_docs)
        
        # Function to check if retrieved docs contain enough relevant information
        def has_relevant_context(docs):
            # If no documents were retrieved or total content length is very short, fallback to direct LLM
            if not docs or len(docs) == 0:
                return False
            
            # Calculate total content length
            total_content = ''.join([doc.page_content for doc in docs])
            
            # If the content is too short, it might not be relevant enough
            return len(total_content) > 100
        
        # Create the direct LLM chain for fallback
        self.direct_llm_chain = DIRECT_PROMPT | self.llm | StrOutputParser()
        
        # Create the RAG chain
        rag_qa_chain = (
            {
                "context": lambda x: x["retrieved_docs"],
                "input": lambda x: x["question"]
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        # Create a branch to decide whether to use RAG or fallback to direct LLM
        self.rag_chain = (
            {
                "question": RunnablePassthrough(),
                "retrieved_docs": self.retriever,
            }
            | RunnableBranch(
                (lambda x: has_relevant_context(x["retrieved_docs"]), rag_qa_chain),
                (lambda x: {"input": x["question"]} | self.direct_llm_chain)
            )
        )
        
        print("RAG chain setup complete with LLM fallback")
        return self.rag_chain
    
    def query(self, question: str) -> str:
        """Query the RAG system with a question"""
        if not self.rag_chain:
            self.setup_rag_chain()
            
        print(f"Querying: {question}")
        response = self.rag_chain.invoke(question)
        return response
    
if __name__ == "__main__":
    # Example usage
    rag = RAGSystem()
    rag.load_documents()
    rag.create_vectorstore()
    rag.setup_rag_chain()
    
    # Test with a sample question
    sample_question = "What are the economic impacts of generative AI?"
    answer = rag.query(sample_question)
    print(f"\nQuestion: {sample_question}")
    print(f"Answer: {answer}") 
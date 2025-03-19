import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def test_embeddings():
    print("Initializing embedding model...")
    try:
        embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        print("Embedding model initialized successfully!")
        
        # Test embeddings
        print("Testing embeddings with a sample text...")
        sample_text = "This is a test sentence to check if the embedding model works."
        embeddings = embed_model.embed_query(sample_text)
        
        print(f"Embedding dimension: {len(embeddings)}")
        print(f"First 5 values: {embeddings[:5]}")
        
        print("Embeddings test successful!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nRecommended solutions:")
        print("1. Make sure sentence-transformers is installed: pip install sentence-transformers")
        print("2. Try using OpenAI embeddings instead (see fallback_openai_embeddings.py)")
    
if __name__ == "__main__":
    test_embeddings() 
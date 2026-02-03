import time
from vector import retriever
from langchain_ollama.llms import OllamaLLM

def test_retriever(query: str, k: int = 5):
    """Test the retrieval time from the vector database."""
    start = time.time()
    try:
        # Use the retriever's __call__ method which is the standard interface
        docs = retriever.invoke(query)
        duration = time.time() - start
        print(f"\n✅ Retrieval test (top {k} docs):")
        print(f"Time taken: {duration:.2f} seconds")
        print(f"Number of documents retrieved: {len(docs)}")
        for i, doc in enumerate(docs[:3]):  # Show first 3 docs
            print(f"\nDocument {i+1} (Score: {doc.metadata.get('score', 'N/A')}):")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
        return duration
    except Exception as e:
        print(f"❌ Retrieval failed: {str(e)}")
        return None

def test_model_response(prompt: str, model_name: str):
    """Test the model's response time."""
    model = OllamaLLM(model=model_name)
    start = time.time()
    try:
        response = model.invoke(prompt)
        duration = time.time() - start
        print(f"\n✅ Model response test (model: {model_name}):")
        print(f"Time taken: {duration:.2f} seconds")
        print("Response:", response[:500] + "..." if len(response) > 500 else response)
        return duration
    except Exception as e:
        print(f"❌ Model response failed: {str(e)}")
        return None

def main():
    test_queries = [
        "What are the legal rights of tenants?",
        "How to file a consumer complaint?",
        "What is the process for divorce in India?"
    ]
    
    model_name = "gemma3:1b"  # You can change this to test different models
    
    print("🚀 Starting performance tests...")
    print(f"Using model: {model_name}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {query}")
        print("="*50)
        
        # Test retrieval
        retrieval_time = test_retriever(query)
        
        # Test model response with a simple prompt
        model_time = test_model_response("Say 'Hello, world!'", model_name)
        
        # Test model response with the actual query
        full_prompt = f"Answer the following question concisely: {query}"
        full_response_time = test_model_response(full_prompt, model_name)
        
        # Print summary
        if retrieval_time is not None and model_time is not None and full_response_time is not None:
            print(f"\n📊 Summary for '{query}':")
            print(f"- Retrieval time: {retrieval_time:.2f}s")
            print(f"- Model response time (simple): {model_time:.2f}s")
            print(f"- Model response time (full): {full_response_time:.2f}s")
            print(f"- Total estimated time: {retrieval_time + full_response_time:.2f}s")

if __name__ == "__main__":
    main()

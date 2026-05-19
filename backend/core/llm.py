import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

def get_llm():
    """
    Returns an LLM instance based on environment variables.
    """
    use_cloud = os.getenv("USE_CLOUD_LLM", "false").lower() == "true"
    
    if use_cloud:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
    
    # Default to local Ollama
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    
    return ChatOllama(
        model=model, 
        base_url=base_url,
        temperature=0.1,    # Lower temperature for faster, more focused output
    )

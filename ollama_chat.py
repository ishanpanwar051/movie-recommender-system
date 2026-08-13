"""
Ollama Integration Example for Movie Recommender
Run: pip install ollama
Make sure Ollama is running: ollama serve (or it runs automatically)
"""
import ollama
from typing import Generator

def chat_with_movies(user_message: str, movie_context: str = "") -> str:
    """
    Chat with an LLM about movies using Ollama.
    
    Args:
        user_message: User's question or prompt
        movie_context: Optional context about movies (e.g., recommendations)
    
    Returns:
        AI response string
    """
    system_prompt = """You are a helpful movie recommendation assistant.
    You help users discover movies based on their preferences.
    Be concise and provide specific movie suggestions with brief explanations."""
    
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    if movie_context:
        messages.append({
            "role": "user", 
            "content": f"Context about available movies:\n{movie_context}"
        })
    
    messages.append({"role": "user", "content": user_message})
    
    response = ollama.chat(model="llama3.2", messages=messages)
    return response["message"]["content"]


def stream_chat(user_message: str) -> Generator[str, None, None]:
    """
    Stream response from Ollama for real-time output.
    """
    messages = [
        {
            "role": "system", 
            "content": "You are a movie recommendation assistant. Be helpful and concise."
        },
        {"role": "user", "content": user_message}
    ]
    
    stream = ollama.chat(model="llama3.2", messages=messages, stream=True)
    
    for chunk in stream:
        yield chunk["message"]["content"]


# Example usage
if __name__ == "__main__":
    # Simple test
    print("Testing Ollama connection...")
    
    response = chat_with_movies(
        "What are 3 sci-fi movies similar to Blade Runner?"
    )
    print("\nAI Response:")
    print(response)
    
    print("\n\nStreaming response:")
    for chunk in stream_chat("Recommend a movie for someone who likes The Matrix"):
        print(chunk, end="", flush=True)

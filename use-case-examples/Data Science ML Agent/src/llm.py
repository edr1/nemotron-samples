# llm.py - Simplified for beginners
import os
from typing import List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up HTTP session with retries
def create_session():
    """Create a session with retry logic for better reliability."""
    session = requests.Session()
    
    # Retry Strategy for Resilient Communication
    retry_strategy = Retry(
        total=3,  # Try 3 times
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these errors
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

class LLMClient:
    """Simple client to talk to AI models."""
    
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "nemotron-3.5-lightning:30b-a3b")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.6"))
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))
        self.session = create_session()

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        """Send messages to Ollama and return its assistant message."""
        if not self.ollama_url:
            raise ValueError("Set OLLAMA_BASE_URL and restart Streamlit to use Ollama chat.")

        endpoint = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if tools:
            payload["tools"] = tools

        try:
            response = self.session.post(endpoint, json=payload, timeout=(10, 300))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not reach Ollama at {endpoint}: {e}") from e

        if not response.ok:
            try:
                detail = response.json().get("error") or response.reason
            except ValueError:
                detail = response.reason
            raise RuntimeError(f"Ollama returned HTTP {response.status_code} for {self.model}: {detail}")

        return response.json()["message"]

def create_client() -> LLMClient:
    """Create a new LLM client."""
    return LLMClient()

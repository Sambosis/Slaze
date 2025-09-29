import logging
import os
import aiohttp
from typing import Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str:
        """Call the LLM with messages and return response."""
        pass

class OpenRouterClient(LLMClient):
    """OpenRouter API client."""
    
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/command-converter",
            "X-Title": "Command Converter"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"OpenRouter API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                if "choices" not in result or not result["choices"]:
                    raise RuntimeError("Invalid OpenRouter response format")
                
                return result["choices"][0]["message"]["content"]

class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
    
    async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"OpenAI API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                if "choices" not in result or not result["choices"]:
                    raise RuntimeError("Invalid OpenAI response format")
                
                return result["choices"][0]["message"]["content"]

class AnthropicClient(LLMClient):
    """Anthropic API client."""
    
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convert messages format for Anthropic
        system_content = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        
        if system_content:
            payload["system"] = system_content
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Anthropic API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                if "content" not in result or not result["content"]:
                    raise RuntimeError("Invalid Anthropic response format")
                
                return result["content"][0]["text"]

def create_llm_client(model: str) -> LLMClient:
    """Factory function to create appropriate LLM client based on model name."""
    # if model.startswith("anthropic/"):
    #     return OpenRouterClient(model)
    if model.startswith("openai/"):
        return OpenRouterClient(model)
    elif model.startswith("google/"):
        return OpenRouterClient(model)
    elif model.startswith("gpt-"):
        return OpenAIClient(model)
    elif model.startswith("claude-"):
        return AnthropicClient(model)
    else:
        # Default to OpenRouter for unknown models
        logger.warning(f"Unknown model format: {model}, defaulting to OpenRouter")
        return OpenRouterClient(model)
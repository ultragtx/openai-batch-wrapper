"""
Provider adapters for different API vendors.

Converts between OpenAI format (canonical) and vendor-specific formats.
Currently supports: OpenAI (native), with interfaces for future Gemini, Anthropic, etc.
"""

from typing import Any, Dict, List
from abc import ABC, abstractmethod


class ProviderAdapter(ABC):
    """Base class for API provider adapters."""
    
    @abstractmethod
    def to_provider_format(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI format request to provider-specific format.
        
        Args:
            openai_request: Request in OpenAI format
            
        Returns:
            Request in provider-specific format
        """
        pass
    
    @abstractmethod
    def from_provider_format(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert provider-specific response to OpenAI format.
        
        Args:
            provider_response: Response in provider-specific format
            
        Returns:
            Response in OpenAI format
        """
        pass
    
    @abstractmethod
    def get_batch_format(self) -> str:
        """Return the batch API format for this provider."""
        pass


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI (no conversion needed)."""
    
    def to_provider_format(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI format is canonical, no conversion needed."""
        return openai_request
    
    def from_provider_format(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI format is canonical, no conversion needed."""
        return provider_response
    
    def get_batch_format(self) -> str:
        """Return OpenAI batch format."""
        return "openai"


class GeminiAdapter(ProviderAdapter):
    """
    Adapter for Google Gemini API.
    
    Note: This is a placeholder. Actual implementation would need:
    - Convert messages format (role names, content structure)
    - Convert response format
    - Handle batch API differences (if Gemini supports batch)
    """
    
    def to_provider_format(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI format to Gemini format.
        
        Example conversions needed:
        - messages[].role: "user" -> "user", "assistant" -> "model", "system" -> prepend to first user message
        - messages[].content: Handle image_url format differently
        - response_format: Convert to Gemini's generationConfig
        """
        # Placeholder implementation
        gemini_request = {
            "contents": [],
            "generationConfig": {}
        }
        
        # TODO: Implement actual conversion
        raise NotImplementedError("Gemini adapter not yet implemented")
    
    def from_provider_format(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Gemini response to OpenAI format.
        
        Example conversions needed:
        - candidates[0].content.parts[0].text -> choices[0].message.content
        - Handle role, finish_reason, etc.
        """
        raise NotImplementedError("Gemini adapter not yet implemented")
    
    def get_batch_format(self) -> str:
        """Return Gemini batch format (if supported)."""
        return "gemini"


class AnthropicAdapter(ProviderAdapter):
    """
    Adapter for Anthropic Claude API.
    
    Note: This is a placeholder. Actual implementation would need:
    - Convert messages format
    - Handle system message differently (separate parameter)
    - Convert response format
    - Handle batch API (if Anthropic supports it)
    """
    
    def to_provider_format(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format to Anthropic format."""
        raise NotImplementedError("Anthropic adapter not yet implemented")
    
    def from_provider_format(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic response to OpenAI format."""
        raise NotImplementedError("Anthropic adapter not yet implemented")
    
    def get_batch_format(self) -> str:
        """Return Anthropic batch format (if supported)."""
        return "anthropic"


# Registry of adapters
_ADAPTERS = {
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "anthropic": AnthropicAdapter,
}


def get_adapter(provider: str) -> ProviderAdapter:
    """
    Get the appropriate adapter for a provider.
    
    Args:
        provider: Provider name (openai, gemini, anthropic)
        
    Returns:
        Provider adapter instance
        
    Example:
        >>> adapter = get_adapter("openai")
        >>> request = adapter.to_provider_format(openai_request)
    """
    provider = provider.lower()
    if provider not in _ADAPTERS:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(_ADAPTERS.keys())}")
    
    return _ADAPTERS[provider]()


def register_adapter(provider: str, adapter_class: type):
    """
    Register a custom adapter.
    
    Args:
        provider: Provider name
        adapter_class: Adapter class (must inherit from ProviderAdapter)
    """
    if not issubclass(adapter_class, ProviderAdapter):
        raise TypeError("adapter_class must inherit from ProviderAdapter")
    _ADAPTERS[provider.lower()] = adapter_class


def convert_batch_request(
    openai_request: Dict[str, Any],
    target_provider: str = "openai"
) -> Dict[str, Any]:
    """
    Convert an OpenAI-format batch request to target provider format.
    
    Args:
        openai_request: Batch request in OpenAI format
        target_provider: Target provider name
        
    Returns:
        Batch request in target provider format
    """
    adapter = get_adapter(target_provider)
    
    # Convert the body
    if "body" in openai_request:
        openai_request["body"] = adapter.to_provider_format(openai_request["body"])
    
    return openai_request


def convert_batch_response(
    provider_response: Dict[str, Any],
    source_provider: str = "openai"
) -> Dict[str, Any]:
    """
    Convert a provider-specific batch response to OpenAI format.
    
    Args:
        provider_response: Batch response in provider format
        source_provider: Source provider name
        
    Returns:
        Batch response in OpenAI format
    """
    adapter = get_adapter(source_provider)
    
    # Convert the response body
    if "response" in provider_response and "body" in provider_response["response"]:
        provider_response["response"]["body"] = adapter.from_provider_format(
            provider_response["response"]["body"]
        )
    
    return provider_response

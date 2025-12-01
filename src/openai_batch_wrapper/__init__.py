"""
OpenAI Batch Wrapper - Transparent cache and batch support for OpenAI API.

This package provides a drop-in wrapper for the OpenAI Python client that adds:
- Response caching to avoid duplicate API calls
- Batch API integration for cost-effective bulk processing
- Multiple operation modes (realtime, batch_write, cache_first)

Quick Start:
    >>> from openai_batch_wrapper import create_wrapped_client
    >>> 
    >>> # Create client (drop-in replacement for AsyncOpenAI)
    >>> client = create_wrapped_client(
    ...     api_key="sk-...",
    ...     mode="realtime",  # or "batch_write", "cache_first"
    ...     model="gpt-4o"
    ... )
    >>> 
    >>> # Use exactly like AsyncOpenAI
    >>> response = await client.chat.completions.parse(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     model="gpt-4o",
    ...     response_format=MyResponseModel
    ... )

Operation Modes:
    - realtime: Normal API calls with caching (default)
    - batch_write: Queue requests for batch processing
    - cache_first: Use cached responses only, queue misses

Batch Workflow:
    1. Run with MODE=batch_write to queue requests
    2. batch-prepare: Convert queue to batch files
    3. batch-upload: Upload to OpenAI Batch API
    4. batch-download: Download completed results
    5. batch-to-cache: Convert results to cache
    6. Run with MODE=cache_first to use cached results
"""

__version__ = "0.1.0"

from .wrapper import (
    WrappedAsyncOpenAI,
    WrappedChat,
    WrappedChatCompletions,
    MockChatCompletion,
    create_wrapped_client,
    get_queue_stats,
)

from .cache_manager import CacheManager
from .batch_queue import BatchQueueManager
from .batch_manager import BatchManager
from .provider_adapters import (
    ProviderAdapter,
    OpenAIAdapter,
    get_adapter,
    convert_batch_request,
    convert_batch_response,
)

__all__ = [
    # Main wrapper
    "create_wrapped_client",
    "WrappedAsyncOpenAI",
    "WrappedChat",
    "WrappedChatCompletions",
    "MockChatCompletion",
    "get_queue_stats",
    # Managers
    "CacheManager",
    "BatchQueueManager",
    "BatchManager",
    # Provider adapters
    "ProviderAdapter",
    "OpenAIAdapter",
    "get_adapter",
    "convert_batch_request",
    "convert_batch_response",
]

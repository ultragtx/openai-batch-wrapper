"""
OpenAI Client Wrapper with transparent cache and batch support.

This wrapper intercepts OpenAI API calls and routes them based on mode:
- realtime: Normal API call with caching (default)
- batch_write: Save request to batch queue, return cache or error
- cache_first: Return cached response or save to batch queue

The wrapper is transparent - existing code doesn't need modification.
"""

import os
import json
import asyncio
import inspect
from typing import Any, Dict, List, Optional
from datetime import datetime

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from .cache_manager import CacheManager
from .batch_queue import BatchQueueManager


class MockChatCompletion:
    """Mock ChatCompletion response for cache hits."""
    
    def __init__(self, content: str, model: str = "cached", tool_calls: Optional[List[Dict[str, Any]]] = None):
        self.id = "cached"
        self.model = model
        self.object = "chat.completion"
        self.created = int(datetime.now().timestamp())
        
        # Build tool_calls if provided
        tool_calls_objs = None
        if tool_calls:
            tool_calls_objs = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=Function(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"]
                    )
                )
                for tc in tool_calls
            ]
        
        message = ChatCompletionMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls_objs
        )
        message.parsed = None  # Will be set by wrapper
        
        finish_reason = "tool_calls" if tool_calls else "stop"
        self.choices = [
            Choice(
                index=0,
                message=message,
                finish_reason=finish_reason
            )
        ]
        
        self.usage = CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )


class WrappedChatCompletions:
    """Wrapper for chat.completions with cache and batch support."""
    
    def __init__(
        self, 
        original_completions: Any,
        mode: str,
        cache_mgr: CacheManager,
        queue_mgr: BatchQueueManager,
        model: str = "",
        semaphore: Optional[asyncio.Semaphore] = None
    ):
        self._original = original_completions
        self.mode = mode
        self.cache_mgr = cache_mgr
        self.queue_mgr = queue_mgr
        self.model = model
        self.semaphore = semaphore
        
        # Get valid API parameters from the original parse method signature
        self._api_param_names = self._get_api_param_names()
    
    def _get_api_param_names(self) -> set:
        """Extract parameter names from the original parse method."""
        try:
            sig = inspect.signature(self._original.parse)
            # Get all parameter names except 'self' and 'kwargs'
            param_names = {
                name for name, param in sig.parameters.items()
                if name not in ('self', 'kwargs')
            }
            return param_names
        except (AttributeError, ValueError):
            # Fallback to empty set if inspection fails
            return set()
    
    async def _process_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_completion_tokens: int,
        response_format_dict: Optional[Dict[str, Any]],
        response_format_class: Optional[Any],
        api_params: Dict[str, Any],
        metadata_params: Dict[str, Any],
        api_caller,  # Callable to call the actual API
    ) -> Any:
        """
        Common request processing logic for both parse() and create().
        
        Args:
            messages: Chat messages
            model: Model name
            temperature: Temperature
            max_completion_tokens: Max completion tokens
            response_format_dict: Response format dict for storage (None for create)
            response_format_class: Response format class for parsing (None for create)
            api_params: API parameters to forward
            metadata_params: Metadata parameters (not forwarded to API)
            api_caller: Async callable that calls the actual API
            
        Returns:
            API response or mock response from cache
        """
        # Extract reasoning_effort for hashing if present
        reasoning_effort = api_params.get("reasoning_effort")
        
        # Compute request hash
        request_hash = self.cache_mgr.compute_request_hash(
            messages, model, temperature, max_completion_tokens,
            reasoning_effort=reasoning_effort,
            tools=api_params.get("tools"),
            tool_choice=api_params.get("tool_choice")
        )
        
        if self.mode == "realtime":
            # Realtime mode: always call API (ignore cache), but save response to cache
            print(f"[Realtime] Calling API for hash {request_hash[:8]}...")
            
            # Call actual API with concurrency limit
            if self.semaphore is not None:
                async with self.semaphore:
                    resp = await api_caller()
            else:
                resp = await api_caller()
            
            # Save to cache
            try:
                response_data = self._extract_response_data(resp)
                self.cache_mgr.save_response(request_hash, response_data)
                self.cache_mgr.save_request(
                    request_hash, messages, model, temperature, max_completion_tokens,
                    response_format=response_format_dict
                )
            except (OSError, KeyError, ValueError) as e:
                print(f"[Cache] Failed to save response: {e}")
            
            return resp
        
        elif self.mode == "cache_first":
            # Cache first mode: check cache, call API if miss
            cached_response = self.cache_mgr.get_cached_response(request_hash)
            if cached_response is not None:
                print(f"[Cache] Hit for hash {request_hash[:8]}...")
                return self._build_response_from_cache(cached_response, response_format_class)
            
            # Cache miss, call actual API
            print(f"[Cache] Miss for hash {request_hash[:8]}, calling API...")
            if self.semaphore is not None:
                async with self.semaphore:
                    resp = await api_caller()
            else:
                resp = await api_caller()
            
            # Save to cache
            try:
                response_data = self._extract_response_data(resp)
                self.cache_mgr.save_response(request_hash, response_data)
                self.cache_mgr.save_request(
                    request_hash, messages, model, temperature, max_completion_tokens,
                    response_format=response_format_dict
                )
            except (OSError, KeyError, ValueError) as e:
                print(f"[Cache] Failed to save response: {e}")
            
            return resp
        
        elif self.mode == "batch_write":
            # Batch write mode: check cache first, then queue
            cached_response = self.cache_mgr.get_cached_response(request_hash)
            if cached_response is not None:
                print(f"[Cache] Hit for hash {request_hash[:8]}...")
                return self._build_response_from_cache(cached_response, response_format_class)
            
            # Add to batch queue (merge API params and metadata params)
            # Ensure response_format is in all_params and not duplicated
            all_params = {**api_params, **metadata_params}
            if response_format_dict is not None:
                all_params["response_format"] = response_format_dict
            
            self.queue_mgr.add_request(
                request_hash, messages, model, temperature, max_completion_tokens,
                **all_params
            )
            print(f"[Batch] Queued request {request_hash[:8]}...")
            
            # Raise an error to signal that this request is queued
            raise RuntimeError(f"REQUEST_QUEUED: {request_hash}")
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    async def parse(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.2,
        max_completion_tokens: int = 1024,
        response_format: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Intercept chat.completions.parse() call.
        
        Routes based on mode:
        - realtime: Always call API (ignore cache), but save response to cache
        - cache_first: Check cache first, call API if miss, save response
        - batch_write: Check cache first, queue if miss
        """
        # Extract response_format info for hashing and storage
        response_format_dict = None
        response_format_class = None
        if response_format is not None:
            response_format_class = response_format
            try:
                # Convert Pydantic model to dict for storage
                response_format_dict = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": response_format.model_json_schema()
                    }
                }
            except (AttributeError, TypeError):
                response_format_dict = None
        
        # Separate API parameters from metadata parameters
        api_params = {}
        metadata_params = {}
        
        for key, value in kwargs.items():
            if key in self._api_param_names:
                api_params[key] = value
            else:
                metadata_params[key] = value
        
        # Create API caller
        async def api_caller():
            return await self._original.parse(
                messages=messages,
                model=model,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                response_format=response_format,
                **api_params
            )
        
        return await self._process_request(
            messages, model, temperature, max_completion_tokens,
            response_format_dict, response_format_class,
            api_params, metadata_params, api_caller
        )
    
    def _extract_response_data(self, resp: Any) -> Dict[str, Any]:
        """Extract response data for caching."""
        if not hasattr(resp, 'choices') or not resp.choices:
            return {}
        
        choice = resp.choices[0]
        message = choice.message
        
        # Serialize tool_calls properly
        tool_calls_data = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
        
        return {
            "id": resp.id,
            "model": resp.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": message.role,
                    "content": message.content,
                    "reasoning_content": message.reasoning_content if hasattr(message, "reasoning_content") else None,
                    "tool_calls": tool_calls_data,
                },
                "finish_reason": choice.finish_reason
            }],
        }
    
    def _build_response_from_cache(
        self, 
        cached_data: Dict[str, Any],
        response_format_class: Optional[Any] = None
    ) -> Any:
        """Build a mock response from cached data."""
        # Extract content from cached response
        response_body = cached_data.get("response", cached_data)
        choices = response_body.get("choices", [])
        
        if not choices:
            raise ValueError("No choices in cached response")
        
        message_data = choices[0].get("message", {})
        content = message_data.get("content", "")
        tool_calls = message_data.get("tool_calls")
        model = response_body.get("model", "cached")
        
        # Create mock response
        mock_resp = MockChatCompletion(content, model, tool_calls=tool_calls)
        
        # Parse content if response format is provided
        if response_format_class and content:
            try:
                parsed_data = json.loads(content)
                mock_resp.choices[0].message.parsed = response_format_class(**parsed_data)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"[Cache] Failed to parse cached content: {e}")
                mock_resp.choices[0].message.parsed = None
        
        return mock_resp
    
    async def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.2,
        max_completion_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Intercept chat.completions.create() call with caching support.
        
        Routes based on mode:
        - realtime: Always call API (ignore cache), but save response to cache
        - cache_first: Check cache first, call API if miss, save response
        - batch_write: Check cache first, queue if miss
        """
        # Separate API parameters from metadata parameters
        api_params = {}
        metadata_params = {}
        
        for key, value in kwargs.items():
            if key in self._api_param_names:
                api_params[key] = value
            else:
                metadata_params[key] = value
        
        # Create API caller
        async def api_caller():
            create_kwargs = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
            }
            if max_completion_tokens is not None:
                create_kwargs["max_completion_tokens"] = max_completion_tokens
            create_kwargs.update(api_params)
            return await self._original.create(**create_kwargs)
        
        return await self._process_request(
            messages, model, temperature, max_completion_tokens or 1024,
            None, None,  # No response_format for create
            api_params, metadata_params, api_caller
        )


class WrappedChat:
    """Wrapper for chat with completions attribute."""
    
    def __init__(
        self,
        original_chat: Any,
        mode: str,
        cache_mgr: CacheManager,
        queue_mgr: BatchQueueManager,
        model: str = "",
        semaphore: Optional[asyncio.Semaphore] = None
    ):
        self._original = original_chat
        self.completions = WrappedChatCompletions(
            original_chat.completions,
            mode,
            cache_mgr,
            queue_mgr,
            model,
            semaphore
        )


class WrappedAsyncOpenAI:
    """
    Wrapper for AsyncOpenAI that intercepts API calls.
    
    Modes:
        - realtime: Always call API (ignore cache), save responses
        - cache_first: Check cache first, call API on miss
        - batch_write: Check cache first, queue on miss
    
    Usage:
        client = create_wrapped_client(api_key=..., mode="cache_first")
    """
    
    def __init__(
        self,
        original_client: AsyncOpenAI,
        mode: str = "realtime",
        cache_dir: str = ".cache",
        task_name: Optional[str] = None,
        model: str = "",
        max_concurrency: int = 20
    ):
        """
        Initialize wrapper.
        
        Args:
            original_client: Original AsyncOpenAI client
            mode: Operation mode (realtime, batch_write, cache_first)
            cache_dir: Directory for cache storage
            task_name: Task name for batch queue organization
            model: Model name (for task info)
            max_concurrency: Maximum concurrent requests in realtime mode (default: 20)
        """
        self._original = original_client
        self.mode = mode
        self.model = model
        self.cache_mgr = CacheManager(cache_dir)
        self.queue_mgr = BatchQueueManager(cache_dir, task_name)
        
        # Create semaphore for concurrency control in realtime mode
        semaphore = asyncio.Semaphore(max_concurrency) if mode == "realtime" else None
        
        # Wrap chat
        self.chat = WrappedChat(
            original_client.chat,
            mode,
            self.cache_mgr,
            self.queue_mgr,
            model,
            semaphore
        )
        
        # Forward other attributes to original client
        self.files = original_client.files
        self.batches = original_client.batches
    
    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to original client."""
        return getattr(self._original, name)


def create_wrapped_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: Optional[str] = None,
    cache_dir: str = ".cache",
    task_name: Optional[str] = None,
    model: str = "",
    strategy: str = "",
    max_concurrency: int = 20,
    **kwargs
) -> WrappedAsyncOpenAI:
    """
    Create a wrapped AsyncOpenAI client with cache and batch support.
    
    Args:
        api_key: OpenAI API key
        base_url: API base URL
        mode: Operation mode:
              - realtime: Always call API, save to cache
              - cache_first: Check cache first, call API on miss (default)
              - batch_write: Check cache first, queue on miss
              If None, reads from MODE environment variable (default: cache_first)
        cache_dir: Directory for cache storage
        task_name: Task name for batch queue (e.g., "gpt-4o_20250104_143022")
                   If None and mode is batch_write, auto-generates from model + timestamp
        model: Model name (used for task naming and info)
        strategy: Generation strategy (saved in task info)
        max_concurrency: Maximum concurrent requests in realtime mode (default: 20)
        **kwargs: Additional arguments for AsyncOpenAI
        
    Returns:
        Wrapped client that behaves like AsyncOpenAI but with cache/batch support
        
    Example:
        >>> client = create_wrapped_client(
        ...     api_key="sk-...", 
        ...     mode="batch_write",
        ...     model="gpt-4o",
        ...     strategy="full_traj_v2"
        ... )
        >>> resp = await client.chat.completions.parse(...)
    """
    if mode is None:
        mode = os.environ.get("MODE", "cache_first")
    
    # Normalize mode
    if mode not in ("realtime", "batch_write", "cache_first"):
        # If mode is something else (like batch commands), default to realtime
        if mode.startswith("batch_"):
            mode = "realtime"  # Batch commands don't need wrapper
    
    # Auto-generate task name for batch_write mode
    if mode in ("batch_write", "cache_first") and task_name is None:
        # Use model name (sanitized) + timestamp
        model_safe = model.replace("/", "_").replace(" ", "_") if model else "default"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = f"{model_safe}_{timestamp}"
    
    # Create original client
    original_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    
    # Wrap it
    wrapped = WrappedAsyncOpenAI(
        original_client, mode, cache_dir, task_name, model, max_concurrency
    )
    
    # Create task info if in batch_write mode
    if mode == "batch_write" and task_name:
        wrapped.queue_mgr.create_task_info(model, strategy)
        print(f"[Task] Created batch task: {task_name}")
    
    return wrapped


def get_queue_stats(cache_dir: str = ".cache") -> Dict[str, Any]:
    """Get statistics about queued requests and cache."""
    queue_mgr = BatchQueueManager(cache_dir)
    cache_mgr = CacheManager(cache_dir)
    
    queued_requests = queue_mgr.get_queued_requests(sort_by_scenario=False)
    cache_stats = cache_mgr.get_cache_stats()
    
    return {
        "queued_requests": len(queued_requests),
        "cached_requests": cache_stats["requests"],
        "cached_responses": cache_stats["responses"],
        "cache_hit_rate": cache_stats["hit_rate"]
    }

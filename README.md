# OpenAI Batch Wrapper

A transparent wrapper for the OpenAI Python client that adds caching and batch API support. Drop-in replacement for `AsyncOpenAI` with zero code changes required.

## Features

- 🚀 **Drop-in Replacement**: Works exactly like `AsyncOpenAI` - no code changes needed
- 💾 **Automatic Caching**: Responses are cached to avoid duplicate API calls
- 📦 **Batch API Support**: Queue requests for OpenAI's 50% cheaper Batch API
- 🔄 **Three Operation Modes**: `realtime`, `batch_write`, `cache_first`
- 🛠️ **CLI Tools**: Complete workflow management via command line
- 🔌 **Extensible**: Provider adapter interface for future multi-provider support

## Installation

```bash
pip install openai-batch-wrapper
```

Or install from source:

```bash
git clone https://github.com/ultragtx/openai-batch-wrapper.git
cd openai-batch-wrapper
pip install -e .
```

## Quick Start

### Basic Usage (Realtime with Caching)

```python
from openai_batch_wrapper import create_wrapped_client

# Create client - drop-in replacement for AsyncOpenAI
client = create_wrapped_client(
    api_key="sk-...",
    model="gpt-4o"
)

# Use exactly like AsyncOpenAI
response = await client.chat.completions.parse(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-4o",
    response_format=MyResponseModel
)
```

### Batch Mode Workflow

For large workloads, use the Batch API to save 50% on costs:

```python
# Step 1: Queue requests (MODE=batch_write)
client = create_wrapped_client(
    api_key="sk-...",
    mode="batch_write",
    model="gpt-4o",
    strategy="my_pipeline"
)

# Run your pipeline - requests are queued instead of sent
for item in data:
    try:
        await process_item(client, item)
    except RuntimeError as e:
        if "REQUEST_QUEUED" in str(e):
            pass  # Expected - request was queued
```

Then use CLI tools to manage the batch:

```bash
# Step 2: Prepare batch files
batch-prepare my_task_name

# Step 3: Upload to OpenAI
batch-upload my_task_name

# Step 4: Wait for completion, then download
batch-download

# Step 5: Convert results to cache
batch-to-cache

# Step 6: Re-run with cached responses
MODE=cache_first python your_pipeline.py
```

## Operation Modes

### `realtime` (default)
- Calls API immediately
- Caches responses automatically
- Returns cached response if available

### `batch_write`
- Queues requests for batch processing
- Returns cached response if available
- Raises `RuntimeError("REQUEST_QUEUED: {hash}")` for new requests

### `cache_first`
- Only uses cached responses
- Queues cache misses for batch processing
- Raises `RuntimeError("CACHE_MISS: {hash}")` for uncached requests

## CLI Commands

| Command | Description |
|---------|-------------|
| `batch-prepare <task>` | Convert queued requests to batch files |
| `batch-upload <task>` | Upload batch files to OpenAI |
| `batch-download` | Download completed batch results |
| `batch-to-cache` | Convert batch results to cache |
| `batch-status` | Show pipeline status |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `realtime` | Operation mode |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `CACHE_DIR` | `.cache` | Cache directory |
| `BATCH_MAX_SIZE` | `100` | Max requests per batch |

## API Reference

### `create_wrapped_client()`

```python
def create_wrapped_client(
    api_key: str = None,
    base_url: str = None,
    mode: str = None,          # "realtime", "batch_write", "cache_first"
    cache_dir: str = ".cache",
    task_name: str = None,     # Auto-generated if None
    model: str = "",           # For task info
    strategy: str = "",        # For task info
    max_concurrency: int = 20, # Concurrent requests limit
    **kwargs                   # Passed to AsyncOpenAI
) -> WrappedAsyncOpenAI
```

### `CacheManager`

```python
from openai_batch_wrapper import CacheManager

cache = CacheManager(".cache")

# Compute request hash
hash = cache.compute_request_hash(messages, model, temp, max_tokens)

# Get/save cached response
response = cache.get_cached_response(hash)
cache.save_response(hash, response_data)

# Get stats
stats = cache.get_cache_stats()
# {"requests": 100, "responses": 95, "hit_rate": 0.95}
```

### `BatchQueueManager`

```python
from openai_batch_wrapper import BatchQueueManager

queue = BatchQueueManager(".cache", "my_task")

# Add request to queue
queue.add_request(hash, messages, model, temp, max_tokens)

# Get queued requests
requests = queue.get_queued_requests(sort_by_scenario=True)

# List all tasks
tasks = BatchQueueManager.list_tasks(".cache")
```

### `BatchManager`

```python
from openai_batch_wrapper import BatchManager

batch_mgr = BatchManager(".cache")

# Generate batch file
batch_mgr.generate_batch_file(task_name, batch_idx, requests, metadata)

# Submit to OpenAI
batch_id = await batch_mgr.submit_batch(client, batch_name)

# Check status
status = await batch_mgr.check_batch_status(client, batch_name)

# Download results
output_path = await batch_mgr.download_batch_results(client, batch_name)
```

## Cache Directory Structure

```
.cache/
├── requests/           # Request metadata (for debugging)
│   └── {hash}.json
├── responses/          # Cached API responses
│   └── {hash}.json
├── batch_queue/        # Queued requests by task
│   └── {task_name}/
│       ├── task_info.json
│       └── {hash}.json
└── batches/            # Batch files
    ├── pending/        # Ready to upload
    ├── submitted/      # Uploaded, waiting
    ├── completed/      # Downloaded
    └── processed/      # Converted to cache
```

## Example: Full Pipeline

```python
import asyncio
import os
from openai_batch_wrapper import create_wrapped_client
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

async def process_item(client, question: str):
    try:
        response = await client.chat.completions.parse(
            messages=[{"role": "user", "content": question}],
            model="gpt-4o",
            response_format=Response
        )
        return response.choices[0].message.parsed
    except RuntimeError as e:
        if "REQUEST_QUEUED" in str(e) or "CACHE_MISS" in str(e):
            return None
        raise

async def main():
    mode = os.environ.get("MODE", "realtime")
    
    client = create_wrapped_client(
        api_key=os.environ.get("OPENAI_API_KEY"),
        mode=mode,
        model="gpt-4o",
        strategy="qa_pipeline"
    )
    
    questions = ["What is 2+2?", "What is the capital of France?"]
    results = []
    
    for q in questions:
        result = await process_item(client, q)
        if result:
            results.append(result)
    
    print(f"Processed {len(results)}/{len(questions)} questions")
    
    if mode == "batch_write":
        print("\nNext steps:")
        print("  batch-prepare")
        print("  batch-upload")
        print("  # wait for completion")
        print("  batch-download")
        print("  batch-to-cache")
        print("  MODE=cache_first python your_script.py")

if __name__ == "__main__":
    asyncio.run(main())
```

## License

MIT License

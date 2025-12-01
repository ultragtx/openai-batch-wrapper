#!/usr/bin/env python
"""
Prepare batch files from queued requests.

Reads requests from a task directory in .cache/batch_queue/ and generates JSONL batch files.
Supports scenario-based grouping and size limits.

Usage:
    batch-prepare <task_name>
    batch-prepare gpt-4o_20250104_143022
    
    # Or list available tasks
    batch-prepare --list
    
Environment Variables:
    BATCH_MAX_SIZE: Maximum requests per batch (default: 100)
    PROVIDER: API provider (default: openai)
    CACHE_DIR: Cache directory (default: .cache)
"""

import os
import sys
import argparse
from typing import Dict, List, Any

from ..batch_manager import BatchManager
from ..batch_queue import BatchQueueManager
from ..provider_adapters import convert_batch_request

# Environment variables
BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "100"))
PROVIDER = os.environ.get("PROVIDER", "openai")
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")


def list_tasks():
    """List available tasks."""
    tasks = BatchQueueManager.list_tasks(CACHE_DIR)
    
    if not tasks:
        print(f"[Info] No tasks found in {CACHE_DIR}/batch_queue/")
        return
    
    print(f"\n[Tasks] Found {len(tasks)} task(s):\n")
    
    for task_name in tasks:
        summary = BatchQueueManager.get_task_summary(CACHE_DIR, task_name)
        if summary:
            print(f"  📦 {task_name}")
            print(f"     Model: {summary.get('model', 'N/A')}")
            print(f"     Strategy: {summary.get('strategy', 'N/A')}")
            print(f"     Queued: {summary.get('queued_requests', 0)} requests")
            print(f"     Created: {summary.get('created_at', 'N/A')}")
            print()
        else:
            print(f"  📦 {task_name}")
            queue_mgr = BatchQueueManager(CACHE_DIR, task_name)
            print(f"     Queued: {queue_mgr.count_queued()} requests")
            print()


def create_batch_requests(
    requests: List[Dict[str, Any]],
    provider: str = "openai"
) -> List[Dict[str, Any]]:
    """
    Convert queued requests to batch API format.
    
    Args:
        requests: List of queued requests
        provider: Target provider
        
    Returns:
        List of batch request objects
    """
    batch_requests = []
    
    for request in requests:
        # Build batch request
        batch_req = {
            "custom_id": request["hash"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": request["model"],
                "messages": request["messages"],
                "temperature": request.get("temperature", 0.2),
                "max_completion_tokens": request.get("max_completion_tokens", 1024)
            }
        }
        
        # Add response_format if present
        if "response_format" in request:
            batch_req["body"]["response_format"] = request["response_format"]
        
        # Convert to provider format if needed
        if provider != "openai":
            batch_req = convert_batch_request(batch_req, provider)
        
        batch_requests.append(batch_req)
    
    return batch_requests


def prepare_batches(task_name: str, batch_max_size: int = BATCH_MAX_SIZE, provider: str = PROVIDER):
    """Main function to prepare batch files."""
    print("\n" + "="*60)
    print("BATCH PREPARATION")
    print("="*60 + "\n")
    
    # Initialize managers
    queue_mgr = BatchQueueManager(CACHE_DIR, task_name)
    batch_mgr = BatchManager(CACHE_DIR)
    
    # Show task info
    task_info = queue_mgr.get_task_info()
    if task_info:
        print(f"[Task] Name: {task_name}")
        print(f"[Task] Model: {task_info.get('model', 'N/A')}")
        print(f"[Task] Strategy: {task_info.get('strategy', 'N/A')}")
        print(f"[Task] Created: {task_info.get('created_at', 'N/A')}")
        print()
    else:
        print(f"[Task] Name: {task_name} (no task info found)")
        print()
    
    # Get queued requests - already sorted by scenario and idx
    queued_requests = queue_mgr.get_queued_requests(sort_by_scenario=True)
    
    if not queued_requests:
        print("[Info] No queued requests to prepare")
        return
    
    print(f"[Queue] Found {len(queued_requests)} queued requests")
    
    # Analyze scenario distribution
    scenario_counts = {}
    for req in queued_requests:
        scenario = req.get("scenario", "unknown")
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    
    print(f"[Scenarios] {len(scenario_counts)} scenario(s) found:")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  - {scenario}: {count} requests")
    print()
    
    # Create batch files
    total_batches = 0
    hash_to_request = {req["hash"]: req for req in queued_requests}
    
    # Split into batches by size (requests are already sorted by scenario)
    num_batches = (len(queued_requests) + batch_max_size - 1) // batch_max_size
    print(f"[Split] Creating {num_batches} batch file(s) (max {batch_max_size} per batch)")
    print(f"[Cache] Requests sorted by scenario+idx to maximize input cache hits")
    print()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_max_size
        end_idx = min((batch_idx + 1) * batch_max_size, len(queued_requests))
        batch_requests_data = queued_requests[start_idx:end_idx]
        
        # Analyze scenario distribution in this batch
        batch_scenarios = {}
        for req in batch_requests_data:
            scenario = req.get("scenario", "unknown")
            batch_scenarios[scenario] = batch_scenarios.get(scenario, 0) + 1
        
        # Convert to batch format
        batch_requests = create_batch_requests(batch_requests_data, provider)
        
        # Extract hashes
        batch_hashes = [req["hash"] for req in batch_requests_data]
        
        # Create metadata
        metadata = {
            "hashes": batch_hashes,
            "hash_map": {h: {
                "hash": h,
                "model": hash_to_request[h]["model"],
                "created_at": hash_to_request[h].get("created_at", ""),
                "task_name": hash_to_request[h].get("task_name", task_name),
                "scenario": hash_to_request[h].get("scenario", "unknown"),
                "idx": hash_to_request[h].get("idx", 0),
            } for h in batch_hashes},
            "provider": provider,
            "task_name": task_name,
            "scenarios": batch_scenarios
        }
        
        # Generate batch file
        batch_mgr.generate_batch_file(task_name, batch_idx, batch_requests, metadata)
        total_batches += 1
        
        # Show scenario distribution in this batch
        scenario_info = ", ".join([f"{s}:{c}" for s, c in sorted(batch_scenarios.items())])
        print(f"  ✓ {task_name}_batch{batch_idx}: {len(batch_requests)} requests ({scenario_info})")
    
    print(f"\n[Summary] Generated {total_batches} batch file(s) in {batch_mgr.pending_dir}")
    print(f"[Queue] {queue_mgr.count_queued()} requests remain in queue (will be cleared after upload)")
    
    # Show what was created
    pending_batches = batch_mgr.get_pending_batches()
    print(f"\n[Ready] {len(pending_batches)} batch file(s) ready to upload:")
    for batch_name in sorted(pending_batches):
        metadata = batch_mgr.get_batch_metadata(batch_name)
        if metadata:
            print(f"  - {batch_name}: {metadata.get('num_requests', 0)} requests")
    
    print(f"\nNext: Run batch-upload {task_name} to upload to OpenAI")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare batch files from queued requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  batch-prepare my_task_name     Prepare batches for a specific task
  batch-prepare --list           List all available tasks
  batch-prepare                  Use most recent task
        """
    )
    parser.add_argument("task_name", nargs="?", help="Task name to prepare")
    parser.add_argument("-l", "--list", action="store_true", help="List available tasks")
    parser.add_argument("--max-size", type=int, default=BATCH_MAX_SIZE,
                        help=f"Maximum requests per batch (default: {BATCH_MAX_SIZE})")
    parser.add_argument("--provider", default=PROVIDER,
                        help=f"API provider (default: {PROVIDER})")
    
    args = parser.parse_args()
    
    if args.list:
        list_tasks()
        return
    
    if args.task_name:
        task_name = args.task_name
    else:
        # Try to find the most recent task
        tasks = BatchQueueManager.list_tasks(CACHE_DIR)
        if not tasks:
            print("[Error] No tasks found. Please run your pipeline with MODE=batch_write first.")
            print("\nOr specify a task name:")
            print("  batch-prepare <task_name>")
            sys.exit(1)
        
        task_name = tasks[-1]  # Most recent (alphabetically last)
        print(f"[Info] Using most recent task: {task_name}")
        print("       (Specify task name to use a different one)\n")
    
    prepare_batches(task_name, args.max_size, args.provider)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Show status of queued requests, batches, and cache.

Displays comprehensive status of the batch processing pipeline.

Usage:
    batch-status
    
Environment Variables:
    OPENAI_API_KEY: OpenAI API key (optional, for detailed batch status)
    OPENAI_BASE_URL: API base URL (default: https://api.openai.com/v1)
    CACHE_DIR: Cache directory (default: .cache)
"""

import os
import asyncio
import argparse

from openai import AsyncOpenAI

from ..batch_manager import BatchManager
from ..batch_queue import BatchQueueManager
from ..wrapper import get_queue_stats

# Environment variables
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")


async def show_status():
    """Main function to show status."""
    print("\n" + "="*60)
    print("BATCH PROCESSING STATUS")
    print("="*60 + "\n")
    
    # Queue and cache stats
    stats = get_queue_stats(CACHE_DIR)
    
    print("Queue & Cache Statistics:")
    print(f"  - Queued requests: {stats['queued_requests']}")
    print(f"  - Cached requests: {stats['cached_requests']}")
    print(f"  - Cached responses: {stats['cached_responses']}")
    print(f"  - Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    # List tasks
    tasks = BatchQueueManager.list_tasks(CACHE_DIR)
    if tasks:
        print(f"\nTasks ({len(tasks)}):")
        for task_name in tasks:
            summary = BatchQueueManager.get_task_summary(CACHE_DIR, task_name)
            if summary:
                print(f"  - {task_name}:")
                print(f"      Model: {summary.get('model', 'N/A')}")
                print(f"      Queued: {summary.get('queued_requests', 0)}")
    
    # Batch stats
    batch_mgr = BatchManager(CACHE_DIR)
    
    pending = batch_mgr.get_pending_batches()
    submitted = batch_mgr.get_submitted_batches()
    completed = batch_mgr.get_completed_batches()
    processed = batch_mgr.get_processed_batches()
    
    print(f"\nBatch Files:")
    print(f"  - Pending: {len(pending)}")
    print(f"  - Submitted: {len(submitted)}")
    print(f"  - Completed: {len(completed)}")
    print(f"  - Processed: {len(processed)}")
    
    if pending:
        print(f"\nPending Batches (ready to upload):")
        for batch_name in pending:
            metadata = batch_mgr.get_batch_metadata(batch_name)
            if metadata:
                print(f"  - {batch_name}: {metadata.get('num_requests', 0)} requests")
    
    if submitted and API_KEY:
        print(f"\nSubmitted Batches (checking status...):")
        client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
        
        for batch_name in submitted:
            status_info = await batch_mgr.check_batch_status(client, batch_name)
            if status_info:
                counts = status_info["request_counts"]
                status = status_info["status"]
                print(f"  - {batch_name}:")
                print(f"      Status: {status}")
                print(f"      Progress: {counts['completed']}/{counts['total']} completed, {counts['failed']} failed")
                
                # Show batch ID for reference
                metadata = batch_mgr.get_batch_metadata(batch_name)
                if metadata and "batch_id" in metadata:
                    print(f"      Batch ID: {metadata['batch_id']}")
            else:
                print(f"  - {batch_name}: Error getting status")
    elif submitted:
        print(f"\nSubmitted Batches ({len(submitted)} batches):")
        print("  (Set OPENAI_API_KEY to see detailed status)")
        for batch_name in submitted:
            metadata = batch_mgr.get_batch_metadata(batch_name)
            if metadata:
                batch_id = metadata.get("batch_id", "unknown")
                print(f"  - {batch_name} (ID: {batch_id})")
    
    if completed:
        print(f"\nCompleted Batches (ready to convert to cache):")
        for batch_name in completed:
            metadata = batch_mgr.get_batch_metadata(batch_name)
            if metadata:
                counts = metadata.get('request_counts', {})
                print(f"  - {batch_name}: {counts.get('completed', 0)}/{counts.get('total', 0)} requests")
    
    # Show next steps
    print("\n" + "-"*60)
    print("Workflow Guide:")
    print("-"*60)
    if stats['queued_requests'] > 0 and not pending:
        print("  → Run: batch-prepare <task_name>")
    if pending:
        print("  → Run: batch-upload <task_name>")
    if submitted:
        print("  → Run: batch-download (check/download results)")
    if completed:
        print("  → Run: batch-to-cache (convert to cache)")
    if stats['cached_responses'] > 0:
        print("  → Run your pipeline with MODE=cache_first")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Show status of batch processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  batch-status            Show full status
        """
    )
    parser.parse_args()
    
    asyncio.run(show_status())


if __name__ == "__main__":
    main()

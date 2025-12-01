#!/usr/bin/env python
"""
Upload prepared batch files to OpenAI.

Uploads JSONL batch files from .cache/batches/pending/ to OpenAI.
After successful upload, clears corresponding queued requests from the task directory.

Usage:
    batch-upload <task_name>
    batch-upload gpt-4o_20250104_143022
    
    # Or upload all pending batches (finds tasks automatically)
    batch-upload --all
    
Environment Variables:
    OPENAI_API_KEY: OpenAI API key
    OPENAI_BASE_URL: API base URL (default: https://api.openai.com/v1)
    CACHE_DIR: Cache directory (default: .cache)
"""

import os
import sys
import asyncio
import argparse

from openai import AsyncOpenAI

from ..batch_manager import BatchManager
from ..batch_queue import BatchQueueManager

# Environment variables
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")


async def upload_batches(task_name: str):
    """Main function to upload prepared batch files."""
    print("\n" + "="*60)
    print("BATCH UPLOAD TO OPENAI")
    print("="*60 + "\n")
    
    print(f"[Task] Uploading batches for task: {task_name}\n")
    
    # Initialize managers
    queue_mgr = BatchQueueManager(CACHE_DIR, task_name)
    batch_mgr = BatchManager(CACHE_DIR)
    
    # Get pending batch files for this task
    all_pending = batch_mgr.get_pending_batches()
    
    # Filter by task_name
    pending_batches = []
    for batch_name in all_pending:
        metadata = batch_mgr.get_batch_metadata(batch_name)
        if metadata and metadata.get("task_name") == task_name:
            pending_batches.append(batch_name)
    
    if not pending_batches:
        print(f"[Info] No pending batch files for task: {task_name}")
        print(f"\nTip: Run batch-prepare {task_name} first to generate batch files")
        return
    
    print(f"[Pending] Found {len(pending_batches)} batch file(s) ready to upload:")
    for batch_name in sorted(pending_batches):
        metadata = batch_mgr.get_batch_metadata(batch_name)
        if metadata:
            print(f"  - {batch_name}: {metadata.get('num_requests', 0)} requests")
    
    # Create API client
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    
    print(f"\n[Upload] Uploading {len(pending_batches)} batch file(s) to OpenAI...")
    
    submitted_count = 0
    failed_count = 0
    total_cleared = 0
    
    for batch_name in sorted(pending_batches):
        print(f"\n[Batch] Uploading: {batch_name}")
        
        # Get metadata before upload
        metadata = batch_mgr.get_batch_metadata(batch_name)
        batch_hashes = metadata.get("hashes", []) if metadata else []
        
        # Submit batch
        batch_id = await batch_mgr.submit_batch(client, batch_name)
        
        if batch_id:
            submitted_count += 1
            print(f"  ✓ Success! Batch ID: {batch_id}")
            
            # Clear queued requests that were successfully submitted
            if batch_hashes:
                cleared = 0
                for req_hash in batch_hashes:
                    if queue_mgr.clear_request(req_hash):
                        cleared += 1
                total_cleared += cleared
                print(f"  ✓ Cleared {cleared} request(s) from queue")
        else:
            failed_count += 1
            print(f"  ✗ Failed to submit {batch_name}")
        
        await asyncio.sleep(1)
    
    print(f"\n[Summary]")
    print(f"  - Submitted: {submitted_count}/{len(pending_batches)}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Cleared from queue: {total_cleared} request(s)")
    print(f"  - Remaining in queue: {queue_mgr.count_queued()}")
    
    if submitted_count > 0:
        print("\nNext steps:")
        print("  1. Wait for batches to complete (can take up to 24 hours)")
        print("  2. Run: batch-download")
        print("  3. Run: batch-to-cache")
    
    if failed_count > 0:
        print("\nNote: Failed batches remain in pending/ directory. Fix issues and re-run.")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Upload prepared batch files to OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  batch-upload my_task_name     Upload batches for a specific task
  batch-upload --all            Upload all pending batches
  batch-upload                  Auto-detect task from pending batches
        """
    )
    parser.add_argument("task_name", nargs="?", help="Task name to upload")
    parser.add_argument("-a", "--all", action="store_true", help="Upload all pending batches")
    
    args = parser.parse_args()
    
    if args.all:
        # Upload all pending batches
        batch_mgr = BatchManager(CACHE_DIR)
        pending_batches = batch_mgr.get_pending_batches()
        
        if not pending_batches:
            print("[Info] No pending batch files to upload")
            return
        
        # Group by task
        tasks_to_upload = set()
        for batch_name in pending_batches:
            metadata = batch_mgr.get_batch_metadata(batch_name)
            if metadata and "task_name" in metadata:
                tasks_to_upload.add(metadata["task_name"])
        
        if not tasks_to_upload:
            print("[Info] No tasks found in pending batches")
            return
        
        print(f"[Info] Found {len(tasks_to_upload)} task(s) with pending batches")
        for task_name in sorted(tasks_to_upload):
            asyncio.run(upload_batches(task_name))
            print()  # Separator between tasks
        return
    
    if args.task_name:
        task_name = args.task_name
    else:
        # Try to find the task from pending batches
        batch_mgr = BatchManager(CACHE_DIR)
        pending_batches = batch_mgr.get_pending_batches()
        
        if not pending_batches:
            print("[Error] No pending batch files found.")
            print("\nRun batch-prepare first:")
            print("  batch-prepare <task_name>")
            sys.exit(1)
        
        # Get task from first pending batch
        metadata = batch_mgr.get_batch_metadata(pending_batches[0])
        if metadata and "task_name" in metadata:
            task_name = metadata["task_name"]
            print(f"[Info] Using task from pending batches: {task_name}")
            print("       (Specify task name to upload a different one)\n")
        else:
            print("[Error] Cannot determine task name from pending batches.")
            print("\nPlease specify task name:")
            print("  batch-upload <task_name>")
            sys.exit(1)
    
    asyncio.run(upload_batches(task_name))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Download completed batch results from OpenAI.

Checks status of submitted batches and downloads completed ones.
Can be run multiple times safely.

Usage:
    batch-download
    
Environment Variables:
    OPENAI_API_KEY: OpenAI API key
    OPENAI_BASE_URL: API base URL (default: https://api.openai.com/v1)
    CACHE_DIR: Cache directory (default: .cache)
"""

import os
import asyncio
import argparse

from openai import AsyncOpenAI

from ..batch_manager import BatchManager

# Environment variables
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")


async def download_batches():
    """Main function to download batches."""
    print("\n" + "="*60)
    print("BATCH DOWNLOAD")
    print("="*60 + "\n")
    
    batch_mgr = BatchManager(CACHE_DIR)
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    
    submitted_batches = batch_mgr.get_submitted_batches()
    
    if not submitted_batches:
        print("[Info] No submitted batches to check")
        return
    
    print(f"[Batch] Found {len(submitted_batches)} submitted batches")
    
    completed_count = 0
    in_progress_count = 0
    failed_count = 0
    
    for batch_name in submitted_batches:
        print(f"\n[Batch] Checking: {batch_name}")
        
        # Check status
        status_info = await batch_mgr.check_batch_status(client, batch_name)
        
        if not status_info:
            print(f"[Error] Could not get status for {batch_name}")
            failed_count += 1
            continue
        
        status = status_info["status"]
        counts = status_info["request_counts"]
        print(f"[Status] {status} - {counts['completed']}/{counts['total']} completed, {counts['failed']} failed")
        
        if status == "completed":
            # Download results
            output_path = await batch_mgr.download_batch_results(client, batch_name)
            if output_path:
                completed_count += 1
                print(f"[Success] Downloaded to {output_path}")
            else:
                failed_count += 1
        elif status in ["failed", "expired", "cancelled"]:
            print(f"[Error] Batch {status}")
            failed_count += 1
        else:
            in_progress_count += 1
            print("[Info] Still in progress")
        
        await asyncio.sleep(0.5)
    
    print(f"\n[Summary] Completed: {completed_count}, In Progress: {in_progress_count}, Failed: {failed_count}")
    
    if completed_count > 0:
        print("\nNext: Run batch-to-cache to convert results to cache")
    if in_progress_count > 0:
        print(f"\nNote: {in_progress_count} batches still in progress. Run this script again later.")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Download completed batch results from OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  batch-download          Check and download all submitted batches
        """
    )
    parser.parse_args()
    
    asyncio.run(download_batches())


if __name__ == "__main__":
    main()

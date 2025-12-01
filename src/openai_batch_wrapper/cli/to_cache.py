#!/usr/bin/env python
"""
Convert batch results to response cache.

Reads completed batch output files and saves responses to cache by hash.
This allows the main pipeline to use cached responses on next run.

Usage:
    batch-to-cache
    
Environment Variables:
    PROVIDER: API provider for format conversion (default: openai)
    CACHE_DIR: Cache directory (default: .cache)
"""

import os
import json
import argparse

from ..batch_manager import BatchManager
from ..cache_manager import CacheManager
from ..provider_adapters import convert_batch_response

# Environment variables
PROVIDER = os.environ.get("PROVIDER", "openai")
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")


def convert_to_cache():
    """Main function to convert batch results to cache."""
    print("\n" + "="*60)
    print("BATCH TO CACHE CONVERSION")
    print("="*60 + "\n")
    
    batch_mgr = BatchManager(CACHE_DIR)
    cache_mgr = CacheManager(CACHE_DIR)
    
    completed_batches = batch_mgr.get_completed_batches()
    
    if not completed_batches:
        print("[Info] No completed batches to process")
        return
    
    print(f"[Batch] Found {len(completed_batches)} completed batches")
    
    total_responses = 0
    failed_responses = 0
    
    for batch_name in completed_batches:
        print(f"\n[Batch] Processing: {batch_name}")
        
        output_path = batch_mgr.completed_dir / f"{batch_name}_output.jsonl"
        metadata = batch_mgr.get_batch_metadata(batch_name)
        
        if not output_path.exists():
            print(f"[Error] Output file not found: {output_path}")
            continue
        
        if not metadata or "hash_map" not in metadata:
            print(f"[Error] Invalid metadata for {batch_name}")
            continue
        
        hash_map = metadata["hash_map"]
        provider = metadata.get("provider", "openai")
        
        # Read output JSONL
        batch_responses = 0
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id")  # This is the hash
                    
                    if not custom_id or custom_id not in hash_map:
                        print(f"[Warning] Unknown custom_id: {custom_id}")
                        continue
                    
                    # Convert provider format to OpenAI format if needed
                    if provider != "openai":
                        result = convert_batch_response(result, provider)
                    
                    # Extract response
                    response_data = result.get("response", {})
                    
                    if response_data.get("status_code") != 200:
                        error_msg = response_data.get("body", {}).get("error", "Unknown error")
                        print(f"[Warning] Request failed for {custom_id[:8]}...: {error_msg}")
                        failed_responses += 1
                        continue
                    
                    body = response_data.get("body", {})
                    
                    # Save to cache
                    cache_mgr.save_response(custom_id, body)
                    batch_responses += 1
                    total_responses += 1
                    
                except json.JSONDecodeError as e:
                    print(f"[Error] Failed to parse line: {e}")
                    failed_responses += 1
                    continue
                except (KeyError, ValueError, TypeError) as e:
                    print(f"[Error] Failed to process response: {e}")
                    failed_responses += 1
                    continue
        
        print(f"[Success] Saved {batch_responses} responses to cache")
        
        # Mark batch as processed
        batch_mgr.mark_batch_processed(batch_name)
    
    print(f"\n[Summary] Total responses saved: {total_responses}")
    print(f"[Summary] Failed responses: {failed_responses}")
    
    # Show cache stats
    stats = cache_mgr.get_cache_stats()
    print(f"\n[Cache] Current statistics:")
    print(f"  - Cached requests: {stats['requests']}")
    print(f"  - Cached responses: {stats['responses']}")
    print(f"  - Hit rate: {stats['hit_rate']:.1%}")
    
    print("\nNext: Run your pipeline with MODE=cache_first to use cached responses")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Convert batch results to response cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  batch-to-cache          Convert all completed batches to cache
        """
    )
    parser.parse_args()
    
    convert_to_cache()


if __name__ == "__main__":
    main()

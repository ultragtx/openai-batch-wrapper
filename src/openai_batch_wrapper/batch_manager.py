"""
Batch manager for OpenAI Batch API operations.

Manages batch file generation, submission, status checking, and result downloading.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from openai import AsyncOpenAI


class BatchManager:
    """Manages OpenAI Batch API operations."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize batch manager.
        
        Args:
            cache_dir: Base directory for batch storage
        """
        self.cache_dir = Path(cache_dir)
        self.batches_dir = self.cache_dir / "batches"
        
        # Create batch directories
        self.pending_dir = self.batches_dir / "pending"
        self.submitted_dir = self.batches_dir / "submitted"
        self.completed_dir = self.batches_dir / "completed"
        self.processed_dir = self.batches_dir / "processed"
        
        for dir_path in [self.pending_dir, self.submitted_dir, self.completed_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_batch_file(
        self,
        scenario: str,
        batch_idx: int,
        requests: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        """
        Generate a batch JSONL file and metadata file.
        
        Args:
            scenario: Scenario name (e.g., "ETH", "HOTEL") or task name
            batch_idx: Batch index number
            requests: List of batch request objects
            metadata: Metadata to save (scenario, hashes, etc.)
            
        Returns:
            (jsonl_path, metadata_path) tuple
        """
        batch_name = f"{scenario}_batch{batch_idx}"
        jsonl_path = self.pending_dir / f"{batch_name}.jsonl"
        metadata_path = self.pending_dir / f"{batch_name}_metadata.json"
        
        # Write JSONL file
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for request in requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        # Write metadata
        full_metadata = {
            "scenario": scenario,
            "batch_idx": batch_idx,
            "batch_name": batch_name,
            "num_requests": len(requests),
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            **metadata
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"[Batch] Generated {batch_name}: {len(requests)} requests")
        return jsonl_path, metadata_path
    
    async def submit_batch(
        self,
        client: AsyncOpenAI,
        batch_name: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h"
    ) -> Optional[str]:
        """
        Submit a batch file to OpenAI.
        
        Args:
            client: OpenAI async client
            batch_name: Name of the batch (without extension)
            endpoint: API endpoint to use
            completion_window: Completion window (e.g., "24h")
            
        Returns:
            Batch ID if successful, None otherwise
        """
        jsonl_path = self.pending_dir / f"{batch_name}.jsonl"
        metadata_path = self.pending_dir / f"{batch_name}_metadata.json"
        
        if not jsonl_path.exists() or not metadata_path.exists():
            print(f"[Batch] Batch files not found: {batch_name}")
            return None
        
        try:
            # Upload file
            print(f"[Batch] Uploading {batch_name}...")
            with open(jsonl_path, 'rb') as f:
                file_response = await client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            input_file_id = file_response.id
            print(f"[Batch] Uploaded file: {input_file_id}")
            
            # Create batch
            print("[Batch] Creating batch...")
            batch_response = await client.batches.create(
                input_file_id=input_file_id,
                endpoint=endpoint,
                completion_window=completion_window
            )
            
            batch_id = batch_response.id
            print(f"[Batch] Created batch: {batch_id}")
            
            # Update metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            metadata.update({
                "batch_id": batch_id,
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
                "submitted_at": datetime.now().isoformat(),
                "status": "submitted"
            })
            
            # Move to submitted directory
            new_jsonl_path = self.submitted_dir / f"{batch_name}.jsonl"
            new_metadata_path = self.submitted_dir / f"{batch_name}_metadata.json"
            
            jsonl_path.rename(new_jsonl_path)
            with open(new_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            metadata_path.unlink()  # Remove old metadata
            
            return batch_id
            
        except Exception as e:
            print(f"[Batch] Failed to submit {batch_name}: {e}")
            return None
    
    async def check_batch_status(
        self,
        client: AsyncOpenAI,
        batch_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check the status of a submitted batch.
        
        Args:
            client: OpenAI async client
            batch_name: Name of the batch
            
        Returns:
            Batch status information or None if failed
        """
        metadata_path = self.submitted_dir / f"{batch_name}_metadata.json"
        
        if not metadata_path.exists():
            print(f"[Batch] Metadata not found for: {batch_name}")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            batch_id = metadata.get("batch_id")
            if not batch_id:
                print(f"[Batch] No batch_id in metadata: {batch_name}")
                return None
            
            # Query batch status
            batch_info = await client.batches.retrieve(batch_id)
            
            status_info = {
                "batch_id": batch_id,
                "status": batch_info.status,
                "created_at": batch_info.created_at,
                "completed_at": batch_info.completed_at,
                "failed_at": batch_info.failed_at,
                "request_counts": {
                    "total": batch_info.request_counts.total,
                    "completed": batch_info.request_counts.completed,
                    "failed": batch_info.request_counts.failed
                }
            }
            
            if batch_info.output_file_id:
                status_info["output_file_id"] = batch_info.output_file_id
            
            if batch_info.error_file_id:
                status_info["error_file_id"] = batch_info.error_file_id
            
            return status_info
            
        except Exception as e:
            print(f"[Batch] Failed to check status for {batch_name}: {e}")
            return None
    
    async def download_batch_results(
        self,
        client: AsyncOpenAI,
        batch_name: str
    ) -> Optional[Path]:
        """
        Download results for a completed batch.
        
        Args:
            client: OpenAI async client
            batch_name: Name of the batch
            
        Returns:
            Path to downloaded output file or None if failed
        """
        metadata_path = self.submitted_dir / f"{batch_name}_metadata.json"
        jsonl_path = self.submitted_dir / f"{batch_name}.jsonl"
        
        if not metadata_path.exists():
            print(f"[Batch] Metadata not found for: {batch_name}")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            batch_id = metadata.get("batch_id")
            if not batch_id:
                print(f"[Batch] No batch_id in metadata: {batch_name}")
                return None
            
            # Get batch info
            batch_info = await client.batches.retrieve(batch_id)
            
            if batch_info.status != "completed":
                print(f"[Batch] Batch not completed: {batch_name} (status: {batch_info.status})")
                return None
            
            output_file_id = batch_info.output_file_id
            if not output_file_id:
                print(f"[Batch] No output file for: {batch_name}")
                return None
            
            # Download output file
            print(f"[Batch] Downloading results for {batch_name}...")
            file_content = await client.files.content(output_file_id)
            
            # Save to completed directory
            output_path = self.completed_dir / f"{batch_name}_output.jsonl"
            with open(output_path, 'wb') as f:
                f.write(file_content.content)
            
            # Update and move metadata
            metadata.update({
                "output_file_id": output_file_id,
                "completed_at": datetime.now().isoformat(),
                "status": "completed",
                "request_counts": {
                    "total": batch_info.request_counts.total,
                    "completed": batch_info.request_counts.completed,
                    "failed": batch_info.request_counts.failed
                }
            })
            
            new_metadata_path = self.completed_dir / f"{batch_name}_metadata.json"
            with open(new_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Move input file to completed
            if jsonl_path.exists():
                new_jsonl_path = self.completed_dir / f"{batch_name}.jsonl"
                jsonl_path.rename(new_jsonl_path)
            
            # Remove from submitted
            metadata_path.unlink()
            
            print(f"[Batch] Downloaded {batch_name}: {metadata['request_counts']['completed']}/{metadata['request_counts']['total']} completed")
            
            return output_path
            
        except Exception as e:
            print(f"[Batch] Failed to download results for {batch_name}: {e}")
            return None
    
    def get_pending_batches(self) -> List[str]:
        """Get list of pending batch names."""
        return [p.stem.replace('_metadata', '') 
                for p in self.pending_dir.glob("*_metadata.json")]
    
    def get_submitted_batches(self) -> List[str]:
        """Get list of submitted batch names."""
        return [p.stem.replace('_metadata', '') 
                for p in self.submitted_dir.glob("*_metadata.json")]
    
    def get_completed_batches(self) -> List[str]:
        """Get list of completed batch names."""
        return [p.stem.replace('_metadata', '') 
                for p in self.completed_dir.glob("*_metadata.json")]
    
    def get_processed_batches(self) -> List[str]:
        """Get list of processed batch names."""
        return [p.stem.replace('_metadata', '') 
                for p in self.processed_dir.glob("*_metadata.json")]
    
    def get_batch_metadata(self, batch_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a batch.
        
        Args:
            batch_name: Name of the batch
            
        Returns:
            Metadata dictionary or None if not found
        """
        # Check all directories
        for dir_path in [self.pending_dir, self.submitted_dir, self.completed_dir, self.processed_dir]:
            metadata_path = dir_path / f"{batch_name}_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[Batch] Failed to read metadata for {batch_name}: {e}")
        
        return None
    
    def mark_batch_processed(self, batch_name: str) -> bool:
        """
        Mark a batch as processed.
        
        Args:
            batch_name: Name of the batch
            
        Returns:
            True if successful, False otherwise
        """
        metadata_path = self.completed_dir / f"{batch_name}_metadata.json"
        
        if not metadata_path.exists():
            print(f"[Batch] Metadata not found in completed: {batch_name}")
            return False
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            metadata.update({
                "processed_at": datetime.now().isoformat(),
                "status": "processed"
            })
            
            # Move to processed directory
            new_metadata_path = self.processed_dir / f"{batch_name}_metadata.json"
            with open(new_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            metadata_path.unlink()
            
            return True
            
        except Exception as e:
            print(f"[Batch] Failed to mark batch as processed: {e}")
            return False
    
    def get_all_batch_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all batches."""
        return {
            "pending": len(self.get_pending_batches()),
            "submitted": len(self.get_submitted_batches()),
            "completed": len(self.get_completed_batches()),
            "processed": len(self.get_processed_batches()),
        }

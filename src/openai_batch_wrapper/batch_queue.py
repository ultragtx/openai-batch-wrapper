"""
Batch queue manager for organizing requests to be batched.

Manages task-based queuing with metadata tracking.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class BatchQueueManager:
    """Manages the queue of requests to be batched."""
    
    def __init__(self, cache_dir: str = ".cache", task_name: Optional[str] = None):
        """
        Initialize batch queue manager.
        
        Args:
            cache_dir: Base cache directory
            task_name: Task directory name (e.g., "gpt-4o_20250104_143022")
                      If None, uses default "default" directory
        """
        self.cache_dir = Path(cache_dir)
        self.queue_base_dir = self.cache_dir / "batch_queue"
        self.queue_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Use task-specific directory
        self.task_name = task_name or "default"
        self.queue_dir = self.queue_base_dir / self.task_name
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        # Task info file
        self.task_info_file = self.queue_dir / "task_info.json"
    
    def create_task_info(self, model: str, strategy: str = "", description: str = ""):
        """
        Create or update task info file.
        
        Args:
            model: Model name
            strategy: Generation strategy
            description: Task description
        """
        task_info = {
            "task_name": self.task_name,
            "model": model,
            "strategy": strategy,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "total_requests": 0
        }
        
        # Update if exists
        if self.task_info_file.exists():
            try:
                with open(self.task_info_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    task_info["created_at"] = existing.get("created_at", task_info["created_at"])
            except Exception:
                pass
        
        with open(self.task_info_file, 'w', encoding='utf-8') as f:
            json.dump(task_info, f, ensure_ascii=False, indent=2)
    
    def get_task_info(self) -> Optional[Dict[str, Any]]:
        """Get task info."""
        if self.task_info_file.exists():
            try:
                with open(self.task_info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def add_request(
        self,
        request_hash: str,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_completion_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Add a request to the batch queue.
        
        Args:
            request_hash: Hash of the request
            messages: Chat messages
            model: Model name
            temperature: Temperature parameter
            max_completion_tokens: Max tokens parameter
            response_format: Response format specification
            **kwargs: Additional parameters (scenario, idx, etc.)
        
        Returns:
            Path to the queued request file
        """
        request_data = {
            "hash": request_hash,
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "created_at": datetime.now().isoformat(),
            "task_name": self.task_name,
            **kwargs
        }
        
        if response_format:
            request_data["response_format"] = response_format
        
        # Save to queue
        queue_file = self.queue_dir / f"{request_hash}.json"
        with open(queue_file, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, ensure_ascii=False, indent=2)
        
        # Update task info
        task_info = self.get_task_info()
        if task_info:
            task_info["total_requests"] = self.count_queued()
            task_info["last_updated"] = datetime.now().isoformat()
            with open(self.task_info_file, 'w', encoding='utf-8') as f:
                json.dump(task_info, f, ensure_ascii=False, indent=2)
        
        return str(queue_file)
    
    def get_queued_requests(self, sort_by_scenario: bool = True) -> List[Dict[str, Any]]:
        """
        Get all queued requests.
        
        Args:
            sort_by_scenario: If True, sort requests by scenario and idx for better cache locality
            
        Returns:
            List of request dictionaries
        """
        requests = []
        for file_path in self.queue_dir.glob("*.json"):
            # Skip task_info.json
            if file_path.name == "task_info.json":
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    request = json.load(f)
                    requests.append(request)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[Queue] Failed to load {file_path}: {e}")
        
        if sort_by_scenario and requests:
            # Sort by scenario (primary), then by idx (secondary) for cache locality
            # Use created_at as fallback if scenario/idx not available
            def sort_key(req):
                scenario = req.get("scenario", "")
                idx = req.get("idx", 0)
                created_at = req.get("created_at", "")
                return (scenario, idx, created_at)
            
            requests.sort(key=sort_key)
            print(f"[Queue] Sorted {len(requests)} requests by scenario and idx for cache locality")
        
        return requests
    
    def clear_request(self, request_hash: str) -> bool:
        """Remove a request from the queue."""
        queue_file = self.queue_dir / f"{request_hash}.json"
        if queue_file.exists():
            queue_file.unlink()
            return True
        return False
    
    def count_queued(self) -> int:
        """Count queued requests (excluding task_info.json)."""
        all_files = list(self.queue_dir.glob("*.json"))
        # Exclude task_info.json
        return len([f for f in all_files if f.name != "task_info.json"])
    
    def clear_all(self) -> int:
        """Clear all queued requests. Returns count of cleared files."""
        count = 0
        for f in self.queue_dir.glob("*.json"):
            if f.name != "task_info.json":
                f.unlink()
                count += 1
        return count
    
    @staticmethod
    def list_tasks(cache_dir: str = ".cache") -> List[str]:
        """
        List all task directories.
        
        Returns:
            List of task names
        """
        queue_base_dir = Path(cache_dir) / "batch_queue"
        if not queue_base_dir.exists():
            return []
        
        tasks = []
        for item in queue_base_dir.iterdir():
            if item.is_dir():
                tasks.append(item.name)
        return sorted(tasks)
    
    @staticmethod
    def get_task_summary(cache_dir: str = ".cache", task_name: str = "") -> Optional[Dict[str, Any]]:
        """
        Get summary of a task.
        
        Args:
            cache_dir: Base cache directory
            task_name: Task directory name
            
        Returns:
            Task summary dictionary
        """
        task_dir = Path(cache_dir) / "batch_queue" / task_name
        task_info_file = task_dir / "task_info.json"
        
        if not task_info_file.exists():
            return None
        
        try:
            with open(task_info_file, 'r', encoding='utf-8') as f:
                task_info = json.load(f)
            
            # Count files
            queue_count = len([f for f in task_dir.glob("*.json") if f.name != "task_info.json"])
            task_info["queued_requests"] = queue_count
            
            return task_info
        except Exception:
            return None
    
    @staticmethod
    def delete_task(cache_dir: str = ".cache", task_name: str = "") -> bool:
        """
        Delete a task and all its queued requests.
        
        Args:
            cache_dir: Base cache directory
            task_name: Task directory name
            
        Returns:
            True if deleted, False otherwise
        """
        import shutil
        task_dir = Path(cache_dir) / "batch_queue" / task_name
        
        if not task_dir.exists():
            return False
        
        try:
            shutil.rmtree(task_dir)
            return True
        except Exception:
            return False

"""
Cache manager for API requests and responses.

Provides hash-based caching to avoid duplicate API calls and enable result reuse.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class CacheManager:
    """Manages caching of API requests and responses."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Base directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.requests_dir = self.cache_dir / "requests"
        self.responses_dir = self.cache_dir / "responses"
        
        # Create cache directories
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
    
    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize messages for hashing by replacing image data URLs with placeholders.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Normalized messages with image placeholders
        """
        normalized = []
        for msg in messages:
            normalized_msg = {"role": msg["role"]}
            
            if isinstance(msg["content"], str):
                normalized_msg["content"] = msg["content"]
            elif isinstance(msg["content"], list):
                normalized_content = []
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        # Replace image data with placeholder
                        normalized_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": "<IMAGE_PLACEHOLDER>",
                                "detail": item.get("image_url", {}).get("detail", None)
                            }
                        })
                    else:
                        normalized_content.append(item)
                normalized_msg["content"] = normalized_content
            else:
                normalized_msg["content"] = msg["content"]
            
            normalized.append(normalized_msg)
        
        return normalized
    
    def compute_request_hash(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_completion_tokens: int,
        **kwargs
    ) -> str:
        """
        Compute a hash for the request parameters.
        
        Args:
            messages: Chat messages
            model: Model name
            temperature: Temperature parameter
            max_completion_tokens: Max tokens parameter
            **kwargs: Additional parameters to include in hash
            
        Returns:
            SHA256 hash string
        """
        # Normalize messages (replace images with placeholders)
        normalized_messages = self._normalize_messages(messages)
        
        # Create hashable representation
        hash_data = {
            "messages": normalized_messages,
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            **kwargs
        }
        
        # Compute hash
        hash_str = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def get_cached_response(self, request_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a request hash.
        
        Args:
            request_hash: Hash of the request
            
        Returns:
            Cached response data or None if not found
        """
        response_file = self.responses_dir / f"{request_hash}.json"
        
        if not response_file.exists():
            return None
        
        try:
            with open(response_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Cache] Failed to load cached response {request_hash}: {e}")
            return None
    
    def save_request(
        self,
        request_hash: str,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_completion_tokens: int,
        **kwargs
    ) -> None:
        """
        Save request data to cache.
        
        Args:
            request_hash: Hash of the request
            messages: Chat messages
            model: Model name
            temperature: Temperature parameter
            max_completion_tokens: Max tokens parameter
            **kwargs: Additional request parameters
        """
        request_file = self.requests_dir / f"{request_hash}.json"
        
        request_data = {
            "hash": request_hash,
            "messages": self._normalize_messages(messages),
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        try:
            with open(request_file, 'w', encoding='utf-8') as f:
                json.dump(request_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Cache] Failed to save request {request_hash}: {e}")
    
    def save_response(
        self,
        request_hash: str,
        response_data: Dict[str, Any]
    ) -> None:
        """
        Save response data to cache.
        
        Args:
            request_hash: Hash of the request
            response_data: Response data to cache
        """
        response_file = self.responses_dir / f"{request_hash}.json"
        
        cached_data = {
            "hash": request_hash,
            "response": response_data,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Cache] Failed to save response {request_hash}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        num_requests = len(list(self.requests_dir.glob("*.json")))
        num_responses = len(list(self.responses_dir.glob("*.json")))
        
        return {
            "requests": num_requests,
            "responses": num_responses,
            "hit_rate": num_responses / num_requests if num_requests > 0 else 0
        }
    
    def clear_cache(self, responses_only: bool = False) -> Dict[str, int]:
        """
        Clear cache files.
        
        Args:
            responses_only: If True, only clear response cache
            
        Returns:
            Dictionary with counts of cleared files
        """
        cleared = {"requests": 0, "responses": 0}
        
        # Clear responses
        for f in self.responses_dir.glob("*.json"):
            f.unlink()
            cleared["responses"] += 1
        
        # Clear requests if requested
        if not responses_only:
            for f in self.requests_dir.glob("*.json"):
                f.unlink()
                cleared["requests"] += 1
        
        return cleared

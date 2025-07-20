"""Context-chain filtering for Enhanced GPTCache."""
import hashlib
import json
import threading
from collections import deque
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer

from gptcache.similarity_evaluation import SimilarityEvaluation
from ..utils.config import get_config
from ..utils.metrics import record_cache_request

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    query: str
    response: Optional[str] = None
    timestamp: float = 0.0
    embedding: Optional[np.ndarray] = None

@dataclass
class ConversationContext:
    """Manages conversation history and context embeddings."""
    conversation_id: str
    turns: deque = field(default_factory=lambda: deque())
    context_embedding: Optional[np.ndarray] = None
    last_updated: float = 0.0
    
    def add_turn(self, query: str, response: Optional[str] = None, timestamp: float = 0.0) -> None:
        """Add a new turn to the conversation."""
        turn = ConversationTurn(query=query, response=response, timestamp=timestamp)
        self.turns.append(turn)
        # Invalidate context embedding when new turn is added
        self.context_embedding = None
    
    def get_recent_context(self, window_size: int) -> List[ConversationTurn]:
        """Get the most recent conversation turns within the window."""
        return list(self.turns)[-window_size:] if len(self.turns) > 0 else []
    
    def to_context_string(self, window_size: int) -> str:
        """Convert recent context to a string representation."""
        recent_turns = self.get_recent_context(window_size)
        context_parts = []
        
        for i, turn in enumerate(recent_turns):
            context_parts.append(f"Turn {i+1}: {turn.query}")
            if turn.response:
                context_parts.append(f"Response {i+1}: {turn.response}")
        
        return " | ".join(context_parts)

class ContextTracker:
    """Tracks and manages multiple conversation contexts."""
    
    def __init__(self, max_conversations: int = 100):
        self.conversations: Dict[str, ConversationContext] = {}
        self.max_conversations = max_conversations
        self.lock = threading.Lock()
    
    def get_or_create_context(self, conversation_id: str) -> ConversationContext:
        """Get existing context or create a new one."""
        with self.lock:
            if conversation_id not in self.conversations:
                # Remove oldest conversation if we exceed the limit
                if len(self.conversations) >= self.max_conversations:
                    oldest_id = min(
                        self.conversations.keys(),
                        key=lambda k: self.conversations[k].last_updated
                    )
                    del self.conversations[oldest_id]
                
                self.conversations[conversation_id] = ConversationContext(
                    conversation_id=conversation_id
                )
            
            return self.conversations[conversation_id]
    
    def add_turn(
        self, 
        conversation_id: str, 
        query: str, 
        response: Optional[str] = None,
        timestamp: float = 0.0
    ) -> None:
        """Add a turn to the specified conversation."""
        context = self.get_or_create_context(conversation_id)
        context.add_turn(query, response, timestamp)
        context.last_updated = timestamp
    
    def get_context_string(self, conversation_id: str, window_size: int) -> str:
        """Get context string for the specified conversation."""
        with self.lock:
            if conversation_id in self.conversations:
                return self.conversations[conversation_id].to_context_string(window_size)
            return ""

class ContextAwareSimilarity(SimilarityEvaluation):
    """GPTCache similarity evaluation with context-aware filtering."""
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        context_window_size: Optional[int] = None,
        divergence_threshold: Optional[float] = None,
        base_similarity_func: Optional[SimilarityEvaluation] = None,
    ):
        """Initialize context-aware similarity evaluation.
        
        Args:
            embedding_model: Model name for context embeddings
            context_window_size: Number of recent turns to consider for context
            divergence_threshold: Minimum similarity for context match
            base_similarity_func: Base similarity function to wrap
        """
        config = get_config()
        
        self.embedding_model_name = embedding_model or config.context.embedding_model
        self.context_window_size = context_window_size or config.context.window_size
        self.divergence_threshold = divergence_threshold or config.context.divergence_threshold
        self.enabled = config.context.enabled
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Context tracking
        self.context_tracker = ContextTracker()
        
        # Base similarity function (fallback to cosine similarity if not provided)
        self.base_similarity_func = base_similarity_func
        
        # Cache for context embeddings to avoid recomputation
        self.context_embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_lock = threading.Lock()
    
    def evaluation(
        self, 
        src_dict: Dict[str, Any], 
        cache_dict: Dict[str, Any], 
        **kwargs
    ) -> float:
        """Evaluate similarity between source query and cached entry with context filtering.
        
        Args:
            src_dict: Source query information
            cache_dict: Cached entry information
            **kwargs: Additional arguments
            
        Returns:
            float: Similarity score (0.0 if context diverged, otherwise base similarity)
        """
        if not self.enabled:
            return self._base_similarity(src_dict, cache_dict, **kwargs)
        
        try:
            # Extract conversation context information
            src_conversation_id = src_dict.get('conversation_id', 'default')
            cache_conversation_id = cache_dict.get('conversation_id', 'default')
            
            # If no context information is available, fall back to base similarity
            if not src_conversation_id or not cache_conversation_id:
                return self._base_similarity(src_dict, cache_dict, **kwargs)
            
            # Compute context similarity
            context_similarity = self._compute_context_similarity(
                src_conversation_id, 
                cache_conversation_id,
                src_dict.get('query', ''),
                cache_dict.get('query', '')
            )
            
            # If context similarity is below threshold, reject the cache hit
            if context_similarity < self.divergence_threshold:
                # Record context filtering decision
                record_cache_request(
                    query=src_dict.get('query', ''),
                    latency_ms=0.0,  # Context filtering latency is negligible
                    cache_hit=False,
                    context_similarity=context_similarity
                )
                return 0.0
            
            # Context is similar enough, compute base similarity
            base_score = self._base_similarity(src_dict, cache_dict, **kwargs)
            
            # Record successful context match
            record_cache_request(
                query=src_dict.get('query', ''),
                latency_ms=0.0,
                cache_hit=base_score > get_config().cache.similarity_threshold,
                context_similarity=context_similarity
            )
            
            return base_score
            
        except Exception as e:
            # If context processing fails, fall back to base similarity
            print(f"Context similarity evaluation failed: {e}")
            return self._base_similarity(src_dict, cache_dict, **kwargs)
    
    def range(self) -> Tuple[float, float]:
        """Return the range of similarity scores (required by GPTCache interface).
        
        Returns:
            Tuple[float, float]: (minimum_score, maximum_score)
        """
        return (0.0, 1.0)
    
    def _base_similarity(
        self, 
        src_dict: Dict[str, Any], 
        cache_dict: Dict[str, Any], 
        **kwargs
    ) -> float:
        """Compute base similarity score."""
        if self.base_similarity_func:
            return self.base_similarity_func.evaluation(src_dict, cache_dict, **kwargs)
        
        # Default cosine similarity implementation
        src_embedding = src_dict.get('embedding')
        cache_embedding = cache_dict.get('embedding')
        
        if src_embedding is None or cache_embedding is None:
            return 0.0
        
        # Convert to numpy arrays if needed
        if not isinstance(src_embedding, np.ndarray):
            src_embedding = np.array(src_embedding)
        if not isinstance(cache_embedding, np.ndarray):
            cache_embedding = np.array(cache_embedding)
        
        # Compute cosine similarity
        dot_product = np.dot(src_embedding, cache_embedding)
        norm_src = np.linalg.norm(src_embedding)
        norm_cache = np.linalg.norm(cache_embedding)
        
        if norm_src == 0 or norm_cache == 0:
            return 0.0
        
        return dot_product / (norm_src * norm_cache)
    
    def _compute_context_similarity(
        self, 
        src_conversation_id: str, 
        cache_conversation_id: str,
        src_query: str,
        cache_query: str
    ) -> float:
        """Compute similarity between conversation contexts."""
        # If it's the same conversation, context is identical
        if src_conversation_id == cache_conversation_id:
            return 1.0
        
        # Get context strings
        src_context = self.context_tracker.get_context_string(
            src_conversation_id, self.context_window_size
        )
        cache_context = self.context_tracker.get_context_string(
            cache_conversation_id, self.context_window_size
        )
        
        # If either context is empty, use query-based similarity
        if not src_context or not cache_context:
            return self._compute_query_context_similarity(src_query, cache_query)
        
        # Compute embeddings for contexts
        src_context_embedding = self._get_context_embedding(src_context)
        cache_context_embedding = self._get_context_embedding(cache_context)
        
        # Compute cosine similarity between context embeddings
        return self._cosine_similarity(src_context_embedding, cache_context_embedding)
    
    def _compute_query_context_similarity(self, src_query: str, cache_query: str) -> float:
        """Compute context similarity based on queries when context history is unavailable."""
        if not src_query or not cache_query:
            return 0.0
        
        # Use embedding model to compute query similarity as context proxy
        src_embedding = self._get_context_embedding(src_query)
        cache_embedding = self._get_context_embedding(cache_query)
        
        return self._cosine_similarity(src_embedding, cache_embedding)
    
    def _get_context_embedding(self, context_text: str) -> np.ndarray:
        """Get embedding for context text with caching."""
        # Create cache key
        context_hash = hashlib.md5(context_text.encode()).hexdigest()
        
        with self.cache_lock:
            if context_hash in self.context_embedding_cache:
                return self.context_embedding_cache[context_hash]
        
        # Compute embedding
        embedding = self.embedding_model.encode(context_text)
        
        with self.cache_lock:
            # Simple cache eviction: keep only the most recent 1000 embeddings
            if len(self.context_embedding_cache) > 1000:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self.context_embedding_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.context_embedding_cache[key]
            
            self.context_embedding_cache[context_hash] = embedding
        
        return embedding
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def add_conversation_turn(
        self, 
        conversation_id: str, 
        query: str, 
        response: Optional[str] = None,
        timestamp: float = 0.0
    ) -> None:
        """Add a turn to the conversation context."""
        self.context_tracker.add_turn(conversation_id, query, response, timestamp)
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about context tracking."""
        with self.context_tracker.lock:
            total_conversations = len(self.context_tracker.conversations)
            total_turns = sum(
                len(conv.turns) 
                for conv in self.context_tracker.conversations.values()
            )
            avg_turns_per_conversation = (
                total_turns / total_conversations if total_conversations > 0 else 0
            )
            
        with self.cache_lock:
            cached_embeddings = len(self.context_embedding_cache)
        
        return {
            'total_conversations': total_conversations,
            'total_turns': total_turns,
            'avg_turns_per_conversation': avg_turns_per_conversation,
            'cached_context_embeddings': cached_embeddings,
            'context_window_size': self.context_window_size,
            'divergence_threshold': self.divergence_threshold,
            'enabled': self.enabled,
        }

# Convenience function for creating context-aware similarity evaluator
def create_context_aware_similarity(
    embedding_model: Optional[str] = None,
    **kwargs
) -> ContextAwareSimilarity:
    """Create a context-aware similarity evaluator with default configuration."""
    return ContextAwareSimilarity(
        embedding_model=embedding_model,
        **kwargs
    )

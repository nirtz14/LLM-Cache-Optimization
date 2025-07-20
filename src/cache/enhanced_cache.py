"""Enhanced GPTCache integration with MeanCache features."""
import time
from typing import Optional, Any, Dict, Callable
from sentence_transformers import SentenceTransformer

from gptcache import Cache
from gptcache.embedding import SBERT
from gptcache.manager import CacheBase, VectorBase
from gptcache.similarity_evaluation import SearchDistanceEvaluation

from ..core.context_similarity import ContextAwareSimilarity
from ..core.pca_wrapper import PCAEmbeddingWrapper
from ..core.tau_manager import TauManager
from ..utils.config import get_config
from ..utils.metrics import get_performance_tracker, BenchmarkTimer, record_cache_request

class EnhancedCache:
    """Enhanced GPTCache with context filtering, PCA compression, and τ-tuning."""
    
    def __init__(
        self,
        cache_base: Optional[CacheBase] = None,
        vector_base: Optional[VectorBase] = None,
        embedding_model: Optional[str] = None,
        enable_context: bool = True,
        enable_pca: bool = True,
        enable_tau: bool = True,
    ):
        """Initialize enhanced cache.
        
        Args:
            cache_base: GPTCache storage backend
            vector_base: GPTCache vector storage backend  
            embedding_model: Sentence transformer model name
            enable_context: Enable context-chain filtering
            enable_pca: Enable PCA embedding compression
            enable_tau: Enable federated τ-tuning
        """
        self.config = get_config()
        
        # Override config with parameters
        if not enable_context:
            self.config.context.enabled = False
        if not enable_pca:
            self.config.pca.enabled = False
        if not enable_tau:
            self.config.federated.enabled = False
        
        self.embedding_model_name = embedding_model or self.config.context.embedding_model
        
        # Initialize base embedding function
        self.base_embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Create PCA wrapper around embedding function
        self.embedding_func = self._create_embedding_function()
        
        # Initialize context-aware similarity
        self.context_similarity = self._create_context_similarity()
        
        # Initialize τ-manager
        self.tau_manager = TauManager() if self.config.federated.enabled else None
        
        # Initialize GPTCache with enhanced components
        self.cache = self._create_gptcache(cache_base, vector_base)
        
        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _create_embedding_function(self) -> Callable:
        """Create embedding function with optional PCA compression."""
        def base_embed(text: str):
            return self.base_embedding_model.encode(text)
        
        if self.config.pca.enabled:
            return PCAEmbeddingWrapper(
                base_embedding_func=base_embed,
                target_dimensions=self.config.pca.target_dimensions,
                model_path=self.config.pca.model_path,
            )
        else:
            return base_embed
    
    def _create_context_similarity(self) -> ContextAwareSimilarity:
        """Create context-aware similarity evaluator."""
        # Use SearchDistanceEvaluation as base similarity
        base_similarity = SearchDistanceEvaluation()
        
        return ContextAwareSimilarity(
            embedding_model=self.embedding_model_name,
            context_window_size=self.config.context.window_size,
            divergence_threshold=self.config.context.divergence_threshold,
            base_similarity_func=base_similarity,
        )
    
    def _create_gptcache(
        self, 
        cache_base: Optional[CacheBase] = None,
        vector_base: Optional[VectorBase] = None
    ) -> Cache:
        """Create GPTCache instance with enhanced components."""
        # Create embedding component using SBERT (compatible with sentence-transformers)
        embedding = SBERT(self.embedding_model_name)
        
        # Initialize cache with enhanced similarity
        cache = Cache()
        cache.init(
            embedding_func=embedding,
            data_manager=cache_base,
            similarity_evaluation=self.context_similarity,
        )
        
        return cache
    
    def query(
        self, 
        query: str, 
        conversation_id: str = "default",
        ground_truth_response: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query the enhanced cache.
        
        Args:
            query: Input query text
            conversation_id: Conversation context identifier
            ground_truth_response: Known correct response (for evaluation)
            **kwargs: Additional query parameters
            
        Returns:
            Dict containing query results and metadata
        """
        start_time = time.time()
        self.total_queries += 1
        
        # Add conversation turn to context tracker
        if self.config.context.enabled:
            self.context_similarity.add_conversation_turn(
                conversation_id=conversation_id,
                query=query,
                timestamp=start_time
            )
        
        # Prepare query data with context information
        query_data = {
            'query': query,
            'conversation_id': conversation_id,
            'timestamp': start_time,
            **kwargs
        }
        
        # Query the cache
        with BenchmarkTimer("cache_query") as timer:
            try:
                # Get embedding for similarity computation
                query_embedding = self.embedding_func(query)
                query_data['embedding'] = query_embedding
                
                # Query cache
                cache_result = self.cache.get(query)
                
                # Determine if we got a cache hit
                cache_hit = cache_result is not None and len(str(cache_result).strip()) > 0
                
                if cache_hit:
                    self.cache_hits += 1
                    response = cache_result
                    
                    # Get similarity score for τ-tuning (approximate)
                    similarity_score = self.config.cache.similarity_threshold + 0.1  # Above threshold
                else:
                    self.cache_misses += 1
                    response = None
                    similarity_score = self.config.cache.similarity_threshold - 0.1  # Below threshold
                
            except Exception as e:
                print(f"Cache query error: {e}")
                cache_hit = False
                response = None
                similarity_score = 0.0
        
        query_latency = timer.elapsed_ms
        
        # Update τ-manager with query result
        current_threshold = self.config.cache.similarity_threshold
        if self.tau_manager:
            # Determine ground truth hit (simple heuristic if not provided)
            ground_truth_hit = None
            if ground_truth_response:
                # If we have ground truth, check if cache response matches
                ground_truth_hit = (
                    cache_hit and 
                    response and 
                    str(response).strip().lower() == ground_truth_response.strip().lower()
                )
            
            current_threshold = self.tau_manager.evaluate_threshold(
                query=query,
                similarity_score=similarity_score,
                cache_hit=cache_hit,
                ground_truth_hit=ground_truth_hit
            )
        
        # Record performance metrics
        record_cache_request(
            query=query,
            latency_ms=query_latency,
            cache_hit=cache_hit,
            context_similarity=None,  # Will be filled by context similarity if enabled
            embedding_compression_ratio=None,  # Will be filled by PCA wrapper if enabled
            tau_threshold=current_threshold
        )
        
        # Prepare result
        result = {
            'query': query,
            'response': response,
            'cache_hit': cache_hit,
            'latency_ms': query_latency,
            'similarity_threshold': current_threshold,
            'conversation_id': conversation_id,
            'timestamp': start_time,
        }
        
        return result
    
    def set(
        self,
        query: str,
        response: str,
        conversation_id: str = "default",
        **kwargs
    ) -> None:
        """Store a query-response pair in the cache.
        
        Args:
            query: Input query text
            response: Response to cache
            conversation_id: Conversation context identifier
            **kwargs: Additional storage parameters
        """
        try:
            # Add to context tracker
            if self.config.context.enabled:
                self.context_similarity.add_conversation_turn(
                    conversation_id=conversation_id,
                    query=query,
                    response=response,
                    timestamp=time.time()
                )
            
            # Store in cache with context information
            cache_data = {
                'query': query,
                'response': response,
                'conversation_id': conversation_id,
                'timestamp': time.time(),
                **kwargs
            }
            
            # Add embedding for storage
            cache_data['embedding'] = self.embedding_func(query)
            
            # Store in GPTCache
            self.cache.set(query, response)
            
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'cache_statistics': {
                'total_queries': self.total_queries,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0,
            },
            'performance_metrics': get_performance_tracker().get_current_stats().__dict__,
        }
        
        # Add component-specific stats
        if self.config.context.enabled and self.context_similarity:
            stats['context_statistics'] = self.context_similarity.get_context_stats()
        
        if self.config.pca.enabled and hasattr(self.embedding_func, 'get_compression_stats'):
            stats['pca_statistics'] = self.embedding_func.get_compression_stats()
        
        if self.config.federated.enabled and self.tau_manager:
            stats['tau_statistics'] = self.tau_manager.get_tau_stats()
        
        return stats
    
    def force_pca_training(self) -> bool:
        """Force PCA model training if embeddings are available."""
        if (self.config.pca.enabled and 
            hasattr(self.embedding_func, 'force_train_pca')):
            return self.embedding_func.force_train_pca()
        return False
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Reset component metrics
        get_performance_tracker().reset_metrics()
        
        if self.tau_manager:
            self.tau_manager.reset_user_performance()
    
    def get_current_threshold(self) -> float:
        """Get current similarity threshold."""
        if self.tau_manager:
            return self.tau_manager.get_current_threshold()
        return self.config.cache.similarity_threshold
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for analysis."""
        return {
            'cache_stats': self.get_stats(),
            'raw_metrics': get_performance_tracker().export_metrics(),
            'config': {
                'context_enabled': self.config.context.enabled,
                'pca_enabled': self.config.pca.enabled,
                'tau_enabled': self.config.federated.enabled,
                'embedding_model': self.embedding_model_name,
            }
        }

def create_enhanced_cache(
    embedding_model: Optional[str] = None,
    cache_base: Optional[CacheBase] = None,
    vector_base: Optional[VectorBase] = None,
    **kwargs
) -> EnhancedCache:
    """Create an enhanced cache with default configuration.
    
    Args:
        embedding_model: Sentence transformer model name
        cache_base: GPTCache storage backend
        vector_base: GPTCache vector storage backend
        **kwargs: Additional configuration parameters
        
    Returns:
        EnhancedCache: Configured enhanced cache instance
    """
    return EnhancedCache(
        embedding_model=embedding_model,
        cache_base=cache_base,
        vector_base=vector_base,
        **kwargs
    )

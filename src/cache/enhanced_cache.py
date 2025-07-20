"""Enhanced GPTCache implementation using proper GPTCache Cache builder pattern."""
import time
from typing import Optional, Any, Dict, List
import numpy as np

from gptcache import Cache, Config
from gptcache.adapter.api import put, get
from gptcache.embedding import SBERT
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.processor.pre import get_prompt

from ..utils.config import get_config
from ..utils.metrics import get_performance_tracker, BenchmarkTimer, record_cache_request


class EnhancedSimilarityEvaluation(SimilarityEvaluation):
    """Enhanced similarity evaluation with context filtering and PCA support."""
    
    def __init__(
        self, 
        base_similarity: SimilarityEvaluation = None,
        enable_context: bool = True,
        enable_tau: bool = True,
        config=None
    ):
        """Initialize enhanced similarity evaluation."""
        self.base_similarity = base_similarity or SearchDistanceEvaluation()
        self.enable_context = enable_context
        self.enable_tau = enable_tau
        self.config = config or get_config()
        
        # Initialize context similarity if enabled
        self.context_similarity = None
        if self.enable_context:
            from ..core.context_similarity import ContextAwareSimilarity
            self.context_similarity = ContextAwareSimilarity(
                embedding_model=self.config.context.embedding_model,
                context_window_size=self.config.context.window_size,
                divergence_threshold=self.config.context.divergence_threshold,
                base_similarity_func=self.base_similarity,
            )
        
        # Initialize τ-manager if enabled
        self.tau_manager = None
        if self.enable_tau:
            from ..core.tau_manager import TauManager
            self.tau_manager = TauManager()
    
    def evaluation(self, src_dict: dict, cache_dict: dict, **kwargs) -> float:
        """Evaluate similarity with context filtering."""
        # Get base similarity score
        similarity_score = self.base_similarity.evaluation(src_dict, cache_dict, **kwargs)
        
        # Apply context filtering if enabled
        if self.enable_context and self.context_similarity:
            context_score = self.context_similarity.evaluation(src_dict, cache_dict)
            if context_score < self.config.context.divergence_threshold:
                return 0.0  # Reject due to context mismatch
        
        return similarity_score
    
    def range(self):
        """Return the range of similarity scores."""
        return self.base_similarity.range()


class EnhancedCache:
    """Enhanced GPTCache using proper Cache builder pattern."""
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        enable_context: bool = True,
        enable_pca: bool = True,
        enable_tau: bool = True,
    ):
        """Initialize enhanced cache."""
        self.config = get_config()
        self.embedding_model_name = embedding_model or self.config.context.embedding_model
        
        # Store feature flags
        self.enable_context = enable_context
        self.enable_pca = enable_pca
        self.enable_tau = enable_tau
        
        # Initialize embedding function (with PCA if enabled)
        self.embedding_func = self._create_embedding_function()
        
        # Initialize data manager
        self.data_manager = self._create_data_manager()
        
        # Initialize similarity evaluation
        self.similarity_evaluation = EnhancedSimilarityEvaluation(
            enable_context=self.enable_context,
            enable_tau=self.enable_tau,
            config=self.config
        )
        
        # Initialize GPTCache with proper builder pattern
        self.cache = Cache()
        
        # Get current threshold (with τ-tuning if enabled)
        current_threshold = self.config.cache.similarity_threshold
        if self.enable_tau and hasattr(self.similarity_evaluation, 'tau_manager'):
            current_threshold = self.similarity_evaluation.tau_manager.get_current_threshold()
        
        # Initialize cache with all components
        self.cache.init(
            embedding_func=self.embedding_func,
            data_manager=self.data_manager,
            similarity_evaluation=self.similarity_evaluation,
            config=Config(similarity_threshold=current_threshold)
        )
        
        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _create_embedding_function(self):
        """Create embedding function with optional PCA compression."""
        # Base SBERT embedding
        base_embedding = SBERT(self.embedding_model_name)
        
        if self.enable_pca:
            from ..core.pca_wrapper import PCAEmbeddingWrapper
            # Wrap with PCA compression
            pca_wrapper = PCAEmbeddingWrapper(
                base_embedding_func=base_embedding.to_embeddings,
                target_dimensions=self.config.pca.target_dimensions,
                model_path=self.config.pca.model_path
            )
            self.pca_wrapper = pca_wrapper
            return pca_wrapper
        else:
            self.pca_wrapper = None
            return base_embedding.to_embeddings
    
    def _create_data_manager(self):
        """Create data manager with vector store for similarity search."""
        if self.enable_pca:
            # Use PCA target dimensions for vector store
            dimension = self.config.pca.target_dimensions
        else:
            # Use SBERT default dimensions
            dimension = SBERT(self.embedding_model_name).dimension
        
        # Use SQLite + FAISS for storage
        cache_base = CacheBase("sqlite")
        vector_base = VectorBase("faiss", dimension=dimension)
        return get_data_manager(cache_base, vector_base)
    
    def query(
        self, 
        query: str, 
        conversation_id: str = "default",
        ground_truth_response: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query the enhanced cache."""
        start_time = time.time()
        self.total_queries += 1
        
        # Add conversation turn to context tracker if enabled
        if (self.enable_context and 
            hasattr(self.similarity_evaluation, 'context_similarity') and
            self.similarity_evaluation.context_similarity):
            self.similarity_evaluation.context_similarity.add_conversation_turn(
                conversation_id=conversation_id,
                query=query,
                timestamp=start_time
            )
        
        # Query the cache using GPTCache API
        with BenchmarkTimer("cache_query") as timer:
            try:
                # Use GPTCache's adapter API
                response = get(query, cache_obj=self.cache)
                
                cache_hit = response is not None
                similarity_score = 1.0 if cache_hit else 0.0  # GPTCache handles similarity internally
                
                # Update τ-manager with results if enabled
                if (self.enable_tau and 
                    hasattr(self.similarity_evaluation, 'tau_manager') and 
                    self.similarity_evaluation.tau_manager):
                    ground_truth_hit = None
                    if ground_truth_response:
                        ground_truth_hit = (
                            cache_hit and 
                            response and 
                            str(response).strip().lower() == ground_truth_response.strip().lower()
                        )
                    
                    self.similarity_evaluation.tau_manager.evaluate_threshold(
                        query=query,
                        similarity_score=similarity_score,
                        cache_hit=cache_hit,
                        ground_truth_hit=ground_truth_hit
                    )
                
                if cache_hit:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                
            except Exception as e:
                print(f"Cache query error: {e}")
                cache_hit = False
                response = None
                similarity_score = 0.0
        
        query_latency = timer.elapsed_ms
        
        # Get current threshold
        try:
            current_threshold = self.cache.config.similarity_threshold
        except (AttributeError, TypeError):
            current_threshold = self.config.cache.similarity_threshold
        
        # Record performance metrics
        compression_ratio = None
        if self.enable_pca and hasattr(self.pca_wrapper, 'get_compression_stats'):
            stats = self.pca_wrapper.get_compression_stats()
            compression_ratio = stats.get('compression_ratio')
        
        record_cache_request(
            query=query,
            latency_ms=query_latency,
            cache_hit=cache_hit,
            context_similarity=similarity_score if self.enable_context else None,
            embedding_compression_ratio=compression_ratio,
            tau_threshold=current_threshold
        )
        
        return {
            'query': query,
            'response': response,
            'cache_hit': cache_hit,
            'latency_ms': query_latency,
            'similarity_threshold': current_threshold,
            'conversation_id': conversation_id,
            'timestamp': start_time,
        }
    
    def set(
        self,
        query: str,
        response: str,
        conversation_id: str = "default",
        **kwargs
    ) -> None:
        """Store a query-response pair in the cache."""
        try:
            # Add to context tracker if enabled
            if (self.enable_context and 
                hasattr(self.similarity_evaluation, 'context_similarity') and
                self.similarity_evaluation.context_similarity):
                self.similarity_evaluation.context_similarity.add_conversation_turn(
                    conversation_id=conversation_id,
                    query=query,
                    response=response,
                    timestamp=time.time()
                )
            
            # Use GPTCache's adapter API
            put(query, response, cache_obj=self.cache)
            
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'cache_statistics': {
                'total_queries': self.total_queries,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0,
            },
            'performance_metrics': get_performance_tracker().get_current_stats().__dict__,
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        get_performance_tracker().reset_metrics()
    
    def get_current_threshold(self) -> float:
        """Get current similarity threshold."""
        try:
            return self.cache.config.similarity_threshold
        except (AttributeError, TypeError):
            return self.config.cache.similarity_threshold


def create_enhanced_cache(
    embedding_model: Optional[str] = None,
    **kwargs
) -> EnhancedCache:
    """Create an enhanced cache using proper GPTCache pattern."""
    return EnhancedCache(
        embedding_model=embedding_model,
        **kwargs
    )

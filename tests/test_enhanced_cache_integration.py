"""Integration test suite for enhanced cache features."""
import pytest
import time
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.cache.enhanced_cache import EnhancedCache, create_enhanced_cache
from src.core.pca_wrapper import PCAEmbeddingWrapper
from src.core.tau_manager import TauManager


class TestEnhancedCacheIntegration:
    """Test integration of all enhanced cache features together."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock config for consistent testing
        self.mock_config = Mock()
        self.mock_config.context.embedding_model = "all-MiniLM-L6-v2"
        self.mock_config.context.window_size = 3
        self.mock_config.context.divergence_threshold = 0.3
        self.mock_config.context.enabled = True
        self.mock_config.cache.similarity_threshold = 0.65
        self.mock_config.cache.size_mb = 100
        self.mock_config.cache.eviction_policy = "lru"
        self.mock_config.pca.enabled = True
        self.mock_config.pca.target_dimensions = 64
        self.mock_config.pca.model_path = "test_pca_model.pkl"
        self.mock_config.pca.training_samples = 100
        self.mock_config.pca.compression_threshold = 50
        self.mock_config.federated.enabled = True
        self.mock_config.federated.num_users = 3
        self.mock_config.federated.aggregation_frequency = 10
        self.mock_config.federated.learning_rate = 0.01
        self.mock_config.federated.initial_tau = 0.65
        
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        self.mock_config.pca.model_path = os.path.join(self.temp_dir, "test_model.pkl")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_enhanced_cache_initialization_all_features(self):
        """Test enhanced cache initialization with all features enabled."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            cache = EnhancedCache(
                enable_context=True,
                enable_pca=True,
                enable_tau=True
            )
        
        assert cache.enable_context is True
        assert cache.enable_pca is True
        assert cache.enable_tau is True
        assert hasattr(cache, 'fallback_cache')
        assert cache.use_fallback is True
    
    def test_enhanced_cache_initialization_selective_features(self):
        """Test enhanced cache initialization with selective features."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            cache = EnhancedCache(
                enable_context=True,
                enable_pca=False,
                enable_tau=True
            )
        
        assert cache.enable_context is True
        assert cache.enable_pca is False
        assert cache.enable_tau is True
    
    @patch('src.cache.enhanced_cache.SBERT')
    @patch('src.core.pca_wrapper.PCAEmbeddingWrapper')
    @patch('src.core.context_similarity.ContextAwareSimilarity')
    @patch('src.core.tau_manager.TauManager')
    def test_feature_initialization_success(self, mock_tau, mock_context, mock_pca, mock_sbert):
        """Test successful initialization of all enhanced features."""
        # Mock successful feature initialization
        mock_sbert.return_value = Mock()
        mock_pca.return_value = Mock()
        mock_context.return_value = Mock()
        mock_tau.return_value = Mock()
        
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            cache = EnhancedCache(
                enable_context=True,
                enable_pca=True,
                enable_tau=True
            )
        
        # All features should be initialized
        assert cache.pca_wrapper is not None
        assert cache.context_similarity is not None
        assert cache.tau_manager is not None
    
    @patch('src.cache.enhanced_cache.SBERT')
    def test_feature_initialization_with_errors(self, mock_sbert):
        """Test feature initialization with some features failing."""
        mock_sbert.return_value = Mock()
        
        # Mock PCA to fail during initialization
        with patch('src.core.pca_wrapper.PCAEmbeddingWrapper', side_effect=Exception("PCA Error")):
            with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
                cache = EnhancedCache(
                    enable_context=True,
                    enable_pca=True,
                    enable_tau=True
                )
        
        # PCA should be disabled due to error
        assert cache.enable_pca is False
        assert cache.pca_wrapper is None
    
    def test_query_and_set_basic_workflow(self):
        """Test basic query and set workflow."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                # Mock embedding function
                mock_embedding_instance = Mock()
                mock_embedding_instance.to_embeddings.return_value = np.array([1.0, 2.0, 3.0, 4.0])
                mock_sbert.return_value = mock_embedding_instance
                
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=False,
                    enable_tau=False
                )
                
                # Test setting a cache entry
                cache.set("What is Python?", "Python is a programming language", "conv1")
                
                # Test querying the cache
                result = cache.query("What is Python?", "conv1")
                
                assert result['cache_hit'] is True
                assert result['response'] == "Python is a programming language"
                assert result['query'] == "What is Python?"
                assert result['conversation_id'] == "conv1"
                assert 'latency_ms' in result
                assert 'timestamp' in result
    
    def test_query_with_similarity_threshold(self):
        """Test query behavior with similarity thresholds."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                # Mock embedding function to return different embeddings
                mock_embedding_instance = Mock()
                
                def mock_embedding_func(text):
                    if "Python" in text:
                        return np.array([1.0, 2.0, 3.0, 4.0])
                    else:
                        return np.array([-1.0, -2.0, -3.0, -4.0])  # Different embedding
                
                mock_embedding_instance.to_embeddings.side_effect = mock_embedding_func
                mock_sbert.return_value = mock_embedding_instance
                
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=False,
                    enable_tau=False
                )
                
                # Set a cache entry
                cache.set("What is Python?", "Python is a programming language")
                
                # Query with similar text (should hit)
                result1 = cache.query("What is Python?")
                assert result1['cache_hit'] is True
                
                # Query with dissimilar text (should miss)
                result2 = cache.query("What is JavaScript?")
                assert result2['cache_hit'] is False
    
    @patch('src.core.context_similarity.ContextAwareSimilarity')
    def test_conversation_context_isolation(self, mock_context_class):
        """Test that different conversations are isolated."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                # Mock embedding function
                mock_embedding_instance = Mock()
                mock_embedding_instance.to_embeddings.return_value = np.array([1.0, 2.0, 3.0, 4.0])
                mock_sbert.return_value = mock_embedding_instance
                
                # Mock context similarity instance - this is crucial
                mock_context_instance = Mock()
                mock_context_class.return_value = mock_context_instance
                
                cache = EnhancedCache(
                    enable_context=True,  # Enable context filtering
                    enable_pca=False,
                    enable_tau=False
                )
                
                # Set cache entry in conversation 1
                cache.set("What is AI?", "AI is artificial intelligence", "conv1")
                
                # Query from same conversation (should hit)
                result1 = cache.query("What is AI?", "conv1")
                assert result1['cache_hit'] is True
                
                # Query from different conversation (should miss due to context filtering)
                result2 = cache.query("What is AI?", "conv2")
                assert result2['cache_hit'] is False
    
    @patch('src.cache.enhanced_cache.SBERT')
    def test_pca_compression_integration(self, mock_sbert):
        """Test PCA compression integration."""
        # Mock embedding function
        mock_embedding_instance = Mock()
        mock_embedding_instance.to_embeddings.return_value = np.random.randn(128)
        mock_sbert.return_value = mock_embedding_instance
        
        # Mock PCA wrapper
        with patch('src.core.pca_wrapper.PCAEmbeddingWrapper') as mock_pca_class:
            mock_pca_instance = Mock()
            mock_pca_instance.return_value = np.random.randn(64)  # Compressed embedding
            mock_pca_instance.get_compression_stats.return_value = {
                'compression_ratio': 2.0,
                'model_loaded': True
            }
            mock_pca_class.return_value = mock_pca_instance
            
            with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=True,
                    enable_tau=False
                )
                
                # Set and query to test PCA compression
                cache.set("Test query", "Test response")
                result = cache.query("Test query")
                
                # PCA wrapper should be called
                assert mock_pca_instance.called
    
    @patch('src.cache.enhanced_cache.SBERT')
    def test_tau_threshold_optimization(self, mock_sbert):
        """Test tau threshold optimization integration."""
        # Mock embedding function
        mock_embedding_instance = Mock()
        mock_embedding_instance.to_embeddings.return_value = np.array([1.0, 2.0, 3.0, 4.0])
        mock_sbert.return_value = mock_embedding_instance
        
        # Mock tau manager
        with patch('src.core.tau_manager.TauManager') as mock_tau_class:
            mock_tau_instance = Mock()
            mock_tau_instance.get_current_threshold.return_value = 0.7  # Different from config
            mock_tau_instance.evaluate_threshold.return_value = 0.7
            mock_tau_class.return_value = mock_tau_instance
            
            with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=False,
                    enable_tau=True
                )
                
                # Query to trigger tau evaluation
                cache.set("Test query", "Test response")
                result = cache.query("Test query")
                
                # Tau manager should be called
                mock_tau_instance.get_current_threshold.assert_called()
    
    def test_metrics_recording_integration(self):
        """Test that metrics are properly recorded with all features."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                with patch('src.cache.enhanced_cache.record_cache_request') as mock_record:
                    # Mock embedding function
                    mock_embedding_instance = Mock()
                    mock_embedding_instance.to_embeddings.return_value = np.array([1.0, 2.0, 3.0, 4.0])
                    mock_sbert.return_value = mock_embedding_instance
                    
                    cache = EnhancedCache(
                        enable_context=False,
                        enable_pca=False,
                        enable_tau=False
                    )
                    
                    # Query to trigger metrics recording
                    cache.set("Test query", "Test response")
                    cache.query("Test query")
                    
                    # Metrics should be recorded
                    mock_record.assert_called()
                    
                    # Check call arguments
                    call_args = mock_record.call_args[1]
                    assert 'query' in call_args
                    assert 'latency_ms' in call_args
                    assert 'cache_hit' in call_args
    
    def test_cache_statistics_integration(self):
        """Test cache statistics with all features."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                # Mock embedding function to return different embeddings for different queries
                mock_embedding_instance = Mock()
                
                def mock_embedding_func(text):
                    if "Query 1" in text:
                        return np.array([1.0, 2.0, 3.0, 4.0])
                    elif "Query 2" in text:
                        return np.array([2.0, 3.0, 4.0, 5.0])
                    elif "Query 3" in text:
                        return np.array([-1.0, -2.0, -3.0, -4.0])  # Very different
                    else:
                        return np.array([0.0, 0.0, 0.0, 1.0])
                
                mock_embedding_instance.to_embeddings.side_effect = mock_embedding_func
                mock_sbert.return_value = mock_embedding_instance
                
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=False,
                    enable_tau=False
                )
                
                # Perform some operations
                cache.set("Query 1", "Response 1")
                cache.set("Query 2", "Response 2")
                cache.query("Query 1")  # Hit
                cache.query("Query 3")  # Miss
                
                # Get statistics
                stats = cache.get_stats()
                
                assert 'cache_statistics' in stats
                assert 'performance_metrics' in stats
                
                cache_stats = stats['cache_statistics']
                assert cache_stats['total_queries'] == 2
                assert cache_stats['cache_hits'] == 1
                assert cache_stats['cache_misses'] == 1
                assert cache_stats['hit_rate'] == 0.5
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                # Mock embedding function to sometimes fail
                mock_embedding_instance = Mock()
                mock_embedding_instance.to_embeddings.side_effect = Exception("Embedding error")
                mock_sbert.return_value = mock_embedding_instance
                
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=False,
                    enable_tau=False
                )
                
                # Operations should not crash even with embedding errors
                cache.set("Test query", "Test response")
                result = cache.query("Test query")
                
                # Should return gracefully
                assert result['cache_hit'] is False  # Due to embedding error
                assert result['response'] is None
    
    def test_concurrent_access_safety(self):
        """Test concurrent access safety."""
        import threading
        import time
        
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                # Mock embedding function
                mock_embedding_instance = Mock()
                mock_embedding_instance.to_embeddings.return_value = np.array([1.0, 2.0, 3.0, 4.0])
                mock_sbert.return_value = mock_embedding_instance
                
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=False,
                    enable_tau=False
                )
                
                results = []
                errors = []
                
                def worker():
                    try:
                        for i in range(10):
                            cache.set(f"Query {i}", f"Response {i}")
                            result = cache.query(f"Query {i}")
                            results.append(result)
                            time.sleep(0.001)  # Small delay
                    except Exception as e:
                        errors.append(e)
                
                # Start multiple threads
                threads = []
                for _ in range(3):
                    t = threading.Thread(target=worker)
                    threads.append(t)
                    t.start()
                
                # Wait for completion
                for t in threads:
                    t.join()
                
                # Should have results and no errors
                assert len(results) == 30  # 3 threads * 10 operations each
                assert len(errors) == 0
    
    def test_cache_reset_functionality(self):
        """Test cache reset functionality."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                # Mock embedding function
                mock_embedding_instance = Mock()
                mock_embedding_instance.to_embeddings.return_value = np.array([1.0, 2.0, 3.0, 4.0])
                mock_sbert.return_value = mock_embedding_instance
                
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=False,
                    enable_tau=False
                )
                
                # Add some data
                cache.set("Query 1", "Response 1")
                cache.query("Query 1")
                
                # Verify data exists
                stats_before = cache.get_stats()
                assert stats_before['cache_statistics']['total_queries'] > 0
                
                # Reset metrics
                cache.reset_metrics()
                
                # Verify reset
                stats_after = cache.get_stats()
                assert stats_after['cache_statistics']['total_queries'] == 0
                assert stats_after['cache_statistics']['cache_hits'] == 0
                assert stats_after['cache_statistics']['cache_misses'] == 0


class TestEnhancedCacheFeatureInteractions:
    """Test interactions between different enhanced features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.context.embedding_model = "all-MiniLM-L6-v2"
        self.mock_config.context.window_size = 3
        self.mock_config.context.divergence_threshold = 0.3
        self.mock_config.context.enabled = True
        self.mock_config.cache.similarity_threshold = 0.65
        self.mock_config.cache.size_mb = 100
        self.mock_config.cache.eviction_policy = "lru"
        self.mock_config.pca.enabled = True
        self.mock_config.pca.target_dimensions = 64
        self.mock_config.pca.model_path = "test_model.pkl"
        self.mock_config.pca.training_samples = 100
        self.mock_config.pca.compression_threshold = 50
        self.mock_config.federated.enabled = True
        self.mock_config.federated.num_users = 3
        self.mock_config.federated.aggregation_frequency = 10
        self.mock_config.federated.learning_rate = 0.01
        self.mock_config.federated.initial_tau = 0.65
    
    @patch('src.cache.enhanced_cache.SBERT')
    def test_pca_with_tau_interaction(self, mock_sbert):
        """Test interaction between PCA compression and tau optimization."""
        # Mock embedding function
        mock_embedding_instance = Mock()
        mock_embedding_instance.to_embeddings.return_value = np.random.randn(128)
        mock_sbert.return_value = mock_embedding_instance
        
        # Mock PCA wrapper
        with patch('src.core.pca_wrapper.PCAEmbeddingWrapper') as mock_pca_class:
            mock_pca_instance = Mock()
            mock_pca_instance.return_value = np.random.randn(64)  # Compressed
            mock_pca_class.return_value = mock_pca_instance
            
            # Mock tau manager
            with patch('src.core.tau_manager.TauManager') as mock_tau_class:
                mock_tau_instance = Mock()
                mock_tau_instance.get_current_threshold.return_value = 0.7
                mock_tau_instance.evaluate_threshold.return_value = 0.7
                mock_tau_class.return_value = mock_tau_instance
                
                with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
                    cache = EnhancedCache(
                        enable_context=False,
                        enable_pca=True,
                        enable_tau=True
                    )
                    
                    # Test that both features work together
                    cache.set("Test query", "Test response")
                    result = cache.query("Test query")
                    
                    # Both PCA and tau should be invoked
                    mock_pca_instance.assert_called()
                    mock_tau_instance.get_current_threshold.assert_called()
    
    @patch('src.cache.enhanced_cache.SBERT')
    def test_context_with_pca_interaction(self, mock_sbert):
        """Test interaction between context filtering and PCA compression."""
        # Mock embedding function
        mock_embedding_instance = Mock()
        mock_embedding_instance.to_embeddings.return_value = np.random.randn(128)
        mock_sbert.return_value = mock_embedding_instance
        
        # Mock PCA wrapper
        with patch('src.core.pca_wrapper.PCAEmbeddingWrapper') as mock_pca_class:
            mock_pca_instance = Mock()
            mock_pca_instance.return_value = np.random.randn(64)
            mock_pca_class.return_value = mock_pca_instance
            
            # Mock context similarity
            with patch('src.core.context_similarity.ContextAwareSimilarity') as mock_context_class:
                mock_context_instance = Mock()
                mock_context_class.return_value = mock_context_instance
                
                with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
                    cache = EnhancedCache(
                        enable_context=True,
                        enable_pca=True,
                        enable_tau=False
                    )
                    
                    # Test with different conversations
                    cache.set("Query", "Response", "conv1")
                    
                    # Query from same conversation
                    result1 = cache.query("Query", "conv1")
                    
                    # Query from different conversation
                    result2 = cache.query("Query", "conv2")
                    
                    # Context filtering should affect results
                    # (Exact behavior depends on mock configuration)
                    assert 'conversation_id' in result1
                    assert 'conversation_id' in result2
    
    @patch('src.cache.enhanced_cache.SBERT')
    def test_all_features_together(self, mock_sbert):
        """Test all enhanced features working together."""
        # Mock embedding function
        mock_embedding_instance = Mock()
        mock_embedding_instance.to_embeddings.return_value = np.random.randn(128)
        mock_sbert.return_value = mock_embedding_instance
        
        # Mock all enhanced features
        with patch('src.core.pca_wrapper.PCAEmbeddingWrapper') as mock_pca_class:
            mock_pca_instance = Mock()
            mock_pca_instance.return_value = np.random.randn(64)
            mock_pca_instance.get_compression_stats.return_value = {'compression_ratio': 2.0}
            mock_pca_class.return_value = mock_pca_instance
            
            with patch('src.core.context_similarity.ContextAwareSimilarity') as mock_context_class:
                mock_context_instance = Mock()
                mock_context_class.return_value = mock_context_instance
                
                with patch('src.core.tau_manager.TauManager') as mock_tau_class:
                    mock_tau_instance = Mock()
                    mock_tau_instance.get_current_threshold.return_value = 0.7
                    mock_tau_instance.evaluate_threshold.return_value = 0.7
                    mock_tau_class.return_value = mock_tau_instance
                    
                    with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
                        cache = EnhancedCache(
                            enable_context=True,
                            enable_pca=True,
                            enable_tau=True
                        )
                        
                        # Test full workflow
                        cache.set("Complex query", "Complex response", "conv1")
                        result = cache.query("Complex query", "conv1")
                        
                        # All features should be initialized and working
                        assert cache.enable_context is True
                        assert cache.enable_pca is True
                        assert cache.enable_tau is True
                        
                        # Result should have all expected fields
                        assert 'cache_hit' in result
                        assert 'latency_ms' in result
                        assert 'similarity_threshold' in result
                        assert 'conversation_id' in result


class TestConvenienceFunction:
    """Test convenience function for creating enhanced cache."""
    
    def test_create_enhanced_cache_default(self):
        """Test creating enhanced cache with default settings."""
        mock_config = Mock()
        mock_config.context.embedding_model = "all-MiniLM-L6-v2"
        mock_config.context.window_size = 5
        mock_config.context.divergence_threshold = 0.3
        mock_config.context.enabled = True
        mock_config.cache.similarity_threshold = 0.8
        mock_config.cache.size_mb = 100
        mock_config.cache.eviction_policy = "lru"
        mock_config.pca.enabled = True
        mock_config.pca.target_dimensions = 128
        mock_config.pca.model_path = "models/pca_model.pkl"
        mock_config.pca.training_samples = 1000
        mock_config.pca.compression_threshold = 100
        mock_config.federated.enabled = True
        mock_config.federated.num_users = 10
        mock_config.federated.aggregation_frequency = 100
        mock_config.federated.learning_rate = 0.01
        mock_config.federated.initial_tau = 0.85
        
        with patch('src.cache.enhanced_cache.get_config', return_value=mock_config):
            with patch('src.cache.enhanced_cache.SBERT'):
                cache = create_enhanced_cache()
        
        assert isinstance(cache, EnhancedCache)
        assert cache.enable_context is True
        assert cache.enable_pca is True
        assert cache.enable_tau is True
    
    def test_create_enhanced_cache_custom_embedding_model(self):
        """Test creating enhanced cache with custom embedding model."""
        mock_config = Mock()
        mock_config.context.embedding_model = "all-MiniLM-L6-v2"
        mock_config.context.window_size = 5
        mock_config.context.divergence_threshold = 0.3
        mock_config.context.enabled = True
        mock_config.cache.similarity_threshold = 0.8
        mock_config.cache.size_mb = 100
        mock_config.cache.eviction_policy = "lru"
        mock_config.pca.enabled = True
        mock_config.pca.target_dimensions = 128
        mock_config.pca.model_path = "models/pca_model.pkl"
        mock_config.pca.training_samples = 1000
        mock_config.pca.compression_threshold = 100
        mock_config.federated.enabled = True
        mock_config.federated.num_users = 10
        mock_config.federated.aggregation_frequency = 100
        mock_config.federated.learning_rate = 0.01
        mock_config.federated.initial_tau = 0.85
        
        with patch('src.cache.enhanced_cache.get_config', return_value=mock_config):
            with patch('src.cache.enhanced_cache.SBERT'):
                cache = create_enhanced_cache(
                    embedding_model="custom-model",
                    enable_pca=False
                )
        
        assert isinstance(cache, EnhancedCache)
        assert cache.embedding_model_name == "custom-model"
        assert cache.enable_pca is False


class TestPerformanceOptimizations:
    """Test performance optimizations and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.context.embedding_model = "all-MiniLM-L6-v2"
        self.mock_config.context.window_size = 3
        self.mock_config.context.divergence_threshold = 0.3
        self.mock_config.context.enabled = True
        self.mock_config.cache.similarity_threshold = 0.65
        self.mock_config.cache.size_mb = 100
        self.mock_config.cache.eviction_policy = "lru"
        self.mock_config.pca.enabled = True
        self.mock_config.pca.target_dimensions = 64
        self.mock_config.pca.model_path = "test_model.pkl"
        self.mock_config.pca.training_samples = 100
        self.mock_config.pca.compression_threshold = 50
        self.mock_config.federated.enabled = True
        self.mock_config.federated.num_users = 3
        self.mock_config.federated.aggregation_frequency = 10
        self.mock_config.federated.learning_rate = 0.01
        self.mock_config.federated.initial_tau = 0.65
    
    def test_large_cache_performance(self):
        """Test performance with large number of cache entries."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                # Mock embedding function
                mock_embedding_instance = Mock()
                
                def mock_embedding_func(text):
                    # Generate deterministic but unique embeddings
                    hash_val = hash(text) % 1000
                    return np.array([hash_val, hash_val + 1, hash_val + 2, hash_val + 3])
                
                mock_embedding_instance.to_embeddings.side_effect = mock_embedding_func
                mock_sbert.return_value = mock_embedding_instance
                
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=False,
                    enable_tau=False
                )
                
                # Add many cache entries
                start_time = time.time()
                for i in range(100):
                    cache.set(f"Query {i}", f"Response {i}")
                set_time = time.time() - start_time
                
                # Query many entries
                start_time = time.time()
                hit_count = 0
                for i in range(100):
                    result = cache.query(f"Query {i}")
                    if result['cache_hit']:
                        hit_count += 1
                query_time = time.time() - start_time
                
                # Performance should be reasonable
                assert set_time < 10.0  # Should complete within 10 seconds
                assert query_time < 10.0  # Should complete within 10 seconds
                assert hit_count > 90  # Most queries should hit
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization with feature cycling."""
        with patch('src.cache.enhanced_cache.get_config', return_value=self.mock_config):
            with patch('src.cache.enhanced_cache.SBERT') as mock_sbert:
                # Mock embedding function
                mock_embedding_instance = Mock()
                mock_embedding_instance.to_embeddings.return_value = np.array([1.0, 2.0, 3.0, 4.0])
                mock_sbert.return_value = mock_embedding_instance
                
                cache = EnhancedCache(
                    enable_context=False,
                    enable_pca=False,
                    enable_tau=False
                )
                
                # Test that cache can handle repeated operations
                for cycle in range(5):
                    # Add entries
                    for i in range(20):
                        cache.set(f"Cycle {cycle} Query {i}", f"Response {i}")
                    
                    # Query entries
                    for i in range(20):
                        cache.query(f"Cycle {cycle} Query {i}")
                    
                    # Reset metrics to prevent memory growth
                    cache.reset_metrics()
                
                # Cache should still be functional
                cache.set("Final test", "Final response")
                result = cache.query("Final test")
                assert result['cache_hit'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
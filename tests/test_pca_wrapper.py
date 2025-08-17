"""Comprehensive test suite for PCA wrapper functionality."""
import pytest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.core.pca_wrapper import (
    PCAModel,
    PCATrainer,
    EmbeddingCompressor,
    PCAEmbeddingWrapper,
    create_pca_embedding_wrapper
)


class TestPCAModel:
    """Test PCAModel dataclass."""
    
    def test_pca_model_creation(self):
        """Test creating a PCAModel instance."""
        mock_pca = Mock()
        mock_scaler = Mock()
        
        model = PCAModel(
            pca=mock_pca,
            scaler=mock_scaler,
            original_dim=512,
            compressed_dim=128,
            explained_variance_ratio=0.95,
            training_samples=1000
        )
        
        assert model.pca == mock_pca
        assert model.scaler == mock_scaler
        assert model.original_dim == 512
        assert model.compressed_dim == 128
        assert model.explained_variance_ratio == 0.95
        assert model.training_samples == 1000
    
    def test_pca_model_defaults(self):
        """Test PCAModel with default values."""
        mock_pca = Mock()
        
        model = PCAModel(pca=mock_pca)
        
        assert model.pca == mock_pca
        assert model.scaler is None
        assert model.original_dim == 0
        assert model.compressed_dim == 0
        assert model.explained_variance_ratio == 0.0
        assert model.training_samples == 0


class TestPCATrainer:
    """Test PCATrainer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = PCATrainer(target_dimensions=64, use_scaling=True)
        # Create sample embeddings for testing
        np.random.seed(42)
        self.sample_embeddings = np.random.randn(100, 128)
    
    def test_trainer_initialization(self):
        """Test PCATrainer initialization."""
        trainer = PCATrainer(target_dimensions=128, use_scaling=False)
        
        assert trainer.target_dimensions == 128
        assert trainer.use_scaling is False
        assert trainer.lock is not None
    
    def test_train_pca_model_success(self):
        """Test successful PCA model training."""
        model = self.trainer.train_pca_model(self.sample_embeddings)
        
        assert isinstance(model, PCAModel)
        assert model.pca is not None
        assert model.scaler is not None  # scaling enabled
        assert model.original_dim == 128
        assert model.compressed_dim <= 64  # target dimensions
        assert 0 < model.explained_variance_ratio <= 1.0
        assert model.training_samples == 100
    
    def test_train_pca_model_without_scaling(self):
        """Test PCA training without scaling."""
        trainer = PCATrainer(target_dimensions=64, use_scaling=False)
        model = trainer.train_pca_model(self.sample_embeddings)
        
        assert model.scaler is None
        assert model.pca is not None
    
    def test_train_pca_model_insufficient_samples(self):
        """Test PCA training with insufficient samples."""
        small_embeddings = np.random.randn(32, 128)  # Less than target_dimensions
        
        with pytest.raises(ValueError, match="Need at least .* samples to train PCA"):
            self.trainer.train_pca_model(small_embeddings)
    
    def test_train_pca_model_explained_variance_adjustment(self):
        """Test PCA training with explained variance threshold adjustment."""
        # Create data that would need more components for good variance
        structured_data = np.random.randn(100, 128)
        
        model = self.trainer.train_pca_model(
            structured_data, 
            explained_variance_threshold=0.99
        )
        
        assert isinstance(model, PCAModel)
        assert model.explained_variance_ratio > 0.0
    
    def test_save_and_load_model(self):
        """Test saving and loading PCA models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            
            # Train and save model
            original_model = self.trainer.train_pca_model(self.sample_embeddings)
            self.trainer.save_model(original_model, model_path)
            
            # Load model
            loaded_model = self.trainer.load_model(model_path)
            
            assert loaded_model is not None
            assert loaded_model.original_dim == original_model.original_dim
            assert loaded_model.compressed_dim == original_model.compressed_dim
            assert loaded_model.training_samples == original_model.training_samples
            assert abs(loaded_model.explained_variance_ratio - original_model.explained_variance_ratio) < 1e-6
    
    def test_load_nonexistent_model(self):
        """Test loading a non-existent model."""
        result = self.trainer.load_model("nonexistent_path.pkl")
        assert result is None
    
    @patch('joblib.load')
    def test_load_model_error_handling(self, mock_load):
        """Test error handling in model loading."""
        mock_load.side_effect = Exception("Load error")
        
        with patch('os.path.exists', return_value=True):
            result = self.trainer.load_model("dummy_path.pkl")
            assert result is None


class TestEmbeddingCompressor:
    """Test EmbeddingCompressor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock PCA model
        self.mock_pca = Mock()
        self.mock_scaler = Mock()
        
        # Set up mock behaviors
        self.mock_pca.transform.return_value = np.random.randn(1, 64)
        self.mock_pca.inverse_transform.return_value = np.random.randn(1, 128)
        self.mock_scaler.transform.return_value = np.random.randn(1, 128)
        self.mock_scaler.inverse_transform.return_value = np.random.randn(1, 128)
        
        self.pca_model = PCAModel(
            pca=self.mock_pca,
            scaler=self.mock_scaler,
            original_dim=128,
            compressed_dim=64,
            explained_variance_ratio=0.95,
            training_samples=100
        )
        
        self.compressor = EmbeddingCompressor(self.pca_model)
    
    def test_compressor_initialization(self):
        """Test EmbeddingCompressor initialization."""
        assert self.compressor.pca_model == self.pca_model
        assert self.compressor.lock is not None
    
    def test_compress_single_embedding(self):
        """Test compressing a single embedding."""
        embedding = np.random.randn(128)
        
        # Set up expected behavior
        self.mock_pca.transform.return_value = np.array([[1, 2, 3]])
        
        result = self.compressor.compress(embedding)
        
        # Should call scaler.transform and pca.transform
        self.mock_scaler.transform.assert_called_once()
        self.mock_pca.transform.assert_called_once()
        
        # Result should be 1D for single input
        assert result.ndim == 1
        np.testing.assert_array_equal(result, [1, 2, 3])
    
    def test_compress_multiple_embeddings(self):
        """Test compressing multiple embeddings."""
        embeddings = np.random.randn(5, 128)
        
        # Set up expected behavior
        self.mock_pca.transform.return_value = np.random.randn(5, 64)
        
        result = self.compressor.compress(embeddings)
        
        # Should maintain batch dimension
        assert result.ndim == 2
        assert result.shape[0] == 5
    
    def test_compress_without_scaler(self):
        """Test compression without scaler."""
        pca_model_no_scaler = PCAModel(
            pca=self.mock_pca,
            scaler=None,
            original_dim=128,
            compressed_dim=64
        )
        compressor = EmbeddingCompressor(pca_model_no_scaler)
        
        embedding = np.random.randn(128)
        compressor.compress(embedding)
        
        # Scaler should not be called
        self.mock_scaler.transform.assert_not_called()
        self.mock_pca.transform.assert_called_once()
    
    def test_decompress_single_embedding(self):
        """Test decompressing a single embedding."""
        compressed = np.random.randn(64)
        
        # Set up expected behavior
        self.mock_pca.inverse_transform.return_value = np.array([[1, 2, 3, 4]])
        
        result = self.compressor.decompress(compressed)
        
        # Should call pca.inverse_transform and scaler.inverse_transform
        self.mock_pca.inverse_transform.assert_called_once()
        self.mock_scaler.inverse_transform.assert_called_once()
        
        # Result should be 1D for single input
        assert result.ndim == 1
    
    def test_decompress_multiple_embeddings(self):
        """Test decompressing multiple embeddings."""
        compressed = np.random.randn(5, 64)
        
        result = self.compressor.decompress(compressed)
        
        # Should maintain batch dimension
        assert result.ndim == 2
        assert result.shape[0] == 5
    
    def test_get_compression_ratio(self):
        """Test getting compression ratio."""
        ratio = self.compressor.get_compression_ratio()
        expected_ratio = 128 / 64  # original_dim / compressed_dim
        assert ratio == expected_ratio
    
    def test_get_explained_variance(self):
        """Test getting explained variance."""
        variance = self.compressor.get_explained_variance()
        assert variance == 0.95


class TestPCAEmbeddingWrapper:
    """Test PCAEmbeddingWrapper functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_func = Mock()
        self.mock_embedding_func.return_value = np.random.randn(128)
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.pca.target_dimensions = 64
        self.mock_config.pca.model_path = "test_model.pkl"
        self.mock_config.pca.training_samples = 100
        self.mock_config.pca.compression_threshold = 50
        self.mock_config.pca.enabled = True
        
        with patch('src.core.pca_wrapper.get_config', return_value=self.mock_config):
            self.wrapper = PCAEmbeddingWrapper(
                base_embedding_func=self.mock_embedding_func,
                auto_train=False  # Disable auto-training for controlled testing
            )
    
    def test_wrapper_initialization(self):
        """Test PCAEmbeddingWrapper initialization."""
        assert self.wrapper.base_embedding_func == self.mock_embedding_func
        assert self.wrapper.target_dimensions == 64
        assert self.wrapper.model_path == "test_model.pkl"
        assert self.wrapper.training_threshold == 100
        assert self.wrapper.enabled is True
        assert self.wrapper.trainer is not None
        assert self.wrapper.compressor is None  # No model loaded initially
        assert len(self.wrapper.collected_embeddings) == 0
    
    @patch('src.core.pca_wrapper.BenchmarkTimer')
    def test_call_without_pca(self, mock_timer):
        """Test calling wrapper without PCA compression."""
        mock_timer_instance = Mock()
        mock_timer_instance.elapsed_ms = 10.0
        mock_timer.return_value.__enter__ = Mock(return_value=mock_timer_instance)
        mock_timer.return_value.__exit__ = Mock(return_value=None)
        
        input_data = "test query"
        result = self.wrapper(input_data)
        
        # Should call base embedding function
        self.mock_embedding_func.assert_called_once_with(input_data)
        
        # Should collect embedding for future training
        assert len(self.wrapper.collected_embeddings) == 1
        
        # Should return original embedding
        assert isinstance(result, np.ndarray)
    
    def test_call_with_pca_compression(self):
        """Test calling wrapper with PCA compression enabled."""
        # Set up a mock compressor
        mock_compressor = Mock()
        mock_compressor.compress.return_value = np.random.randn(64)
        mock_compressor.get_compression_ratio.return_value = 2.0
        self.wrapper.compressor = mock_compressor
        
        with patch('src.core.pca_wrapper.BenchmarkTimer') as mock_timer:
            mock_timer_instance = Mock()
            mock_timer_instance.elapsed_ms = 5.0
            mock_timer.return_value.__enter__ = Mock(return_value=mock_timer_instance)
            mock_timer.return_value.__exit__ = Mock(return_value=None)
            
            input_data = "test query"
            result = self.wrapper(input_data)
        
        # Should call base embedding function
        self.mock_embedding_func.assert_called_once_with(input_data)
        
        # Should call compressor
        mock_compressor.compress.assert_called_once()
        
        # Should return compressed embedding
        assert isinstance(result, np.ndarray)
        assert len(result) == 64  # compressed size
    
    def test_force_train_pca_success(self):
        """Test forcing PCA training."""
        # Provide sufficient embeddings
        embeddings = np.random.randn(100, 128)
        
        with patch.object(self.wrapper.trainer, 'train_pca_model') as mock_train:
            mock_model = Mock()
            mock_train.return_value = mock_model
            
            with patch.object(self.wrapper.trainer, 'save_model') as mock_save:
                result = self.wrapper.force_train_pca(embeddings)
        
        assert result is True
        mock_train.assert_called_once_with(embeddings)
        mock_save.assert_called_once_with(mock_model, self.wrapper.model_path)
        assert self.wrapper.compressor is not None
    
    def test_force_train_pca_insufficient_data(self):
        """Test forcing PCA training with insufficient data."""
        # No embeddings provided and no collected embeddings
        result = self.wrapper.force_train_pca()
        assert result is False
    
    def test_force_train_pca_error_handling(self):
        """Test error handling in force PCA training."""
        embeddings = np.random.randn(100, 128)
        
        with patch.object(self.wrapper.trainer, 'train_pca_model') as mock_train:
            mock_train.side_effect = Exception("Training error")
            
            result = self.wrapper.force_train_pca(embeddings)
        
        assert result is False
    
    def test_get_compression_stats_without_compressor(self):
        """Test getting compression stats without compressor."""
        stats = self.wrapper.get_compression_stats()
        
        assert stats['enabled'] is True
        assert stats['model_loaded'] is False
        assert stats['collected_embeddings'] == 0
        assert stats['training_threshold'] == 100
        assert stats['total_compressions'] == 0
        assert stats['avg_compression_time_ms'] == 0.0
    
    def test_get_compression_stats_with_compressor(self):
        """Test getting compression stats with compressor."""
        # Set up mock compressor
        mock_compressor = Mock()
        mock_compressor.get_compression_ratio.return_value = 2.0
        mock_compressor.get_explained_variance.return_value = 0.95
        mock_compressor.pca_model.original_dim = 128
        mock_compressor.pca_model.compressed_dim = 64
        mock_compressor.pca_model.training_samples = 100
        
        self.wrapper.compressor = mock_compressor
        self.wrapper.total_compressions = 10
        self.wrapper.total_compression_time_ms = 50.0
        
        stats = self.wrapper.get_compression_stats()
        
        assert stats['model_loaded'] is True
        assert stats['compression_ratio'] == 2.0
        assert stats['explained_variance'] == 0.95
        assert stats['original_dimensions'] == 128
        assert stats['compressed_dimensions'] == 64
        assert stats['training_samples'] == 100
        assert stats['total_compressions'] == 10
        assert stats['avg_compression_time_ms'] == 5.0
    
    def test_decompress_embedding_without_compressor(self):
        """Test decompressing embedding without compressor."""
        embedding = np.random.randn(64)
        result = self.wrapper.decompress_embedding(embedding)
        
        # Should return as-is
        np.testing.assert_array_equal(result, embedding)
    
    def test_decompress_embedding_with_compressor(self):
        """Test decompressing embedding with compressor."""
        mock_compressor = Mock()
        decompressed = np.random.randn(128)
        mock_compressor.decompress.return_value = decompressed
        
        self.wrapper.compressor = mock_compressor
        
        compressed = np.random.randn(64)
        result = self.wrapper.decompress_embedding(compressed)
        
        mock_compressor.decompress.assert_called_once_with(compressed)
        np.testing.assert_array_equal(result, decompressed)
    
    def test_auto_training_trigger(self):
        """Test automatic PCA training when threshold is reached."""
        wrapper = PCAEmbeddingWrapper(
            base_embedding_func=self.mock_embedding_func,
            auto_train=True,
            training_samples_threshold=3
        )
        
        with patch.object(wrapper, '_train_pca_model') as mock_train:
            # Call wrapper multiple times to reach threshold
            for i in range(4):
                wrapper("test query")
            
            # Training should be triggered on the 3rd call
            mock_train.assert_called_once()


class TestConvenienceFunction:
    """Test convenience function for creating PCA wrapper."""
    
    def test_create_pca_embedding_wrapper(self):
        """Test creating PCA wrapper using convenience function."""
        mock_func = Mock()
        
        with patch('src.core.pca_wrapper.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.pca.target_dimensions = 64
            mock_config.pca.model_path = "test.pkl"
            mock_config.pca.training_samples = 100
            mock_config.pca.compression_threshold = 50
            mock_config.pca.enabled = True
            mock_get_config.return_value = mock_config
            
            wrapper = create_pca_embedding_wrapper(
                base_embedding_func=mock_func,
                target_dimensions=128
            )
        
        assert isinstance(wrapper, PCAEmbeddingWrapper)
        assert wrapper.base_embedding_func == mock_func
        assert wrapper.target_dimensions == 128  # Override should work


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def test_full_pca_workflow(self):
        """Test complete PCA workflow from training to compression."""
        # Create real embedding function for integration test
        def mock_embedding_func(text):
            np.random.seed(hash(text) % 1000)  # Deterministic but varied
            return np.random.randn(128)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "integration_test_model.pkl")
            
            with patch('src.core.pca_wrapper.get_config') as mock_get_config:
                mock_config = Mock()
                mock_config.pca.target_dimensions = 64
                mock_config.pca.model_path = model_path
                mock_config.pca.training_samples = 10  # Small for test
                mock_config.pca.compression_threshold = 5
                mock_config.pca.enabled = True
                mock_get_config.return_value = mock_config
                
                wrapper = PCAEmbeddingWrapper(
                    base_embedding_func=mock_embedding_func,
                    auto_train=False
                )
                
                # Generate training data
                training_embeddings = []
                for i in range(20):
                    emb = mock_embedding_func(f"training query {i}")
                    training_embeddings.append(emb)
                
                training_data = np.vstack(training_embeddings)
                
                # Force train PCA
                success = wrapper.force_train_pca(training_data)
                assert success is True
                
                # Test compression
                test_embedding = mock_embedding_func("test query")
                compressed = wrapper(test_embedding)
                
                # Verify compression worked
                assert len(compressed) < len(test_embedding)
                assert wrapper.compressor is not None
                
                # Test decompression
                decompressed = wrapper.decompress_embedding(compressed)
                assert len(decompressed) == len(test_embedding)
                
                # Verify model was saved
                assert os.path.exists(model_path)
    
    def test_pca_with_different_input_types(self):
        """Test PCA wrapper with different input types."""
        def embedding_func(text):
            return [1.0, 2.0, 3.0, 4.0]  # Return list instead of numpy array
        
        with patch('src.core.pca_wrapper.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.pca.target_dimensions = 2
            mock_config.pca.model_path = "test.pkl"
            mock_config.pca.training_samples = 100
            mock_config.pca.compression_threshold = 50
            mock_config.pca.enabled = True
            mock_get_config.return_value = mock_config
            
            wrapper = PCAEmbeddingWrapper(
                base_embedding_func=embedding_func
            )
            
            result = wrapper("test")
            
            # Should convert to numpy array internally
            assert isinstance(result, np.ndarray)
            assert len(result) == 4  # No compression yet
    
    def test_threading_safety(self):
        """Test thread safety of PCA components."""
        import threading
        import time
        
        def embedding_func(text):
            return np.random.randn(10)
        
        with patch('src.core.pca_wrapper.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.pca.target_dimensions = 5
            mock_config.pca.model_path = "test.pkl"
            mock_config.pca.training_samples = 100
            mock_config.pca.compression_threshold = 50
            mock_config.pca.enabled = True
            mock_get_config.return_value = mock_config
            
            wrapper = PCAEmbeddingWrapper(
                base_embedding_func=embedding_func
            )
            
            results = []
            errors = []
            
            def worker():
                try:
                    for i in range(10):
                        result = wrapper(f"query {i}")
                        results.append(result)
                        time.sleep(0.01)  # Small delay
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
            assert len(results) == 30  # 3 threads * 10 calls each
            assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
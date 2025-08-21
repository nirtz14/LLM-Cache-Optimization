"""PCA embedding compression for Enhanced GPTCache."""
import os
import joblib
import threading
from typing import Callable, Optional, Any, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.config import get_config
from ..utils.metrics import record_cache_request, BenchmarkTimer

@dataclass
class PCAModel:
    """Container for PCA model and related components."""
    pca: PCA
    scaler: Optional[StandardScaler] = None
    original_dim: int = 0
    compressed_dim: int = 0
    explained_variance_ratio: float = 0.0
    training_samples: int = 0

class PCATrainer:
    """Handles PCA model training and persistence."""
    
    def __init__(self, target_dimensions: int = 128, use_scaling: bool = True):
        self.target_dimensions = target_dimensions
        self.use_scaling = use_scaling
        self.lock = threading.Lock()
    
    def train_pca_model(
        self, 
        embeddings: np.ndarray,
        explained_variance_threshold: float = 0.95
    ) -> PCAModel:
        """Train PCA model on collected embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            explained_variance_threshold: Minimum explained variance to retain
            
        Returns:
            PCAModel: Trained PCA model container
        """
        with self.lock:
            original_dim = embeddings.shape[1]
            n_samples = embeddings.shape[0]
            
            # Adaptive threshold based on sample size - be more flexible with smaller datasets
            min_samples_needed = max(self.target_dimensions, 10)  # Stricter requirement
            
            if n_samples < min_samples_needed:
                raise ValueError(
                    f"Need at least {min_samples_needed} samples to train PCA, "
                    f"got {n_samples}. Consider using a smaller target_dimensions or collecting more data."
                )
            
            # Optional scaling
            scaler = None
            if self.use_scaling:
                scaler = StandardScaler()
                embeddings = scaler.fit_transform(embeddings)
            
            # Determine optimal number of components - be more aggressive with small datasets
            if n_samples < 50:
                # For very small datasets, use much smaller dimensions
                n_components = min(
                    max(2, n_samples // 2),  # At least 2 components, but not more than half the samples
                    self.target_dimensions,
                    original_dim
                )
            else:
                # Normal case for larger datasets
                n_components = min(self.target_dimensions, n_samples - 1, original_dim)
            
            # Train PCA
            pca = PCA(n_components=n_components)
            pca.fit(embeddings)
            
            # Calculate explained variance
            explained_variance = np.sum(pca.explained_variance_ratio_)
            
            # For small datasets, be more lenient with explained variance requirements
            if n_samples < 50:
                min_variance_threshold = max(0.7, explained_variance_threshold - 0.2)
            else:
                min_variance_threshold = explained_variance_threshold
            
            # If explained variance is too low, try to increase components
            if explained_variance < min_variance_threshold and n_components < min(original_dim, n_samples - 1):
                if n_samples < 50:
                    # For small datasets, be more conservative
                    max_components = min(
                        n_samples - 1,  # Can't exceed samples - 1
                        int(original_dim * 0.8),  # Use up to 80% of original dimensions
                        self.target_dimensions * 2  # Allow up to 2x target dimensions for small datasets
                    )
                else:
                    # Original logic for larger datasets
                    max_components = min(
                        int(original_dim * 0.5),  # Don't use more than 50% of original dimensions
                        n_samples - 1
                    )
                
                if max_components > n_components:
                    pca = PCA(n_components=max_components)
                    pca.fit(embeddings)
                    explained_variance = np.sum(pca.explained_variance_ratio_)
                    n_components = max_components
            
            return PCAModel(
                pca=pca,
                scaler=scaler,
                original_dim=original_dim,
                compressed_dim=n_components,
                explained_variance_ratio=explained_variance,
                training_samples=embeddings.shape[0]
            )
    
    def save_model(self, model: PCAModel, filepath: str) -> None:
        """Save PCA model to disk."""
        with self.lock:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model components
            model_data = {
                'pca': model.pca,
                'scaler': model.scaler,
                'original_dim': model.original_dim,
                'compressed_dim': model.compressed_dim,
                'explained_variance_ratio': model.explained_variance_ratio,
                'training_samples': model.training_samples,
            }
            
            joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> Optional[PCAModel]:
        """Load PCA model from disk."""
        with self.lock:
            try:
                if not os.path.exists(filepath):
                    return None
                
                model_data = joblib.load(filepath)
                
                return PCAModel(
                    pca=model_data['pca'],
                    scaler=model_data.get('scaler'),
                    original_dim=model_data['original_dim'],
                    compressed_dim=model_data['compressed_dim'],
                    explained_variance_ratio=model_data['explained_variance_ratio'],
                    training_samples=model_data['training_samples']
                )
                
            except Exception as e:
                print(f"Failed to load PCA model from {filepath}: {e}")
                return None

class EmbeddingCompressor:
    """Handles embedding compression and decompression operations."""
    
    def __init__(self, pca_model: PCAModel):
        self.pca_model = pca_model
        self.lock = threading.Lock()
    
    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """Compress embeddings using trained PCA model.
        
        Args:
            embeddings: Array of shape (n_samples, original_dim) or (original_dim,)
            
        Returns:
            np.ndarray: Compressed embeddings
        """
        with self.lock:
            # Handle single embedding (1D array)
            is_single = embeddings.ndim == 1
            if is_single:
                embeddings = embeddings.reshape(1, -1)
            
            # Apply scaling if used during training
            if self.pca_model.scaler is not None:
                embeddings = self.pca_model.scaler.transform(embeddings)
            
            # Apply PCA compression
            compressed = self.pca_model.pca.transform(embeddings)
            
            # Return single embedding if input was single
            if is_single:
                return compressed[0]
            
            return compressed
    
    def decompress(self, compressed_embeddings: np.ndarray) -> np.ndarray:
        """Decompress embeddings using trained PCA model.
        
        Args:
            compressed_embeddings: Array of shape (n_samples, compressed_dim) or (compressed_dim,)
            
        Returns:
            np.ndarray: Decompressed embeddings (approximate reconstruction)
        """
        with self.lock:
            # Handle single embedding (1D array)
            is_single = compressed_embeddings.ndim == 1
            if is_single:
                compressed_embeddings = compressed_embeddings.reshape(1, -1)
            
            # Apply PCA decompression (inverse transform)
            decompressed = self.pca_model.pca.inverse_transform(compressed_embeddings)
            
            # Apply inverse scaling if used during training
            if self.pca_model.scaler is not None:
                decompressed = self.pca_model.scaler.inverse_transform(decompressed)
            
            # Return single embedding if input was single
            if is_single:
                return decompressed[0]
            
            return decompressed
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio (original_dim / compressed_dim)."""
        return self.pca_model.original_dim / self.pca_model.compressed_dim
    
    def get_explained_variance(self) -> float:
        """Get explained variance ratio."""
        return self.pca_model.explained_variance_ratio

class PCAEmbeddingWrapper:
    """Wrapper around any embedding function with PCA compression."""
    
    def __init__(
        self,
        base_embedding_func: Callable[[Any], np.ndarray],
        target_dimensions: Optional[int] = None,
        model_path: Optional[str] = None,
        auto_train: bool = True,
        training_samples_threshold: Optional[int] = None,
    ):
        """Initialize PCA embedding wrapper.
        
        Args:
            base_embedding_func: Base embedding function to wrap
            target_dimensions: Target embedding dimensions
            model_path: Path to save/load PCA model
            auto_train: Whether to automatically train PCA when threshold is reached
            training_samples_threshold: Number of samples needed before training PCA
        """
        config = get_config()
        
        self.base_embedding_func = base_embedding_func
        self.target_dimensions = target_dimensions or config.pca.target_dimensions
        self.model_path = model_path or config.pca.model_path
        self.auto_train = auto_train
        self.training_threshold = training_samples_threshold or config.pca.training_samples
        self.compression_threshold = config.pca.compression_threshold
        self.enabled = config.pca.enabled
        
        # PCA components
        self.trainer = PCATrainer(target_dimensions=self.target_dimensions)
        self.compressor: Optional[EmbeddingCompressor] = None
        
        # Embedding collection for training
        self.collected_embeddings: List[np.ndarray] = []
        self.lock = threading.Lock()
        
        # Statistics
        self.total_compressions = 0
        self.total_compression_time_ms = 0.0
        
        # Try to load existing model
        self._load_existing_model()
    
    def __call__(self, input_data: Any) -> np.ndarray:
        """Generate embedding with optional PCA compression.
        
        Args:
            input_data: Input to embed (text, etc.)
            
        Returns:
            np.ndarray: Embedding (compressed if PCA is available)
        """
        # Generate base embedding
        with BenchmarkTimer("base_embedding") as timer:
            base_embedding = self.base_embedding_func(input_data)
        
        base_latency = timer.elapsed_ms
        
        # Convert to numpy array if needed
        if not isinstance(base_embedding, np.ndarray):
            base_embedding = np.array(base_embedding)
        
        # If PCA is disabled or not available, return base embedding
        if not self.enabled or self.compressor is None:
            # Collect embeddings for future PCA training
            if self.enabled and len(self.collected_embeddings) < self.training_threshold:
                with self.lock:
                    self.collected_embeddings.append(base_embedding.copy())
                    
                    # Auto-train when threshold is reached
                    if (self.auto_train and 
                        len(self.collected_embeddings) >= self.training_threshold):
                        self._train_pca_model()
            
            return base_embedding
        
        # Apply PCA compression
        with BenchmarkTimer("pca_compression") as timer:
            compressed_embedding = self.compressor.compress(base_embedding)
        
        compression_latency = timer.elapsed_ms
        
        # Update statistics
        with self.lock:
            self.total_compressions += 1
            self.total_compression_time_ms += compression_latency
        
        # Record compression metrics
        compression_ratio = self.compressor.get_compression_ratio()
        record_cache_request(
            query=str(input_data)[:100],  # Truncate for privacy
            latency_ms=base_latency + compression_latency,
            cache_hit=False,  # This is embedding generation, not cache lookup
            embedding_compression_ratio=compression_ratio
        )
        
        return compressed_embedding
    
    def _load_existing_model(self) -> None:
        """Load existing PCA model if available."""
        pca_model = self.trainer.load_model(self.model_path)
        if pca_model is not None:
            self.compressor = EmbeddingCompressor(pca_model)
            print(f"Loaded PCA model: {pca_model.original_dim}D → {pca_model.compressed_dim}D "
                  f"({pca_model.explained_variance_ratio:.2%} variance explained)")
    
    def _train_pca_model(self) -> None:
        """Train PCA model on collected embeddings."""
        with self.lock:
            if len(self.collected_embeddings) < self.training_threshold:
                return
            
            print(f"Training PCA model on {len(self.collected_embeddings)} embeddings...")
            
            try:
                # Stack embeddings for training
                embeddings_array = np.vstack(self.collected_embeddings)
                
                # Train PCA model
                pca_model = self.trainer.train_pca_model(embeddings_array)
                
                # Create compressor
                self.compressor = EmbeddingCompressor(pca_model)
                
                # Save model
                self.trainer.save_model(pca_model, self.model_path)
                
                print(f"PCA model trained: {pca_model.original_dim}D → {pca_model.compressed_dim}D "
                      f"({pca_model.explained_variance_ratio:.2%} variance explained)")
                
                # Clear collected embeddings to save memory
                self.collected_embeddings.clear()
                
            except Exception as e:
                print(f"Failed to train PCA model: {e}")
    
    def force_train_pca(self, embeddings: Optional[np.ndarray] = None) -> bool:
        """Force PCA model training.
        
        Args:
            embeddings: Optional embeddings to use for training
            
        Returns:
            bool: True if training succeeded, False otherwise
        """
        with self.lock:
            training_data = embeddings
            
            if training_data is None:
                min_samples_needed = max(3, min(self.target_dimensions // 4, 10))
                if len(self.collected_embeddings) < min_samples_needed:
                    print(f"Insufficient embeddings for training: {len(self.collected_embeddings)} "
                          f"< {min_samples_needed} (minimum needed)")
                    return False
                training_data = np.vstack(self.collected_embeddings)
            
            try:
                pca_model = self.trainer.train_pca_model(training_data)
                self.compressor = EmbeddingCompressor(pca_model)
                self.trainer.save_model(pca_model, self.model_path)
                
                print(f"PCA model force-trained: {pca_model.original_dim}D → "
                      f"{pca_model.compressed_dim}D")
                return True
                
            except Exception as e:
                print(f"Failed to force-train PCA model: {e}")
                return False
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        with self.lock:
            stats = {
                'enabled': self.enabled,
                'model_loaded': self.compressor is not None,
                'collected_embeddings': len(self.collected_embeddings),
                'training_threshold': self.training_threshold,
                'total_compressions': self.total_compressions,
                'avg_compression_time_ms': (
                    self.total_compression_time_ms / self.total_compressions
                    if self.total_compressions > 0 else 0.0
                ),
            }
            
            if self.compressor is not None:
                stats.update({
                    'compression_ratio': self.compressor.get_compression_ratio(),
                    'explained_variance': self.compressor.get_explained_variance(),
                    'original_dimensions': self.compressor.pca_model.original_dim,
                    'compressed_dimensions': self.compressor.pca_model.compressed_dim,
                    'training_samples': self.compressor.pca_model.training_samples,
                })
        
        return stats
    
    def decompress_embedding(self, compressed_embedding: np.ndarray) -> np.ndarray:
        """Decompress a compressed embedding back to original space.
        
        Args:
            compressed_embedding: Compressed embedding
            
        Returns:
            np.ndarray: Decompressed embedding (approximate reconstruction)
        """
        if self.compressor is None:
            return compressed_embedding  # Return as-is if no compression available
        
        return self.compressor.decompress(compressed_embedding)

# Convenience function for creating PCA-wrapped embedding function
def create_pca_embedding_wrapper(
    base_embedding_func: Callable[[Any], np.ndarray],
    **kwargs
) -> PCAEmbeddingWrapper:
    """Create a PCA embedding wrapper with default configuration."""
    return PCAEmbeddingWrapper(
        base_embedding_func=base_embedding_func,
        **kwargs
    )

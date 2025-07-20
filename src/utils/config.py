"""Configuration management for Enhanced GPTCache."""
import os
import yaml
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class CacheConfig:
    """Cache-related configuration."""
    size_mb: int = 100
    similarity_threshold: float = 0.8
    eviction_policy: str = "lru"


@dataclass
class ContextConfig:
    """Context-chain filtering configuration."""
    enabled: bool = True
    window_size: int = 5
    divergence_threshold: float = 0.3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class PCAConfig:
    """PCA embedding compression configuration."""
    enabled: bool = True
    target_dimensions: int = 128
    training_samples: int = 1000
    model_path: str = "models/pca_model.pkl"
    compression_threshold: int = 100


@dataclass
class FederatedConfig:
    """Federated τ-tuning configuration."""
    enabled: bool = True
    num_users: int = 10
    aggregation_frequency: int = 100
    learning_rate: float = 0.01
    initial_tau: float = 0.8


@dataclass
class BenchmarkConfig:
    """Benchmarking configuration."""
    output_dir: str = "data/results"
    query_count: int = 1000
    variants: list = None
    
    def __post_init__(self):
        if self.variants is None:
            self.variants = ["baseline", "context", "pca", "tau", "full"]


@dataclass
class Config:
    """Main configuration class."""
    cache: CacheConfig
    context: ContextConfig
    pca: PCAConfig
    federated: FederatedConfig
    benchmark: BenchmarkConfig
    
    def __init__(
        self,
        cache: Optional[CacheConfig] = None,
        context: Optional[ContextConfig] = None,
        pca: Optional[PCAConfig] = None,
        federated: Optional[FederatedConfig] = None,
        benchmark: Optional[BenchmarkConfig] = None,
    ):
        self.cache = cache or CacheConfig()
        self.context = context or ContextConfig()
        self.pca = pca or PCAConfig()
        self.federated = federated or FederatedConfig()
        self.benchmark = benchmark or BenchmarkConfig()


class ConfigManager:
    """Manages configuration loading and environment variable overrides."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config: Optional[Config] = None
    
    def load_config(self) -> Config:
        """Load configuration from file and environment variables."""
        if self._config is not None:
            return self._config
            
        # Load from YAML file if exists
        config_dict = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Create config objects
        cache_config = CacheConfig(**config_dict.get('cache', {}))
        context_config = ContextConfig(**config_dict.get('context', {}))
        pca_config = PCAConfig(**config_dict.get('pca', {}))
        federated_config = FederatedConfig(**config_dict.get('federated', {}))
        benchmark_config = BenchmarkConfig(**config_dict.get('benchmark', {}))
        
        self._config = Config(
            cache=cache_config,
            context=context_config,
            pca=pca_config,
            federated=federated_config,
            benchmark=benchmark_config,
        )
        
        return self._config
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            # Cache config
            'CACHE_SIZE_MB': ('cache', 'size_mb', int),
            'CACHE_SIMILARITY_THRESHOLD': ('cache', 'similarity_threshold', float),
            'CACHE_EVICTION_POLICY': ('cache', 'eviction_policy', str),
            
            # Context config
            'CONTEXT_ENABLED': ('context', 'enabled', lambda x: x.lower() == 'true'),
            'CONTEXT_WINDOW_SIZE': ('context', 'window_size', int),
            'CONTEXT_DIVERGENCE_THRESHOLD': ('context', 'divergence_threshold', float),
            'CONTEXT_EMBEDDING_MODEL': ('context', 'embedding_model', str),
            
            # PCA config
            'PCA_ENABLED': ('pca', 'enabled', lambda x: x.lower() == 'true'),
            'PCA_TARGET_DIMENSIONS': ('pca', 'target_dimensions', int),
            'PCA_TRAINING_SAMPLES': ('pca', 'training_samples', int),
            'PCA_MODEL_PATH': ('pca', 'model_path', str),
            'PCA_COMPRESSION_THRESHOLD': ('pca', 'compression_threshold', int),
            
            # Federated config
            'FEDERATED_ENABLED': ('federated', 'enabled', lambda x: x.lower() == 'true'),
            'FEDERATED_NUM_USERS': ('federated', 'num_users', int),
            'FEDERATED_AGGREGATION_FREQUENCY': ('federated', 'aggregation_frequency', int),
            'FEDERATED_LEARNING_RATE': ('federated', 'learning_rate', float),
            'FEDERATED_INITIAL_TAU': ('federated', 'initial_tau', float),
            
            # Benchmark config
            'BENCHMARK_OUTPUT_DIR': ('benchmark', 'output_dir', str),
            'BENCHMARK_QUERY_COUNT': ('benchmark', 'query_count', int),
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            if env_var in os.environ:
                if section not in config_dict:
                    config_dict[section] = {}
                try:
                    config_dict[section][key] = converter(os.environ[env_var])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {e}")
        
        return config_dict
    
    def save_config(self, config: Config, path: Optional[str] = None) -> None:
        """Save configuration to YAML file."""
        save_path = path or self.config_path
        config_dict = {
            'cache': asdict(config.cache),
            'context': asdict(config.context),
            'pca': asdict(config.pca),
            'federated': asdict(config.federated),
            'benchmark': asdict(config.benchmark),
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.load_config()


def init_config(config_path: Optional[str] = None) -> Config:
    """Initialize configuration with custom path."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager.load_config()

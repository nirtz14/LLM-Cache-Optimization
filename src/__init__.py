"""Enhanced GPTCache with MeanCache features."""

__version__ = "0.1.0"
__author__ = "Enhanced Cache Team"
__description__ = "GPTCache extension with context-chain filtering, PCA compression, and federated τ-tuning"

from .utils.config import get_config, init_config
from .utils.metrics import get_performance_tracker, record_cache_request

__all__ = [
    "get_config",
    "init_config", 
    "get_performance_tracker",
    "record_cache_request",
]

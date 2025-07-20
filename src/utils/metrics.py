"""Performance metrics collection and tracking utilities."""
import time
import psutil
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

@dataclass
class RequestMetrics:
    """Metrics for a single cache request."""
    query: str
    timestamp: float
    latency_ms: float
    cache_hit: bool
    memory_usage_mb: float
    cpu_percent: float
    context_similarity: Optional[float] = None
    embedding_compression_ratio: Optional[float] = None
    tau_threshold: Optional[float] = None

@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time period."""
    total_requests: int = 0
    cache_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_context_similarity: Optional[float] = None
    avg_compression_ratio: Optional[float] = None
    avg_tau_threshold: Optional[float] = None

class PerformanceTracker:
    """Tracks cache performance metrics and statistics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
        # Real-time counters
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Latency tracking
        self.latencies: deque = deque(maxlen=window_size)
        
        # System resource tracking
        self.process = psutil.Process()
        
    def record_request(
        self,
        query: str,
        latency_ms: float,
        cache_hit: bool,
        context_similarity: Optional[float] = None,
        embedding_compression_ratio: Optional[float] = None,
        tau_threshold: Optional[float] = None,
    ) -> None:
        """Record metrics for a single cache request."""
        with self.lock:
            # System metrics
            memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = self.process.cpu_percent()
            
            metrics = RequestMetrics(
                query=query,
                timestamp=time.time(),
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                context_similarity=context_similarity,
                embedding_compression_ratio=embedding_compression_ratio,
                tau_threshold=tau_threshold,
            )
            
            self.metrics_history.append(metrics)
            self.latencies.append(latency_ms)
            
            # Update counters
            self.total_requests += 1
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def get_current_stats(self) -> AggregatedMetrics:
        """Get current aggregated statistics."""
        with self.lock:
            if not self.metrics_history:
                return AggregatedMetrics()
            
            recent_metrics = list(self.metrics_history)
            latencies = [m.latency_ms for m in recent_metrics]
            memory_usage = [m.memory_usage_mb for m in recent_metrics]
            cpu_usage = [m.cpu_percent for m in recent_metrics]
            
            # Context similarity stats (filter None values)
            context_similarities = [
                m.context_similarity for m in recent_metrics 
                if m.context_similarity is not None
            ]
            
            # Compression ratio stats
            compression_ratios = [
                m.embedding_compression_ratio for m in recent_metrics
                if m.embedding_compression_ratio is not None
            ]
            
            # Tau threshold stats
            tau_thresholds = [
                m.tau_threshold for m in recent_metrics
                if m.tau_threshold is not None
            ]
            
            return AggregatedMetrics(
                total_requests=len(recent_metrics),
                cache_hit_rate=sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics),
                avg_latency_ms=np.mean(latencies),
                p95_latency_ms=np.percentile(latencies, 95),
                p99_latency_ms=np.percentile(latencies, 99),
                avg_memory_mb=np.mean(memory_usage),
                avg_cpu_percent=np.mean(cpu_usage),
                avg_context_similarity=np.mean(context_similarities) if context_similarities else None,
                avg_compression_ratio=np.mean(compression_ratios) if compression_ratios else None,
                avg_tau_threshold=np.mean(tau_thresholds) if tau_thresholds else None,
            )
    
    def get_hit_rate(self) -> float:
        """Get current cache hit rate."""
        with self.lock:
            if self.total_requests == 0:
                return 0.0
            return self.cache_hits / self.total_requests
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        with self.lock:
            if not self.latencies:
                return {'mean': 0.0, 'p95': 0.0, 'p99': 0.0}
            
            latencies = list(self.latencies)
            return {
                'mean': np.mean(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'min': np.min(latencies),
                'max': np.max(latencies),
            }
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self.lock:
            self.metrics_history.clear()
            self.latencies.clear()
            self.total_requests = 0
            self.cache_hits = 0
            self.cache_misses = 0
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export all metrics as a list of dictionaries."""
        with self.lock:
            return [
                {
                    'query': m.query,
                    'timestamp': m.timestamp,
                    'latency_ms': m.latency_ms,
                    'cache_hit': m.cache_hit,
                    'memory_usage_mb': m.memory_usage_mb,
                    'cpu_percent': m.cpu_percent,
                    'context_similarity': m.context_similarity,
                    'embedding_compression_ratio': m.embedding_compression_ratio,
                    'tau_threshold': m.tau_threshold,
                }
                for m in self.metrics_history
            ]

class ConfusionMatrix:
    """Track classification metrics for cache hit/miss decisions."""
    
    def __init__(self):
        self.true_positives = 0   # Correct cache hits
        self.false_positives = 0  # Incorrect cache hits (should have been miss)
        self.true_negatives = 0   # Correct cache misses
        self.false_negatives = 0  # Incorrect cache misses (should have been hit)
        self.lock = threading.Lock()
    
    def record_prediction(self, predicted_hit: bool, actual_hit: bool) -> None:
        """Record a cache hit/miss prediction."""
        with self.lock:
            if predicted_hit and actual_hit:
                self.true_positives += 1
            elif predicted_hit and not actual_hit:
                self.false_positives += 1
            elif not predicted_hit and not actual_hit:
                self.true_negatives += 1
            else:  # not predicted_hit and actual_hit
                self.false_negatives += 1
    
    def get_precision(self) -> float:
        """Calculate precision: TP / (TP + FP)."""
        with self.lock:
            denominator = self.true_positives + self.false_positives
            return self.true_positives / denominator if denominator > 0 else 0.0
    
    def get_recall(self) -> float:
        """Calculate recall: TP / (TP + FN)."""
        with self.lock:
            denominator = self.true_positives + self.false_negatives
            return self.true_positives / denominator if denominator > 0 else 0.0
    
    def get_f1_score(self) -> float:
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall)."""
        precision = self.get_precision()
        recall = self.get_recall()
        denominator = precision + recall
        return 2 * (precision * recall) / denominator if denominator > 0 else 0.0
    
    def get_accuracy(self) -> float:
        """Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)."""
        with self.lock:
            total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
            return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get all classification statistics."""
        return {
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'f1_score': self.get_f1_score(),
            'accuracy': self.get_accuracy(),
        }
    
    def reset(self) -> None:
        """Reset all counters."""
        with self.lock:
            self.true_positives = 0
            self.false_positives = 0
            self.true_negatives = 0
            self.false_negatives = 0

class BenchmarkTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = 0.0
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.time() - self.start_time) * 1000

# Global performance tracker instance
_performance_tracker: Optional[PerformanceTracker] = None

def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker

def record_cache_request(
    query: str,
    latency_ms: float,
    cache_hit: bool,
    **kwargs
) -> None:
    """Record a cache request in the global performance tracker."""
    tracker = get_performance_tracker()
    tracker.record_request(query, latency_ms, cache_hit, **kwargs)

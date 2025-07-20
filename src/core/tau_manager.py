"""Federated τ-tuning for Enhanced GPTCache threshold optimization."""
import threading
import time
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

from ..utils.config import get_config
from ..utils.metrics import record_cache_request, ConfusionMatrix

@dataclass
class ThresholdUpdate:
    """Represents a threshold update from a user."""
    user_id: str
    old_threshold: float
    new_threshold: float
    performance_gain: float
    timestamp: float
    confidence: float = 1.0

@dataclass
class UserState:
    """State for a single federated user."""
    user_id: str
    current_threshold: float
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    confusion_matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    total_queries: int = 0
    last_update_time: float = 0.0
    learning_rate: float = 0.01
    
    def get_current_performance(self) -> Dict[str, float]:
        """Get current performance metrics."""
        stats = self.confusion_matrix.get_stats()
        stats['total_queries'] = self.total_queries
        return stats
    
    def reset_performance(self) -> None:
        """Reset performance tracking."""
        self.confusion_matrix.reset()
        self.performance_history.clear()

class ThresholdOptimizer:
    """Local threshold optimization algorithms."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def gradient_based_update(
        self,
        current_threshold: float,
        performance_metrics: Dict[str, float],
        target_metric: str = "f1_score"
    ) -> float:
        """Simple gradient-based threshold update.
        
        Args:
            current_threshold: Current similarity threshold
            performance_metrics: Current performance metrics
            target_metric: Metric to optimize (f1_score, precision, recall)
            
        Returns:
            float: Updated threshold
        """
        current_score = performance_metrics.get(target_metric, 0.0)
        
        # Simple heuristic: if F1 score is low, adjust threshold
        if current_score < 0.5:
            # If precision is much higher than recall, lower threshold
            precision = performance_metrics.get('precision', 0.0)
            recall = performance_metrics.get('recall', 0.0)
            
            if precision > recall + 0.1:
                # Lower threshold to increase recall
                adjustment = -self.learning_rate * (precision - recall)
            elif recall > precision + 0.1:
                # Raise threshold to increase precision
                adjustment = self.learning_rate * (recall - precision)
            else:
                # Small random exploration
                adjustment = random.uniform(-self.learning_rate, self.learning_rate)
        else:
            # Good performance, small random exploration
            adjustment = random.uniform(-self.learning_rate/2, self.learning_rate/2)
        
        new_threshold = np.clip(current_threshold + adjustment, 0.1, 0.95)
        return new_threshold
    
    def bandit_based_update(
        self,
        current_threshold: float,
        performance_history: List[Dict[str, float]],
        exploration_rate: float = 0.1
    ) -> float:
        """Bandit-style threshold exploration.
        
        Args:
            current_threshold: Current threshold
            performance_history: History of performance metrics
            exploration_rate: Exploration vs exploitation balance
            
        Returns:
            float: Updated threshold
        """
        if len(performance_history) < 5:
            # Not enough history, use gradient method
            latest_performance = performance_history[-1] if performance_history else {}
            return self.gradient_based_update(current_threshold, latest_performance)
        
        # Explore vs exploit decision
        if random.random() < exploration_rate:
            # Exploration: random threshold adjustment
            adjustment = random.uniform(-0.1, 0.1)
            return np.clip(current_threshold + adjustment, 0.1, 0.95)
        else:
            # Exploitation: move toward best performing threshold
            best_performance = max(performance_history[-10:], key=lambda x: x.get('f1_score', 0.0))
            best_threshold = best_performance.get('threshold', current_threshold)
            
            # Move partially toward best threshold
            movement = (best_threshold - current_threshold) * self.learning_rate
            return np.clip(current_threshold + movement, 0.1, 0.95)

class PerformanceTracker:
    """Tracks performance metrics for threshold optimization."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.query_results: deque = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def record_query(
        self,
        query: str,
        similarity_score: float,
        threshold: float,
        cache_hit: bool,
        ground_truth_hit: Optional[bool] = None
    ) -> None:
        """Record a query result for performance tracking."""
        with self.lock:
            predicted_hit = similarity_score >= threshold
            
            # If ground truth is not available, use heuristic
            if ground_truth_hit is None:
                # Simple heuristic: very high similarity scores are likely correct hits
                ground_truth_hit = similarity_score > 0.9 or (
                    cache_hit and similarity_score > 0.7
                )
            
            result = {
                'query': query,
                'similarity_score': similarity_score,
                'threshold': threshold,
                'predicted_hit': predicted_hit,
                'actual_hit': ground_truth_hit,
                'cache_hit': cache_hit,
                'timestamp': time.time(),
            }
            
            self.query_results.append(result)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from recent queries."""
        with self.lock:
            if not self.query_results:
                return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'accuracy': 0.0}
            
            # Build confusion matrix
            tp = fp = tn = fn = 0
            
            for result in self.query_results:
                predicted = result['predicted_hit']
                actual = result['actual_hit']
                
                if predicted and actual:
                    tp += 1
                elif predicted and not actual:
                    fp += 1
                elif not predicted and not actual:
                    tn += 1
                else:  # not predicted and actual
                    fn += 1
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (2 * precision * recall / (precision + recall) 
                       if (precision + recall) > 0 else 0.0)
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'total_samples': len(self.query_results),
            }

class FederatedAggregator:
    """Simulates federated learning aggregation for threshold updates."""
    
    def __init__(
        self,
        num_users: int = 10,
        aggregation_frequency: int = 100,
        min_participants: int = 3,
    ):
        self.num_users = num_users
        self.aggregation_frequency = aggregation_frequency
        self.min_participants = min_participants
        
        # Track updates from users
        self.pending_updates: List[ThresholdUpdate] = []
        self.update_history: List[ThresholdUpdate] = []
        self.lock = threading.Lock()
        
        # Global threshold state
        self.global_threshold: float = 0.8
        self.last_aggregation_time: float = time.time()
        self.total_aggregations: int = 0
    
    def submit_update(self, update: ThresholdUpdate) -> None:
        """Submit a threshold update from a user."""
        with self.lock:
            self.pending_updates.append(update)
            self.update_history.append(update)
            
            # Trigger aggregation if enough updates
            if len(self.pending_updates) >= self.aggregation_frequency:
                self._aggregate_updates()
    
    def _aggregate_updates(self) -> None:
        """Aggregate pending threshold updates using federated averaging."""
        if len(self.pending_updates) < self.min_participants:
            return
        
        # Weighted federated averaging
        total_weight = 0.0
        weighted_sum = 0.0
        
        for update in self.pending_updates:
            # Weight by performance gain and confidence
            weight = update.performance_gain * update.confidence
            total_weight += weight
            weighted_sum += weight * update.new_threshold
        
        if total_weight > 0:
            # Update global threshold
            new_global_threshold = weighted_sum / total_weight
            
            # Apply momentum to avoid drastic changes
            momentum = 0.7
            self.global_threshold = (
                momentum * self.global_threshold + 
                (1 - momentum) * new_global_threshold
            )
            
            # Clip to reasonable bounds
            self.global_threshold = np.clip(self.global_threshold, 0.1, 0.95)
        
        # Clear pending updates and update stats
        self.pending_updates.clear()
        self.last_aggregation_time = time.time()
        self.total_aggregations += 1
        
        print(f"Federated aggregation #{self.total_aggregations}: "
              f"Global threshold updated to {self.global_threshold:.3f}")
    
    def get_global_threshold(self) -> float:
        """Get current global threshold."""
        with self.lock:
            return self.global_threshold
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about federated aggregation."""
        with self.lock:
            return {
                'global_threshold': self.global_threshold,
                'total_aggregations': self.total_aggregations,
                'pending_updates': len(self.pending_updates),
                'total_updates': len(self.update_history),
                'last_aggregation_time': self.last_aggregation_time,
                'num_users': self.num_users,
                'aggregation_frequency': self.aggregation_frequency,
            }

class TauManager:
    """Main class for managing federated τ-tuning."""
    
    def __init__(
        self,
        num_users: Optional[int] = None,
        aggregation_frequency: Optional[int] = None,
        learning_rate: Optional[float] = None,
        initial_tau: Optional[float] = None,
    ):
        """Initialize τ-tuning manager.
        
        Args:
            num_users: Number of simulated federated users
            aggregation_frequency: Updates between aggregations
            learning_rate: Threshold adjustment rate
            initial_tau: Starting similarity threshold
        """
        config = get_config()
        
        self.num_users = num_users or config.federated.num_users
        self.aggregation_frequency = aggregation_frequency or config.federated.aggregation_frequency
        self.learning_rate = learning_rate or config.federated.learning_rate
        self.initial_tau = initial_tau or config.federated.initial_tau
        self.enabled = config.federated.enabled
        
        # Initialize components
        self.aggregator = FederatedAggregator(
            num_users=self.num_users,
            aggregation_frequency=self.aggregation_frequency
        )
        
        self.optimizer = ThresholdOptimizer(learning_rate=self.learning_rate)
        
        # User states
        self.users: Dict[str, UserState] = {}
        self.performance_trackers: Dict[str, PerformanceTracker] = {}
        self.lock = threading.Lock()
        
        # Initialize simulated users
        self._initialize_users()
        
        # Current threshold (starts as global threshold)
        self.current_threshold = self.initial_tau
        self.query_count = 0
    
    def _initialize_users(self) -> None:
        """Initialize simulated federated users."""
        with self.lock:
            for i in range(self.num_users):
                user_id = f"user_{i:03d}"
                
                # Add some variation to initial thresholds
                user_threshold = self.initial_tau + random.uniform(-0.1, 0.1)
                user_threshold = np.clip(user_threshold, 0.1, 0.95)
                
                self.users[user_id] = UserState(
                    user_id=user_id,
                    current_threshold=user_threshold,
                    learning_rate=self.learning_rate + random.uniform(-0.005, 0.005)
                )
                
                self.performance_trackers[user_id] = PerformanceTracker()
    
    def evaluate_threshold(
        self,
        query: str,
        similarity_score: float,
        cache_hit: bool,
        ground_truth_hit: Optional[bool] = None
    ) -> float:
        """Evaluate and potentially update threshold based on query result.
        
        Args:
            query: Query text
            similarity_score: Computed similarity score
            cache_hit: Whether cache hit occurred
            ground_truth_hit: Ground truth for cache hit (optional)
            
        Returns:
            float: Current threshold to use
        """
        if not self.enabled:
            return self.current_threshold
        
        self.query_count += 1
        
        # Select a random user to simulate distributed queries
        user_id = random.choice(list(self.users.keys()))
        user_state = self.users[user_id]
        tracker = self.performance_trackers[user_id]
        
        # Record query for this user
        tracker.record_query(
            query=query,
            similarity_score=similarity_score,
            threshold=user_state.current_threshold,
            cache_hit=cache_hit,
            ground_truth_hit=ground_truth_hit
        )
        
        user_state.total_queries += 1
        
        # Periodically update user's local threshold
        if user_state.total_queries % 20 == 0:  # Update every 20 queries
            self._update_user_threshold(user_id)
        
        # Use global threshold if available, otherwise user's local threshold
        global_threshold = self.aggregator.get_global_threshold()
        self.current_threshold = global_threshold if global_threshold != self.initial_tau else user_state.current_threshold
        
        # Record metrics
        record_cache_request(
            query=query,
            latency_ms=0.0,
            cache_hit=cache_hit,
            tau_threshold=self.current_threshold
        )
        
        return self.current_threshold
    
    def _update_user_threshold(self, user_id: str) -> None:
        """Update threshold for a specific user."""
        with self.lock:
            user_state = self.users[user_id]
            tracker = self.performance_trackers[user_id]
            
            # Get current performance
            current_performance = tracker.get_performance_metrics()
            
            # Optimize threshold
            old_threshold = user_state.current_threshold
            new_threshold = self.optimizer.gradient_based_update(
                current_threshold=old_threshold,
                performance_metrics=current_performance
            )
            
            # Calculate performance gain
            old_f1 = user_state.performance_history[-1].get('f1_score', 0.0) if user_state.performance_history else 0.0
            new_f1 = current_performance.get('f1_score', 0.0)
            performance_gain = max(0.0, new_f1 - old_f1)  # Only positive gains
            
            # Update user state
            user_state.current_threshold = new_threshold
            user_state.performance_history.append(current_performance)
            user_state.last_update_time = time.time()
            
            # Submit update to federated aggregator if there's improvement
            if performance_gain > 0.01:  # Minimum improvement threshold
                update = ThresholdUpdate(
                    user_id=user_id,
                    old_threshold=old_threshold,
                    new_threshold=new_threshold,
                    performance_gain=performance_gain,
                    timestamp=time.time(),
                    confidence=min(1.0, current_performance.get('total_samples', 0) / 50.0)
                )
                
                self.aggregator.submit_update(update)
    
    def get_current_threshold(self) -> float:
        """Get current threshold value."""
        return self.current_threshold
    
    def get_tau_stats(self) -> Dict[str, Any]:
        """Get statistics about τ-tuning."""
        with self.lock:
            user_stats = {}
            for user_id, user_state in self.users.items():
                performance = self.performance_trackers[user_id].get_performance_metrics()
                user_stats[user_id] = {
                    'current_threshold': user_state.current_threshold,
                    'total_queries': user_state.total_queries,
                    'performance': performance,
                    'last_update_time': user_state.last_update_time,
                }
        
        aggregator_stats = self.aggregator.get_aggregation_stats()
        
        return {
            'enabled': self.enabled,
            'current_threshold': self.current_threshold,
            'total_queries': self.query_count,
            'num_users': self.num_users,
            'user_statistics': user_stats,
            'aggregator_statistics': aggregator_stats,
        }
    
    def force_aggregation(self) -> None:
        """Force federated aggregation regardless of frequency."""
        with self.lock:
            self.aggregator._aggregate_updates()
    
    def reset_user_performance(self, user_id: Optional[str] = None) -> None:
        """Reset performance tracking for specific user or all users."""
        with self.lock:
            if user_id:
                if user_id in self.users:
                    self.users[user_id].reset_performance()
                    self.performance_trackers[user_id] = PerformanceTracker()
            else:
                # Reset all users
                for uid in self.users.keys():
                    self.users[uid].reset_performance()
                    self.performance_trackers[uid] = PerformanceTracker()

# Convenience function for creating tau manager
def create_tau_manager(**kwargs) -> TauManager:
    """Create a τ-tuning manager with default configuration."""
    return TauManager(**kwargs)

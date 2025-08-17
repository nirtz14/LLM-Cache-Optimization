"""Comprehensive test suite for Tau manager functionality."""
import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from src.core.tau_manager import (
    ThresholdUpdate,
    UserState,
    ThresholdOptimizer,
    PerformanceTracker,
    FederatedAggregator,
    TauManager,
    create_tau_manager
)


class TestThresholdUpdate:
    """Test ThresholdUpdate dataclass."""
    
    def test_threshold_update_creation(self):
        """Test creating a ThresholdUpdate instance."""
        update = ThresholdUpdate(
            user_id="user_001",
            old_threshold=0.8,
            new_threshold=0.75,
            performance_gain=0.05,
            timestamp=1234567890.0,
            confidence=0.9
        )
        
        assert update.user_id == "user_001"
        assert update.old_threshold == 0.8
        assert update.new_threshold == 0.75
        assert update.performance_gain == 0.05
        assert update.timestamp == 1234567890.0
        assert update.confidence == 0.9
    
    def test_threshold_update_default_confidence(self):
        """Test ThresholdUpdate with default confidence."""
        update = ThresholdUpdate(
            user_id="user_001",
            old_threshold=0.8,
            new_threshold=0.75,
            performance_gain=0.05,
            timestamp=1234567890.0
        )
        
        assert update.confidence == 1.0  # Default value


class TestUserState:
    """Test UserState functionality."""
    
    def test_user_state_creation(self):
        """Test creating a UserState instance."""
        user_state = UserState(
            user_id="user_001",
            current_threshold=0.8,
            learning_rate=0.01
        )
        
        assert user_state.user_id == "user_001"
        assert user_state.current_threshold == 0.8
        assert user_state.learning_rate == 0.01
        assert user_state.total_queries == 0
        assert user_state.last_update_time == 0.0
        assert isinstance(user_state.performance_history, deque)
        assert user_state.performance_history.maxlen == 100
    
    def test_get_current_performance(self):
        """Test getting current performance metrics."""
        user_state = UserState(
            user_id="user_001",
            current_threshold=0.8
        )
        
        # Mock some confusion matrix data
        user_state.confusion_matrix.true_positives = 10
        user_state.confusion_matrix.false_positives = 5
        user_state.confusion_matrix.true_negatives = 15
        user_state.confusion_matrix.false_negatives = 3
        user_state.total_queries = 33
        
        performance = user_state.get_current_performance()
        
        assert performance['total_queries'] == 33
        assert 'precision' in performance
        assert 'recall' in performance
        assert 'f1_score' in performance
    
    def test_reset_performance(self):
        """Test resetting performance metrics."""
        user_state = UserState(
            user_id="user_001",
            current_threshold=0.8
        )
        
        # Add some data
        user_state.performance_history.append({'f1_score': 0.8})
        user_state.confusion_matrix.true_positives = 10
        
        user_state.reset_performance()
        
        assert len(user_state.performance_history) == 0
        assert user_state.confusion_matrix.true_positives == 0


class TestThresholdOptimizer:
    """Test ThresholdOptimizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = ThresholdOptimizer(learning_rate=0.01)
    
    def test_optimizer_initialization(self):
        """Test ThresholdOptimizer initialization."""
        optimizer = ThresholdOptimizer(learning_rate=0.05)
        assert optimizer.learning_rate == 0.05
    
    def test_gradient_based_update_low_f1(self):
        """Test gradient-based update with low F1 score."""
        current_threshold = 0.8
        performance_metrics = {
            'f1_score': 0.3,
            'precision': 0.8,
            'recall': 0.2
        }
        
        new_threshold = self.optimizer.gradient_based_update(
            current_threshold, performance_metrics
        )
        
        # Should lower threshold to increase recall
        assert new_threshold < current_threshold
        assert 0.1 <= new_threshold <= 0.95
    
    def test_gradient_based_update_high_recall_low_precision(self):
        """Test gradient-based update with high recall, low precision."""
        current_threshold = 0.5
        performance_metrics = {
            'f1_score': 0.3,
            'precision': 0.2,
            'recall': 0.8
        }
        
        new_threshold = self.optimizer.gradient_based_update(
            current_threshold, performance_metrics
        )
        
        # Should raise threshold to increase precision
        assert new_threshold > current_threshold
        assert 0.1 <= new_threshold <= 0.95
    
    def test_gradient_based_update_good_performance(self):
        """Test gradient-based update with good performance."""
        current_threshold = 0.7
        performance_metrics = {
            'f1_score': 0.8,
            'precision': 0.8,
            'recall': 0.8
        }
        
        new_threshold = self.optimizer.gradient_based_update(
            current_threshold, performance_metrics
        )
        
        # Should make small adjustments only
        assert abs(new_threshold - current_threshold) < 0.1
        assert 0.1 <= new_threshold <= 0.95
    
    def test_bandit_based_update_insufficient_history(self):
        """Test bandit-based update with insufficient history."""
        current_threshold = 0.7
        performance_history = [
            {'f1_score': 0.5, 'threshold': 0.6},
            {'f1_score': 0.6, 'threshold': 0.7}
        ]
        
        # Should fall back to gradient method
        with patch.object(self.optimizer, 'gradient_based_update') as mock_gradient:
            mock_gradient.return_value = 0.75
            
            result = self.optimizer.bandit_based_update(
                current_threshold, performance_history
            )
            
            mock_gradient.assert_called_once()
            assert result == 0.75
    
    def test_bandit_based_update_exploration(self):
        """Test bandit-based update in exploration mode."""
        current_threshold = 0.7
        performance_history = [
            {'f1_score': 0.5, 'threshold': 0.6},
            {'f1_score': 0.6, 'threshold': 0.7},
            {'f1_score': 0.7, 'threshold': 0.8},
            {'f1_score': 0.8, 'threshold': 0.75},
            {'f1_score': 0.6, 'threshold': 0.65}
        ]
        
        # Force exploration
        with patch('random.random', return_value=0.05):  # < exploration_rate
            new_threshold = self.optimizer.bandit_based_update(
                current_threshold, performance_history, exploration_rate=0.1
            )
            
            # Should be different from current (exploration)
            assert new_threshold != current_threshold
            assert 0.1 <= new_threshold <= 0.95
    
    def test_bandit_based_update_exploitation(self):
        """Test bandit-based update in exploitation mode."""
        current_threshold = 0.7
        performance_history = [
            {'f1_score': 0.5, 'threshold': 0.6},
            {'f1_score': 0.6, 'threshold': 0.7},
            {'f1_score': 0.7, 'threshold': 0.8},
            {'f1_score': 0.9, 'threshold': 0.75},  # Best performance
            {'f1_score': 0.6, 'threshold': 0.65}
        ]
        
        # Force exploitation
        with patch('random.random', return_value=0.5):  # > exploration_rate
            new_threshold = self.optimizer.bandit_based_update(
                current_threshold, performance_history, exploration_rate=0.1
            )
            
            # Should move toward best threshold (0.75)
            if current_threshold < 0.75:
                assert new_threshold > current_threshold
            elif current_threshold > 0.75:
                assert new_threshold < current_threshold


class TestPerformanceTracker:
    """Test PerformanceTracker functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = PerformanceTracker(window_size=50)
    
    def test_tracker_initialization(self):
        """Test PerformanceTracker initialization."""
        tracker = PerformanceTracker(window_size=100)
        assert tracker.window_size == 100
        assert tracker.query_results.maxlen == 100
        assert len(tracker.query_results) == 0
    
    def test_record_query_with_ground_truth(self):
        """Test recording query with ground truth."""
        self.tracker.record_query(
            query="test query",
            similarity_score=0.85,
            threshold=0.8,
            cache_hit=True,
            ground_truth_hit=True
        )
        
        assert len(self.tracker.query_results) == 1
        result = self.tracker.query_results[0]
        
        assert result['query'] == "test query"
        assert result['similarity_score'] == 0.85
        assert result['threshold'] == 0.8
        assert result['predicted_hit'] is True  # 0.85 >= 0.8
        assert result['actual_hit'] is True
        assert result['cache_hit'] is True
    
    def test_record_query_without_ground_truth(self):
        """Test recording query without ground truth (uses heuristic)."""
        self.tracker.record_query(
            query="test query",
            similarity_score=0.95,
            threshold=0.8,
            cache_hit=True,
            ground_truth_hit=None  # Will use heuristic
        )
        
        result = self.tracker.query_results[0]
        assert result['actual_hit'] is True  # High similarity score
    
    def test_record_query_heuristic_moderate_score(self):
        """Test recording query with moderate similarity score."""
        self.tracker.record_query(
            query="test query",
            similarity_score=0.75,
            threshold=0.8,
            cache_hit=True,
            ground_truth_hit=None
        )
        
        result = self.tracker.query_results[0]
        assert result['actual_hit'] is True  # cache_hit=True and score > 0.7
    
    def test_get_performance_metrics_empty(self):
        """Test getting performance metrics with no data."""
        metrics = self.tracker.get_performance_metrics()
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
        assert metrics['accuracy'] == 0.0
    
    def test_get_performance_metrics_with_data(self):
        """Test getting performance metrics with data."""
        # Add some test data: TP, FP, TN, FN
        test_cases = [
            (0.9, 0.8, True, True),   # TP: predicted=True, actual=True
            (0.9, 0.8, True, False),  # FP: predicted=True, actual=False
            (0.7, 0.8, False, False), # TN: predicted=False, actual=False
            (0.7, 0.8, False, True),  # FN: predicted=False, actual=True
        ]
        
        for score, threshold, cache_hit, ground_truth in test_cases:
            self.tracker.record_query(
                query="test",
                similarity_score=score,
                threshold=threshold,
                cache_hit=cache_hit,
                ground_truth_hit=ground_truth
            )
        
        metrics = self.tracker.get_performance_metrics()
        
        # TP=1, FP=1, TN=1, FN=1
        expected_precision = 1 / (1 + 1)  # TP / (TP + FP) = 0.5
        expected_recall = 1 / (1 + 1)     # TP / (TP + FN) = 0.5
        expected_f1 = 2 * 0.5 * 0.5 / (0.5 + 0.5)  # 0.5
        expected_accuracy = (1 + 1) / (1 + 1 + 1 + 1)  # (TP + TN) / total = 0.5
        
        assert abs(metrics['precision'] - expected_precision) < 1e-6
        assert abs(metrics['recall'] - expected_recall) < 1e-6
        assert abs(metrics['f1_score'] - expected_f1) < 1e-6
        assert abs(metrics['accuracy'] - expected_accuracy) < 1e-6
        assert metrics['total_samples'] == 4
    
    def test_window_size_enforcement(self):
        """Test that window size is enforced."""
        tracker = PerformanceTracker(window_size=3)
        
        # Add more queries than window size
        for i in range(5):
            tracker.record_query(
                query=f"query {i}",
                similarity_score=0.8,
                threshold=0.7,
                cache_hit=True,
                ground_truth_hit=True
            )
        
        # Should only keep last 3
        assert len(tracker.query_results) == 3
        assert tracker.query_results[0]['query'] == "query 2"
        assert tracker.query_results[-1]['query'] == "query 4"


class TestFederatedAggregator:
    """Test FederatedAggregator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aggregator = FederatedAggregator(
            num_users=5,
            aggregation_frequency=3,
            min_participants=2
        )
    
    def test_aggregator_initialization(self):
        """Test FederatedAggregator initialization."""
        assert self.aggregator.num_users == 5
        assert self.aggregator.aggregation_frequency == 3
        assert self.aggregator.min_participants == 2
        assert len(self.aggregator.pending_updates) == 0
        assert len(self.aggregator.update_history) == 0
        assert self.aggregator.global_threshold == 0.8
        assert self.aggregator.total_aggregations == 0
    
    def test_submit_update_below_frequency(self):
        """Test submitting updates below aggregation frequency."""
        update1 = ThresholdUpdate(
            user_id="user_001",
            old_threshold=0.8,
            new_threshold=0.75,
            performance_gain=0.05,
            timestamp=time.time(),
            confidence=1.0
        )
        
        self.aggregator.submit_update(update1)
        
        assert len(self.aggregator.pending_updates) == 1
        assert len(self.aggregator.update_history) == 1
        assert self.aggregator.total_aggregations == 0  # No aggregation yet
    
    def test_submit_update_triggers_aggregation(self):
        """Test submitting updates that trigger aggregation."""
        updates = []
        for i in range(3):  # Reach aggregation frequency
            update = ThresholdUpdate(
                user_id=f"user_{i:03d}",
                old_threshold=0.8,
                new_threshold=0.75 + i * 0.02,
                performance_gain=0.05,
                timestamp=time.time(),
                confidence=1.0
            )
            updates.append(update)
        
        # Submit updates
        for update in updates:
            self.aggregator.submit_update(update)
        
        # Should trigger aggregation
        assert self.aggregator.total_aggregations == 1
        assert len(self.aggregator.pending_updates) == 0  # Cleared after aggregation
        assert len(self.aggregator.update_history) == 3
        
        # Global threshold should be updated
        assert self.aggregator.global_threshold != 0.8
    
    def test_aggregation_with_insufficient_participants(self):
        """Test aggregation with insufficient participants."""
        # Set min_participants higher than updates
        self.aggregator.min_participants = 5
        
        update = ThresholdUpdate(
            user_id="user_001",
            old_threshold=0.8,
            new_threshold=0.75,
            performance_gain=0.05,
            timestamp=time.time(),
            confidence=1.0
        )
        
        # Manually add to pending updates and try aggregation
        self.aggregator.pending_updates.append(update)
        self.aggregator._aggregate_updates()
        
        # Should not aggregate
        assert self.aggregator.total_aggregations == 0
        assert len(self.aggregator.pending_updates) == 1
    
    def test_weighted_federated_averaging(self):
        """Test weighted federated averaging calculation."""
        # Create updates with different weights
        updates = [
            ThresholdUpdate("user_001", 0.8, 0.7, 0.1, time.time(), 1.0),   # weight: 0.1
            ThresholdUpdate("user_002", 0.8, 0.6, 0.2, time.time(), 0.5),   # weight: 0.1
            ThresholdUpdate("user_003", 0.8, 0.9, 0.05, time.time(), 1.0),  # weight: 0.05
        ]
        
        self.aggregator.pending_updates = updates
        initial_threshold = self.aggregator.global_threshold
        
        self.aggregator._aggregate_updates()
        
        # Should update threshold based on weighted average
        assert self.aggregator.global_threshold != initial_threshold
        assert 0.1 <= self.aggregator.global_threshold <= 0.95
        assert self.aggregator.total_aggregations == 1
    
    def test_get_global_threshold(self):
        """Test getting global threshold."""
        threshold = self.aggregator.get_global_threshold()
        assert threshold == 0.8  # Initial value
    
    def test_get_aggregation_stats(self):
        """Test getting aggregation statistics."""
        stats = self.aggregator.get_aggregation_stats()
        
        assert stats['global_threshold'] == 0.8
        assert stats['total_aggregations'] == 0
        assert stats['pending_updates'] == 0
        assert stats['total_updates'] == 0
        assert stats['num_users'] == 5
        assert stats['aggregation_frequency'] == 3
        assert 'last_aggregation_time' in stats
    
    def test_momentum_in_aggregation(self):
        """Test momentum application in threshold updates."""
        original_threshold = self.aggregator.global_threshold
        
        # Create update that would dramatically change threshold
        extreme_update = ThresholdUpdate(
            user_id="user_001",
            old_threshold=0.8,
            new_threshold=0.1,  # Very different
            performance_gain=1.0,
            timestamp=time.time(),
            confidence=1.0
        )
        
        self.aggregator.pending_updates = [extreme_update, extreme_update]
        self.aggregator._aggregate_updates()
        
        # Should be moderated by momentum (0.7)
        new_threshold = self.aggregator.global_threshold
        change = abs(new_threshold - original_threshold)
        extreme_change = abs(0.1 - original_threshold)
        
        assert change < extreme_change  # Momentum should moderate the change


class TestTauManager:
    """Test TauManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock config
        self.mock_config = Mock()
        self.mock_config.federated.num_users = 5
        self.mock_config.federated.aggregation_frequency = 10
        self.mock_config.federated.learning_rate = 0.01
        self.mock_config.federated.initial_tau = 0.8
        self.mock_config.federated.enabled = True
        
        with patch('src.core.tau_manager.get_config', return_value=self.mock_config):
            self.tau_manager = TauManager()
    
    def test_tau_manager_initialization(self):
        """Test TauManager initialization."""
        assert self.tau_manager.num_users == 5
        assert self.tau_manager.aggregation_frequency == 10
        assert self.tau_manager.learning_rate == 0.01
        assert self.tau_manager.initial_tau == 0.8
        assert self.tau_manager.enabled is True
        assert self.tau_manager.current_threshold == 0.8
        assert len(self.tau_manager.users) == 5
        assert len(self.tau_manager.performance_trackers) == 5
        assert self.tau_manager.query_count == 0
    
    def test_user_initialization_variation(self):
        """Test that users are initialized with varied thresholds."""
        thresholds = [user.current_threshold for user in self.tau_manager.users.values()]
        
        # Should have some variation around initial_tau
        assert len(set(thresholds)) > 1  # Not all the same
        
        # All should be within reasonable bounds
        for threshold in thresholds:
            assert 0.1 <= threshold <= 0.95
    
    def test_evaluate_threshold_disabled(self):
        """Test threshold evaluation when disabled."""
        self.tau_manager.enabled = False
        original_threshold = self.tau_manager.current_threshold
        
        result = self.tau_manager.evaluate_threshold(
            query="test query",
            similarity_score=0.9,
            cache_hit=True
        )
        
        assert result == original_threshold
        assert self.tau_manager.query_count == 0  # Should not increment
    
    def test_evaluate_threshold_enabled(self):
        """Test threshold evaluation when enabled."""
        original_count = self.tau_manager.query_count
        
        result = self.tau_manager.evaluate_threshold(
            query="test query",
            similarity_score=0.9,
            cache_hit=True
        )
        
        assert isinstance(result, float)
        assert 0.1 <= result <= 0.95
        assert self.tau_manager.query_count == original_count + 1
    
    def test_evaluate_threshold_with_ground_truth(self):
        """Test threshold evaluation with ground truth."""
        result = self.tau_manager.evaluate_threshold(
            query="test query",
            similarity_score=0.85,
            cache_hit=True,
            ground_truth_hit=True
        )
        
        assert isinstance(result, float)
        assert self.tau_manager.query_count == 1
    
    def test_user_threshold_update_trigger(self):
        """Test that user threshold updates are triggered periodically."""
        # Mock a user's performance tracker to have enough queries
        user_id = list(self.tau_manager.users.keys())[0]
        user_state = self.tau_manager.users[user_id]
        
        with patch.object(self.tau_manager, '_update_user_threshold') as mock_update:
            # Simulate 20 queries for this user to trigger update
            for i in range(20):
                with patch('random.choice', return_value=user_id):
                    self.tau_manager.evaluate_threshold(
                        query=f"query {i}",
                        similarity_score=0.8,
                        cache_hit=True
                    )
            
            # Should trigger update on the 20th query
            mock_update.assert_called_with(user_id)
    
    def test_user_threshold_update_with_improvement(self):
        """Test user threshold update with performance improvement."""
        user_id = list(self.tau_manager.users.keys())[0]
        user_state = self.tau_manager.users[user_id]
        
        # Add some performance history
        user_state.performance_history.append({'f1_score': 0.6})
        
        # Mock performance tracker to return improved performance
        tracker = self.tau_manager.performance_trackers[user_id]
        with patch.object(tracker, 'get_performance_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'f1_score': 0.7,  # Improvement from 0.6
                'precision': 0.8,
                'recall': 0.7,
                'total_samples': 50
            }
            
            with patch.object(self.tau_manager.aggregator, 'submit_update') as mock_submit:
                self.tau_manager._update_user_threshold(user_id)
                
                # Should submit update due to improvement
                mock_submit.assert_called_once()
                
                # Check the submitted update
                submitted_update = mock_submit.call_args[0][0]
                assert submitted_update.user_id == user_id
                assert submitted_update.performance_gain > 0
    
    def test_user_threshold_update_no_improvement(self):
        """Test user threshold update without significant improvement."""
        user_id = list(self.tau_manager.users.keys())[0]
        user_state = self.tau_manager.users[user_id]
        
        # Add performance history
        user_state.performance_history.append({'f1_score': 0.7})
        
        # Mock performance tracker to return no improvement
        tracker = self.tau_manager.performance_trackers[user_id]
        with patch.object(tracker, 'get_performance_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'f1_score': 0.69,  # Slight decrease
                'precision': 0.8,
                'recall': 0.7,
                'total_samples': 50
            }
            
            with patch.object(self.tau_manager.aggregator, 'submit_update') as mock_submit:
                self.tau_manager._update_user_threshold(user_id)
                
                # Should not submit update (no significant improvement)
                mock_submit.assert_not_called()
    
    def test_get_current_threshold(self):
        """Test getting current threshold."""
        threshold = self.tau_manager.get_current_threshold()
        assert threshold == self.tau_manager.current_threshold
        assert isinstance(threshold, float)
    
    def test_get_tau_stats(self):
        """Test getting comprehensive tau statistics."""
        # Add some query history
        for i in range(5):
            self.tau_manager.evaluate_threshold(
                query=f"query {i}",
                similarity_score=0.8,
                cache_hit=True
            )
        
        stats = self.tau_manager.get_tau_stats()
        
        assert stats['enabled'] is True
        assert stats['current_threshold'] == self.tau_manager.current_threshold
        assert stats['total_queries'] == 5
        assert stats['num_users'] == 5
        assert 'user_statistics' in stats
        assert 'aggregator_statistics' in stats
        
        # Check user statistics
        user_stats = stats['user_statistics']
        assert len(user_stats) == 5
        
        for user_id, user_stat in user_stats.items():
            assert 'current_threshold' in user_stat
            assert 'total_queries' in user_stat
            assert 'performance' in user_stat
            assert 'last_update_time' in user_stat
    
    def test_force_aggregation(self):
        """Test forcing federated aggregation."""
        with patch.object(self.tau_manager.aggregator, '_aggregate_updates') as mock_aggregate:
            self.tau_manager.force_aggregation()
            mock_aggregate.assert_called_once()
    
    def test_reset_user_performance_single_user(self):
        """Test resetting performance for a single user."""
        user_id = list(self.tau_manager.users.keys())[0]
        user_state = self.tau_manager.users[user_id]
        
        # Add some data
        user_state.performance_history.append({'f1_score': 0.8})
        user_state.confusion_matrix.true_positives = 10
        
        self.tau_manager.reset_user_performance(user_id)
        
        assert len(user_state.performance_history) == 0
        assert user_state.confusion_matrix.true_positives == 0
    
    def test_reset_user_performance_all_users(self):
        """Test resetting performance for all users."""
        # Add data to all users
        for user_state in self.tau_manager.users.values():
            user_state.performance_history.append({'f1_score': 0.8})
            user_state.confusion_matrix.true_positives = 10
        
        self.tau_manager.reset_user_performance()
        
        # All users should be reset
        for user_state in self.tau_manager.users.values():
            assert len(user_state.performance_history) == 0
            assert user_state.confusion_matrix.true_positives == 0
    
    def test_threshold_update_with_global_aggregation(self):
        """Test that global threshold is used when available."""
        # Set a different global threshold
        self.tau_manager.aggregator.global_threshold = 0.65
        
        result = self.tau_manager.evaluate_threshold(
            query="test query",
            similarity_score=0.8,
            cache_hit=True
        )
        
        # Should use global threshold
        assert result == 0.65
        assert self.tau_manager.current_threshold == 0.65


class TestConvenienceFunction:
    """Test convenience function for creating tau manager."""
    
    def test_create_tau_manager(self):
        """Test creating tau manager using convenience function."""
        with patch('src.core.tau_manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.federated.num_users = 3
            mock_config.federated.aggregation_frequency = 5
            mock_config.federated.learning_rate = 0.02
            mock_config.federated.initial_tau = 0.75
            mock_config.federated.enabled = True
            mock_get_config.return_value = mock_config
            
            tau_manager = create_tau_manager(
                num_users=10,  # Override config
                learning_rate=0.05
            )
        
        assert isinstance(tau_manager, TauManager)
        assert tau_manager.num_users == 10  # Override should work
        assert tau_manager.learning_rate == 0.05


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def test_full_tau_workflow(self):
        """Test complete tau workflow from queries to aggregation."""
        mock_config = Mock()
        mock_config.federated.num_users = 3
        mock_config.federated.aggregation_frequency = 5
        mock_config.federated.learning_rate = 0.1
        mock_config.federated.initial_tau = 0.8
        mock_config.federated.enabled = True
        
        with patch('src.core.tau_manager.get_config', return_value=mock_config):
            tau_manager = TauManager()
        
        initial_threshold = tau_manager.current_threshold
        
        # Simulate many queries to trigger updates and aggregation
        for i in range(100):
            tau_manager.evaluate_threshold(
                query=f"query {i}",
                similarity_score=0.7 + (i % 10) * 0.02,  # Varied scores
                cache_hit=i % 3 == 0,  # Varied cache hits
                ground_truth_hit=i % 2 == 0  # Varied ground truth
            )
        
        # Check that system has adapted
        final_threshold = tau_manager.current_threshold
        
        # Threshold should have been updated (though direction depends on performance)
        stats = tau_manager.get_tau_stats()
        assert stats['total_queries'] == 100
        
        # Users should have performance history
        for user_stats in stats['user_statistics'].values():
            if user_stats['total_queries'] > 0:
                assert 'performance' in user_stats
    
    def test_threading_safety(self):
        """Test thread safety of tau manager."""
        mock_config = Mock()
        mock_config.federated.num_users = 3
        mock_config.federated.aggregation_frequency = 10
        mock_config.federated.learning_rate = 0.01
        mock_config.federated.initial_tau = 0.8
        mock_config.federated.enabled = True
        
        with patch('src.core.tau_manager.get_config', return_value=mock_config):
            tau_manager = TauManager()
        
        results = []
        errors = []
        
        def worker():
            try:
                for i in range(20):
                    threshold = tau_manager.evaluate_threshold(
                        query=f"query {i}",
                        similarity_score=0.8,
                        cache_hit=True
                    )
                    results.append(threshold)
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
        assert len(results) == 60  # 3 threads * 20 calls each
        assert len(errors) == 0
        
        # All thresholds should be valid
        for threshold in results:
            assert 0.1 <= threshold <= 0.95
    
    def test_performance_degradation_handling(self):
        """Test handling of performance degradation."""
        mock_config = Mock()
        mock_config.federated.num_users = 2
        mock_config.federated.aggregation_frequency = 20
        mock_config.federated.learning_rate = 0.1
        mock_config.federated.initial_tau = 0.8
        mock_config.federated.enabled = True
        
        with patch('src.core.tau_manager.get_config', return_value=mock_config):
            tau_manager = TauManager()
        
        # Simulate consistently poor performance
        for i in range(50):
            tau_manager.evaluate_threshold(
                query=f"query {i}",
                similarity_score=0.9,  # High similarity
                cache_hit=False,       # But no cache hits
                ground_truth_hit=True  # Ground truth says should hit
            )
        
        # System should attempt to adapt thresholds
        stats = tau_manager.get_tau_stats()
        
        # Verify that performance tracking is working
        assert stats['total_queries'] == 50
        
        # Users should have recorded the poor performance
        for user_stats in stats['user_statistics'].values():
            if user_stats['total_queries'] > 0:
                performance = user_stats['performance']
                # Poor performance should be reflected in metrics
                assert 'precision' in performance
                assert 'recall' in performance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
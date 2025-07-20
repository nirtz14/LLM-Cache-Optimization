"""Tests for context-chain filtering functionality."""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.core.context_similarity import (
    ContextAwareSimilarity, 
    ConversationContext, 
    ContextTracker,
    ConversationTurn
)

class TestConversationContext:
    """Test ConversationContext functionality."""
    
    def test_conversation_context_creation(self):
        """Test creating a conversation context."""
        context = ConversationContext(conversation_id="test_conv")
        assert context.conversation_id == "test_conv"
        assert len(context.turns) == 0
        assert context.context_embedding is None
    
    def test_add_turn(self):
        """Test adding turns to conversation."""
        context = ConversationContext(conversation_id="test_conv")
        context.add_turn("Hello", "Hi there", timestamp=123.0)
        
        assert len(context.turns) == 1
        turn = context.turns[0]
        assert turn.query == "Hello"
        assert turn.response == "Hi there"
        assert turn.timestamp == 123.0
        assert context.context_embedding is None  # Should be invalidated
    
    def test_get_recent_context(self):
        """Test getting recent context within window."""
        context = ConversationContext(conversation_id="test_conv")
        
        # Add multiple turns
        for i in range(10):
            context.add_turn(f"Query {i}", f"Response {i}")
        
        # Get recent context with window size 3
        recent = context.get_recent_context(window_size=3)
        assert len(recent) == 3
        assert recent[0].query == "Query 7"  # Last 3 turns
        assert recent[1].query == "Query 8"
        assert recent[2].query == "Query 9"
    
    def test_to_context_string(self):
        """Test converting context to string representation."""
        context = ConversationContext(conversation_id="test_conv")
        context.add_turn("Hello", "Hi there")
        context.add_turn("How are you?", "I'm good")
        
        context_str = context.to_context_string(window_size=5)
        
        assert "Turn 1: Hello" in context_str
        assert "Response 1: Hi there" in context_str
        assert "Turn 2: How are you?" in context_str
        assert "Response 2: I'm good" in context_str

class TestContextTracker:
    """Test ContextTracker functionality."""
    
    def test_context_tracker_creation(self):
        """Test creating a context tracker."""
        tracker = ContextTracker(max_conversations=10)
        assert len(tracker.conversations) == 0
        assert tracker.max_conversations == 10
    
    def test_get_or_create_context(self):
        """Test getting or creating conversation contexts."""
        tracker = ContextTracker()
        
        # Create new context
        context1 = tracker.get_or_create_context("conv1")
        assert context1.conversation_id == "conv1"
        assert len(tracker.conversations) == 1
        
        # Get existing context
        context2 = tracker.get_or_create_context("conv1")
        assert context2 is context1  # Should be same object
        assert len(tracker.conversations) == 1
    
    def test_max_conversations_limit(self):
        """Test that oldest conversations are removed when limit exceeded."""
        tracker = ContextTracker(max_conversations=2)
        
        # Create contexts with different timestamps
        context1 = tracker.get_or_create_context("conv1")
        context1.last_updated = 1.0
        
        context2 = tracker.get_or_create_context("conv2")  
        context2.last_updated = 2.0
        
        # This should remove conv1 (oldest)
        context3 = tracker.get_or_create_context("conv3")
        context3.last_updated = 3.0
        
        assert len(tracker.conversations) == 2
        assert "conv1" not in tracker.conversations
        assert "conv2" in tracker.conversations
        assert "conv3" in tracker.conversations
    
    def test_add_turn(self):
        """Test adding turns through tracker."""
        tracker = ContextTracker()
        tracker.add_turn("conv1", "Hello", "Hi", timestamp=123.0)
        
        context = tracker.conversations["conv1"]
        assert len(context.turns) == 1
        assert context.turns[0].query == "Hello"
        assert context.last_updated == 123.0

class TestContextAwareSimilarity:
    """Test ContextAwareSimilarity functionality."""
    
    @patch('src.core.context_similarity.SentenceTransformer')
    def test_context_aware_similarity_init(self, mock_transformer):
        """Test initialization of context-aware similarity."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        similarity = ContextAwareSimilarity(
            embedding_model="test-model",
            context_window_size=3,
            divergence_threshold=0.5
        )
        
        assert similarity.embedding_model_name == "test-model"
        assert similarity.context_window_size == 3
        assert similarity.divergence_threshold == 0.5
        assert similarity.embedding_model == mock_model
        mock_transformer.assert_called_once_with("test-model")
    
    @patch('src.core.context_similarity.SentenceTransformer')
    def test_same_conversation_context_similarity(self, mock_transformer):
        """Test that same conversation ID returns similarity 1.0."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        similarity = ContextAwareSimilarity()
        
        # Mock the context similarity computation to return 1.0 for same conversation
        result = similarity._compute_context_similarity(
            "conv1", "conv1", "query1", "query2"
        )
        
        assert result == 1.0
    
    @patch('src.core.context_similarity.SentenceTransformer')
    def test_evaluation_with_context_filtering(self, mock_transformer):
        """Test evaluation with context filtering."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_transformer.return_value = mock_model
        
        similarity = ContextAwareSimilarity(divergence_threshold=0.5)
        
        # Mock base similarity to return high score
        similarity.base_similarity_func = Mock()
        similarity.base_similarity_func.evaluation.return_value = 0.9
        
        src_dict = {
            'query': 'test query',
            'conversation_id': 'conv1',
            'embedding': np.array([1, 0, 0])
        }
        
        cache_dict = {
            'query': 'cached query', 
            'conversation_id': 'conv1',  # Same conversation
            'embedding': np.array([0, 1, 0])
        }
        
        # Should use base similarity since same conversation
        result = similarity.evaluation(src_dict, cache_dict)
        assert result == 0.9
        similarity.base_similarity_func.evaluation.assert_called_once()
    
    @patch('src.core.context_similarity.SentenceTransformer')
    def test_evaluation_without_context_info(self, mock_transformer):
        """Test evaluation falls back to base similarity without context info."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        similarity = ContextAwareSimilarity()
        similarity.base_similarity_func = Mock()
        similarity.base_similarity_func.evaluation.return_value = 0.7
        
        src_dict = {'query': 'test query'}  # No conversation_id
        cache_dict = {'query': 'cached query'}
        
        result = similarity.evaluation(src_dict, cache_dict)
        assert result == 0.7
        similarity.base_similarity_func.evaluation.assert_called_once()
    
    @patch('src.core.context_similarity.SentenceTransformer')
    def test_cosine_similarity(self, mock_transformer):
        """Test cosine similarity computation."""
        mock_transformer.return_value = Mock()
        similarity = ContextAwareSimilarity()
        
        # Test identical vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        result = similarity._cosine_similarity(vec1, vec2)
        assert result == 1.0
        
        # Test orthogonal vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        result = similarity._cosine_similarity(vec1, vec2)
        assert result == 0.0
        
        # Test zero vector
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([1, 0, 0])
        result = similarity._cosine_similarity(vec1, vec2)
        assert result == 0.0
    
    @patch('src.core.context_similarity.SentenceTransformer')
    def test_add_conversation_turn(self, mock_transformer):
        """Test adding conversation turns."""
        mock_transformer.return_value = Mock()
        similarity = ContextAwareSimilarity()
        
        similarity.add_conversation_turn("conv1", "Hello", "Hi", 123.0)
        
        # Check that turn was added to context tracker
        context = similarity.context_tracker.get_or_create_context("conv1")
        assert len(context.turns) == 1
        assert context.turns[0].query == "Hello"
        assert context.turns[0].response == "Hi"
    
    @patch('src.core.context_similarity.SentenceTransformer')
    def test_get_context_stats(self, mock_transformer):
        """Test getting context statistics."""
        mock_transformer.return_value = Mock()
        similarity = ContextAwareSimilarity()
        
        # Add some conversation data
        similarity.add_conversation_turn("conv1", "Hello", "Hi")
        similarity.add_conversation_turn("conv2", "Hey", "Hello")
        
        stats = similarity.get_context_stats()
        
        assert stats['total_conversations'] == 2
        assert stats['total_turns'] == 2
        assert stats['avg_turns_per_conversation'] == 1.0
        assert stats['context_window_size'] == similarity.context_window_size
        assert stats['divergence_threshold'] == similarity.divergence_threshold
        assert 'enabled' in stats

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.context.enabled = True
    config.context.window_size = 5
    config.context.divergence_threshold = 0.3
    config.context.embedding_model = "test-model"
    return config

def test_context_embedding_caching():
    """Test that context embeddings are cached properly."""
    with patch('src.core.context_similarity.SentenceTransformer') as mock_transformer:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_transformer.return_value = mock_model
        
        similarity = ContextAwareSimilarity()
        
        # First call should encode
        embedding1 = similarity._get_context_embedding("test context")
        assert mock_model.encode.call_count == 1
        
        # Second call with same context should use cache
        embedding2 = similarity._get_context_embedding("test context")
        assert mock_model.encode.call_count == 1  # No additional call
        
        # Results should be identical
        np.testing.assert_array_equal(embedding1, embedding2)

def test_context_disabled():
    """Test behavior when context filtering is disabled."""
    with patch('src.core.context_similarity.get_config') as mock_get_config:
        mock_config = Mock()
        mock_config.context.enabled = False
        mock_get_config.return_value = mock_config
        
        with patch('src.core.context_similarity.SentenceTransformer'):
            similarity = ContextAwareSimilarity()
            similarity.base_similarity_func = Mock()
            similarity.base_similarity_func.evaluation.return_value = 0.8
            
            src_dict = {'query': 'test'}
            cache_dict = {'query': 'cached'}
            
            result = similarity.evaluation(src_dict, cache_dict)
            assert result == 0.8
            similarity.base_similarity_func.evaluation.assert_called_once()

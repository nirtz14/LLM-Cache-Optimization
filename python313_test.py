#!/usr/bin/env python3
"""
Simple cache that works without FAISS (using sklearn for similarity)
Good fallback for Python 3.13 compatibility issues
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class SimpleSimilarityCache:
    """Cache using sklearn for similarity search instead of FAISS."""
    
    def __init__(self, similarity_threshold=0.8):
        self.cache = {}  # query_id -> {'query': str, 'response': str, 'embedding': np.array}
        self.embeddings = []  # List of embeddings for batch similarity
        self.query_ids = []   # Corresponding query IDs
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
        
    def _init_embedding_model(self):
        """Initialize embedding model lazily."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Sentence transformer model loaded")
            except Exception as e:
                print(f"❌ Failed to load sentence transformer: {e}")
                raise
    
    def _get_embedding(self, text):
        """Get embedding for text."""
        self._init_embedding_model()
        return self.embedding_model.encode(text)
    
    def set(self, query, response):
        """Store query-response pair."""
        try:
            embedding = self._get_embedding(query)
            query_id = f"q_{len(self.cache)}"
            
            self.cache[query_id] = {
                'query': query,
                'response': response,
                'embedding': embedding
            }
            
            self.embeddings.append(embedding)
            self.query_ids.append(query_id)
            
            return True
            
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def get(self, query):
        """Get response for query if similar enough."""
        try:
            if not self.cache:
                return None, 0.0
            
            query_embedding = self._get_embedding(query)
            
            # Use sklearn for batch similarity computation
            if len(self.embeddings) > 0:
                embeddings_matrix = np.array(self.embeddings)
                similarities = cosine_similarity(
                    query_embedding.reshape(1, -1), 
                    embeddings_matrix
                )[0]
                
                # Find best match above threshold
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                if best_score >= self.similarity_threshold:
                    best_query_id = self.query_ids[best_idx]
                    response = self.cache[best_query_id]['response']
                    return response, best_score
            
            return None, 0.0
            
        except Exception as e:
            print(f"Cache get error: {e}")
            return None, 0.0
    
    def save(self, filepath):
        """Save cache to file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'embeddings': self.embeddings,
                    'query_ids': self.query_ids,
                    'similarity_threshold': self.similarity_threshold
                }, f)
            return True
        except Exception as e:
            print(f"Cache save error: {e}")
            return False
    
    def load(self, filepath):
        """Load cache from file."""
        try:
            if not os.path.exists(filepath):
                return False
                
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.cache = data['cache']
            self.embeddings = data['embeddings']
            self.query_ids = data['query_ids']
            self.similarity_threshold = data.get('similarity_threshold', 0.8)
            
            return True
        except Exception as e:
            print(f"Cache load error: {e}")
            return False
    
    def stats(self):
        """Get cache statistics."""
        return {
            'total_entries': len(self.cache),
            'similarity_threshold': self.similarity_threshold,
            'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0
        }

# Test the cache
def test_simple_cache():
    """Test the simple cache implementation."""
    print("🧪 Testing Simple Cache (no FAISS required)")
    
    try:
        cache = SimpleSimilarityCache(similarity_threshold=0.8)
        
        # Test set operations
        test_data = [
            ("What is Python?", "Python is a programming language."),
            ("How do I install packages?", "Use pip install package_name."),
            ("What is machine learning?", "ML is a branch of AI."),
        ]
        
        for query, response in test_data:
            success = cache.set(query, response)
            if success:
                print(f"  ✅ Stored: {query[:30]}...")
            else:
                print(f"  ❌ Failed to store: {query[:30]}...")
                return False
        
        # Test exact retrieval
        result, score = cache.get("What is Python?")
        if result and score > 0.9:
            print(f"  ✅ Exact match works (score: {score:.3f})")
        else:
            print(f"  ❌ Exact match failed (score: {score:.3f})")
            return False
        
        # Test similar query
        result, score = cache.get("Tell me about Python")
        if score > 0.5:  # Should be somewhat similar
            print(f"  ✅ Similarity search works (score: {score:.3f})")
        else:
            print(f"  ⚠️  Similarity search low score: {score:.3f}")
        
        # Test cache persistence
        cache.save("test_cache.pkl")
        new_cache = SimpleSimilarityCache()
        if new_cache.load("test_cache.pkl"):
            print(f"  ✅ Cache persistence works")
            os.remove("test_cache.pkl")  # Cleanup
        
        print(f"  📊 Cache stats: {cache.stats()}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_cache()
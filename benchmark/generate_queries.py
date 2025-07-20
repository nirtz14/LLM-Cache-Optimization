"""Generate synthetic query datasets for benchmarking Enhanced GPTCache."""
import json
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid

@dataclass
class QueryItem:
    """Represents a single query item in the dataset."""
    id: str
    query: str
    conversation_id: str
    category: str
    expected_response: Optional[str] = None
    context_history: List[str] = None
    similarity_group: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context_history is None:
            self.context_history = []
        if self.metadata is None:
            self.metadata = {}

class QueryGenerator:
    """Generates synthetic queries for different cache testing scenarios."""
    
    def __init__(self, seed: int = 42):
        """Initialize query generator with random seed for reproducibility."""
        random.seed(seed)
        self.query_id_counter = 0
        
        # Templates for different query categories
        self.repetitive_templates = [
            "What is the weather today?",
            "How do I reset my password?",
            "What are your business hours?",
            "Can you help me with my order?",
            "Where is your nearest location?",
            "What payment methods do you accept?",
            "How do I contact customer service?",
            "What is your return policy?",
            "Do you offer free shipping?",
            "How do I track my order?",
        ]
        
        self.conversational_templates = [
            ("Tell me about artificial intelligence", "AI is a field of computer science..."),
            ("What are the main types of machine learning?", "The main types are supervised, unsupervised, and reinforcement learning..."),
            ("How does deep learning work?", "Deep learning uses neural networks with multiple layers..."),
            ("What is natural language processing?", "NLP is a branch of AI that helps computers understand human language..."),
            ("Explain computer vision", "Computer vision enables machines to interpret visual information..."),
        ]
        
        self.programming_templates = [
            ("How do I create a Python list?", "You can create a list using square brackets: my_list = []"),
            ("What is a function in programming?", "A function is a reusable block of code that performs a specific task"),
            ("How do I handle errors in Python?", "Use try-except blocks to handle exceptions"),
            ("What is object-oriented programming?", "OOP is a programming paradigm based on objects and classes"),
            ("How do I read a file in Python?", "Use the open() function with a file path and mode"),
        ]
        
        self.contextual_conversations = [
            [
                "I need help with my computer",
                "What seems to be the problem?",
                "It's running very slowly",
                "Let's try restarting it first",
                "That didn't work, what else can I do?",
                "Try checking for malware or clearing temporary files"
            ],
            [
                "I want to learn programming",
                "What programming language interests you?",
                "I'm thinking about Python",
                "Python is great for beginners. Would you like some resources?",
                "Yes, please recommend some tutorials",
                "I recommend starting with Python.org's official tutorial"
            ],
            [
                "I'm planning a trip to Europe",
                "Which countries are you considering?",
                "France, Italy, and Spain",
                "How long is your trip?",
                "About two weeks",
                "That's perfect for visiting all three countries"
            ]
        ]
    
    def _get_next_id(self) -> str:
        """Generate next query ID."""
        self.query_id_counter += 1
        return f"query_{self.query_id_counter:06d}"
    
    def generate_repetitive_queries(self, count: int) -> List[QueryItem]:
        """Generate repetitive short queries that should have high cache hit rates."""
        queries = []
        
        for i in range(count):
            # Pick a template and slightly modify it
            base_template = random.choice(self.repetitive_templates)
            
            # Add slight variations
            variations = [
                base_template,
                base_template + "?",
                base_template.replace("?", " please?"),
                "Can you tell me " + base_template.lower(),
                base_template + " Thanks!",
            ]
            
            query_text = random.choice(variations)
            
            query = QueryItem(
                id=self._get_next_id(),
                query=query_text,
                conversation_id=f"repetitive_conv_{i % 20}",  # Group into conversations
                category="repetitive",
                similarity_group=base_template,  # Group similar queries
                metadata={"template": base_template, "variation": len([q for q in queries if q.similarity_group == base_template])}
            )
            
            queries.append(query)
        
        return queries
    
    def generate_novel_queries(self, count: int) -> List[QueryItem]:
        """Generate novel long queries that are unlikely to be cached."""
        queries = []
        
        # Complex, unique query patterns
        complex_templates = [
            "I have a very specific problem with {topic} involving {detail1} and {detail2}, can you help me understand {aspect}?",
            "My {item} broke and I need to {action} it but I'm not sure about {concern} - what should I do?",
            "I'm trying to {goal} but every time I {attempt}, I get {error}. Any suggestions for {context}?",
            "Can you explain the relationship between {concept1} and {concept2} specifically in the context of {scenario}?",
            "I have {constraint} and need to {objective} while considering {limitation} - is this possible?",
        ]
        
        # Fill-in options for templates
        topics = ["software development", "data analysis", "machine learning", "web design", "database management"]
        details = ["performance issues", "memory constraints", "security requirements", "scalability concerns", "user experience"]
        aspects = ["implementation details", "best practices", "common pitfalls", "optimization strategies", "alternative approaches"]
        items = ["laptop", "application", "database", "network", "server"]
        actions = ["repair", "optimize", "configure", "troubleshoot", "upgrade"]
        concerns = ["data loss", "downtime", "compatibility", "cost", "complexity"]
        
        for i in range(count):
            template = random.choice(complex_templates)
            
            # Fill template with random values
            filled_query = template.format(
                topic=random.choice(topics),
                detail1=random.choice(details),
                detail2=random.choice(details),
                aspect=random.choice(aspects),
                item=random.choice(items),
                action=random.choice(actions),
                concern=random.choice(concerns),
                goal=f"accomplish {random.choice(aspects)}",
                attempt=f"try to {random.choice(actions)}",
                error=f"encounter {random.choice(concerns)}",
                context=random.choice(topics),
                concept1=random.choice(topics),
                concept2=random.choice(aspects),
                scenario=random.choice(details),
                constraint=f"limited {random.choice(['time', 'budget', 'resources'])}",
                objective=f"achieve {random.choice(aspects)}",
                limitation=random.choice(concerns)
            )
            
            query = QueryItem(
                id=self._get_next_id(),
                query=filled_query,
                conversation_id=f"novel_conv_{uuid.uuid4().hex[:8]}",  # Unique conversation
                category="novel",
                metadata={"template": template, "length": len(filled_query)}
            )
            
            queries.append(query)
        
        return queries
    
    def generate_contextual_conversations(self, num_conversations: int) -> List[QueryItem]:
        """Generate conversational queries with context dependencies."""
        queries = []
        
        # Extend conversations to ensure we have enough variety
        extended_conversations = self.contextual_conversations.copy()
        
        # Add more conversation templates to increase variety
        additional_conversations = [
            [
                "I'm having trouble with my website",
                "What kind of issues are you experiencing?",
                "The pages are loading very slowly and sometimes not at all"
            ],
            [
                "Can you help me choose a programming language?",
                "What do you want to build with it?",
                "I want to create web applications and maybe some data analysis"
            ],
            [
                "I need advice on buying a laptop",
                "What will you primarily use it for?",
                "Mostly programming work and some light gaming"
            ],
            [
                "How do I improve my presentation skills?",
                "What aspect would you like to focus on?",
                "I get really nervous and forget what to say"
            ],
            [
                "I want to start exercising but don't know where to begin",
                "What are your fitness goals?",
                "I just want to be healthier and have more energy"
            ]
        ]
        extended_conversations.extend(additional_conversations)
        
        for conv_idx in range(num_conversations):
            conversation = random.choice(extended_conversations)
            conversation_id = f"contextual_conv_{conv_idx:03d}"
            
            # Ensure each conversation has exactly 3 turns
            if len(conversation) > 3:
                conversation = conversation[:3]
            elif len(conversation) < 3:
                # Extend conversation to 3 turns if needed
                while len(conversation) < 3:
                    if len(conversation) == 1:
                        conversation.append("Can you tell me more about that?")
                    else:
                        conversation.append("That's interesting, what else should I know?")
            
            context_history = []
            
            for turn_idx, turn in enumerate(conversation):
                query = QueryItem(
                    id=self._get_next_id(),
                    query=turn,
                    conversation_id=conversation_id,
                    category="contextual",
                    context_history=context_history.copy(),
                    metadata={
                        "turn_number": turn_idx + 1,
                        "total_turns": 3,  # Always 3 turns as specified
                        "conversation_topic": f"topic_{conv_idx % len(extended_conversations)}"
                    }
                )
                
                queries.append(query)
                context_history.append(turn)
        
        return queries
    
    def generate_similar_query_groups(self, num_groups: int, queries_per_group: int) -> List[QueryItem]:
        """Generate groups of similar queries for testing similarity thresholds."""
        queries = []
        
        base_queries = [
            "How to install Python on Windows",
            "What is machine learning",
            "Best practices for database design",
            "How to optimize web performance",
            "Understanding neural networks",
        ]
        
        for group_idx in range(num_groups):
            base_query = base_queries[group_idx % len(base_queries)]
            group_id = f"similar_group_{group_idx:03d}"
            conversation_id = f"similar_conv_{group_idx:03d}"
            
            # Create variations of the base query
            variations = [
                base_query,
                f"Can you explain {base_query.lower()}",
                f"{base_query}?",
                f"I need help with {base_query.lower()}",
                f"Tell me about {base_query.lower()}",
                f"What's the best way to {base_query.lower()}",
                f"Please help me understand {base_query.lower()}",
            ]
            
            for i in range(queries_per_group):
                if i < len(variations):
                    query_text = variations[i]
                else:
                    # Generate more variations
                    prefixes = ["Could you", "Would you", "Can you please", "I'm wondering about"]
                    suffixes = ["please", "thanks", "if possible", "step by step"]
                    
                    query_text = f"{random.choice(prefixes)} {base_query.lower()} {random.choice(suffixes)}"
                
                query = QueryItem(
                    id=self._get_next_id(),
                    query=query_text,
                    conversation_id=conversation_id,
                    category="similar",
                    similarity_group=group_id,
                    expected_response=f"Response for {base_query}",
                    metadata={
                        "base_query": base_query,
                        "variation_index": i,
                        "group_size": queries_per_group
                    }
                )
                
                queries.append(query)
        
        return queries
    
    def generate_mixed_dataset(
        self,
        total_count: int,
        repetitive_ratio: float = 0.3,
        novel_ratio: float = 0.3,
        contextual_ratio: float = 0.2,
        similar_ratio: float = 0.2
    ) -> List[QueryItem]:
        """Generate a mixed dataset with different query types.
        
        Args:
            total_count: Total number of queries to generate
            repetitive_ratio: Fraction of repetitive queries
            novel_ratio: Fraction of novel queries  
            contextual_ratio: Fraction of contextual queries
            similar_ratio: Fraction of similar query groups
            
        Returns:
            List of QueryItem objects
        """
        # Calculate counts for each category
        repetitive_count = int(total_count * repetitive_ratio)
        novel_count = int(total_count * novel_ratio)
        contextual_conversations = max(1, int(total_count * contextual_ratio) // 6)  # ~6 queries per conversation
        similar_groups = max(1, int(total_count * similar_ratio) // 5)  # 5 queries per group
        
        queries = []
        
        # Generate each category
        queries.extend(self.generate_repetitive_queries(repetitive_count))
        queries.extend(self.generate_novel_queries(novel_count))
        queries.extend(self.generate_contextual_conversations(contextual_conversations))
        queries.extend(self.generate_similar_query_groups(similar_groups, 5))
        
        # Shuffle to mix categories
        random.shuffle(queries)
        
        # Trim to exact count if needed
        return queries[:total_count]

def generate_query_dataset(
    output_path: str,
    count: int = 1000,
    dataset_type: str = "mixed",
    seed: int = 42,
    **kwargs
) -> List[QueryItem]:
    """Generate and save a query dataset.
    
    Args:
        output_path: Path to save the dataset JSON file
        count: Number of queries to generate
        dataset_type: Type of dataset ('mixed', 'repetitive', 'novel', 'contextual', 'similar')
        seed: Random seed for reproducibility
        **kwargs: Additional arguments for specific dataset types
        
    Returns:
        List of generated QueryItem objects
    """
    generator = QueryGenerator(seed=seed)
    
    if dataset_type == "mixed":
        queries = generator.generate_mixed_dataset(count, **kwargs)
    elif dataset_type == "repetitive":
        queries = generator.generate_repetitive_queries(count)
    elif dataset_type == "novel":
        queries = generator.generate_novel_queries(count)
    elif dataset_type == "contextual":
        num_conversations = kwargs.get('num_conversations', count // 6)
        queries = generator.generate_contextual_conversations(num_conversations)
    elif dataset_type == "similar":
        num_groups = kwargs.get('num_groups', count // 5)
        queries_per_group = kwargs.get('queries_per_group', 5)
        queries = generator.generate_similar_query_groups(num_groups, queries_per_group)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Convert to serializable format
    dataset = {
        'metadata': {
            'total_queries': len(queries),
            'dataset_type': dataset_type,
            'seed': seed,
            'generation_parameters': kwargs,
        },
        'queries': [asdict(query) for query in queries]
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(queries)} queries and saved to {output_path}")
    
    # Print category statistics
    categories = {}
    for query in queries:
        cat = query.category
        categories[cat] = categories.get(cat, 0) + 1
    
    print("Category distribution:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} ({count/len(queries)*100:.1f}%)")
    
    return queries

def main():
    """Command-line interface for query generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic query datasets")
    parser.add_argument("--output", "--out", "-o", default="data/prompts_large.json", 
                       help="Output JSON file path (default: data/prompts_large.json)")
    parser.add_argument("--count", "-c", type=int, default=1000, help="Number of queries to generate")
    parser.add_argument("--type", "-t", default="mixed", 
                       choices=["mixed", "repetitive", "novel", "contextual", "similar"],
                       help="Type of dataset to generate")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    
    # Specific count arguments for mixed dataset (400/300/300 default split)
    parser.add_argument("--n-repeat", type=int, default=400, help="Number of repetitive queries")
    parser.add_argument("--n-context", type=int, default=300, help="Number of contextual queries") 
    parser.add_argument("--n-novel", type=int, default=300, help="Number of novel queries")
    
    # Legacy ratio arguments (kept for backwards compatibility)
    parser.add_argument("--repetitive-ratio", type=float, help="Fraction of repetitive queries (legacy)")
    parser.add_argument("--novel-ratio", type=float, help="Fraction of novel queries (legacy)")
    parser.add_argument("--contextual-ratio", type=float, help="Fraction of contextual queries (legacy)")
    parser.add_argument("--similar-ratio", type=float, help="Fraction of similar queries (legacy)")
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.type == "mixed":
        # Use specific counts if provided, otherwise fall back to ratios or defaults
        if any([args.repetitive_ratio, args.novel_ratio, args.contextual_ratio, args.similar_ratio]):
            # Use legacy ratio-based approach
            kwargs.update({
                'repetitive_ratio': args.repetitive_ratio or 0.3,
                'novel_ratio': args.novel_ratio or 0.3,
                'contextual_ratio': args.contextual_ratio or 0.2,
                'similar_ratio': args.similar_ratio or 0.2,
            })
        else:
            # Use new count-based approach
            total_specified = args.n_repeat + args.n_context + args.n_novel
            if args.count != 1000 and total_specified == 1000:
                # User specified --count but used default n-* values, scale proportionally
                scale_factor = args.count / 1000
                repetitive_count = int(args.n_repeat * scale_factor)
                novel_count = int(args.n_novel * scale_factor)
                contextual_count = int(args.n_context * scale_factor)
            else:
                # Use exact counts specified
                repetitive_count = args.n_repeat
                novel_count = args.n_novel
                contextual_count = args.n_context
                args.count = repetitive_count + novel_count + contextual_count + 100  # Add some similar queries
            
            # Convert to ratios for the existing mixed dataset function
            total = repetitive_count + novel_count + contextual_count
            kwargs.update({
                'repetitive_ratio': repetitive_count / args.count,
                'novel_ratio': novel_count / args.count,
                'contextual_ratio': contextual_count / args.count,
                'similar_ratio': max(0.1, 1.0 - (repetitive_count + novel_count + contextual_count) / args.count),
            })
    
    generate_query_dataset(
        output_path=args.output,
        count=args.count,
        dataset_type=args.type,
        seed=args.seed,
        **kwargs
    )

if __name__ == "__main__":
    main()

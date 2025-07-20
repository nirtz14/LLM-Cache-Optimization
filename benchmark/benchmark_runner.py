"""Benchmark runner for Enhanced GPTCache with different feature combinations."""
import json
import time
import argparse
import os
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import pandas as pd
from tqdm import tqdm

from src.cache.enhanced_cache import create_enhanced_cache
from src.utils.config import init_config
from src.utils.metrics import get_performance_tracker, BenchmarkTimer
from .generate_queries import QueryItem

class BenchmarkRunner:
    """Runs benchmarks comparing different Enhanced GPTCache configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize benchmark runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = init_config(config_path)
        self.results = []
        
    def create_cache_variant(self, variant: str):
        """Create cache instance for specific variant.
        
        Args:
            variant: Cache variant ('baseline', 'context', 'pca', 'tau', 'full')
            
        Returns:
            Enhanced cache instance
        """
        if variant == "baseline":
            # Standard GPTCache without enhancements
            return create_enhanced_cache(
                enable_context=False,
                enable_pca=False,
                enable_tau=False
            )
        elif variant == "context":
            # Only context-chain filtering
            return create_enhanced_cache(
                enable_context=True,
                enable_pca=False,
                enable_tau=False
            )
        elif variant == "pca":
            # Only PCA compression
            return create_enhanced_cache(
                enable_context=False,
                enable_pca=True,
                enable_tau=False
            )
        elif variant == "tau":
            # Only τ-tuning
            return create_enhanced_cache(
                enable_context=False,
                enable_pca=False,
                enable_tau=True
            )
        elif variant == "full":
            # All features enabled
            return create_enhanced_cache(
                enable_context=True,
                enable_pca=True,
                enable_tau=True
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")
    
    def run_cache_warmup(self, cache, queries: List[QueryItem], warmup_fraction: float = 0.2):
        """Warm up cache with initial queries.
        
        Args:
            cache: Cache instance to warm up
            queries: List of queries for warmup
            warmup_fraction: Fraction of queries to use for warmup
        """
        warmup_count = int(len(queries) * warmup_fraction)
        warmup_queries = queries[:warmup_count]
        
        print(f"Warming up cache with {warmup_count} queries...")
        
        for query_item in tqdm(warmup_queries, desc="Cache warmup"):
            # Store some queries in cache to create realistic hit/miss scenarios
            if query_item.expected_response:
                cache.set(
                    query=query_item.query,
                    response=query_item.expected_response,
                    conversation_id=query_item.conversation_id
                )
            elif query_item.similarity_group:
                # For similar queries, cache some variations
                response = f"Cached response for {query_item.similarity_group}"
                cache.set(
                    query=query_item.query,
                    response=response,
                    conversation_id=query_item.conversation_id
                )
        
        print("Cache warmup complete")
    
    def run_single_benchmark(
        self, 
        variant: str, 
        queries: List[QueryItem],
        warmup: bool = True
    ) -> Dict[str, Any]:
        """Run benchmark for a single cache variant.
        
        Args:
            variant: Cache variant name
            queries: List of queries to benchmark
            warmup: Whether to warm up cache before benchmarking
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n=== Running benchmark for variant: {variant} ===")
        
        # Create cache instance
        cache = self.create_cache_variant(variant)
        
        # Warm up cache if requested
        if warmup:
            self.run_cache_warmup(cache, queries)
        
        # Reset metrics before actual benchmark
        cache.reset_metrics()
        get_performance_tracker().reset_metrics()
        
        # Run benchmark queries
        benchmark_queries = queries[int(len(queries) * 0.2):] if warmup else queries
        
        print(f"Running {len(benchmark_queries)} benchmark queries...")
        
        query_results = []
        start_time = time.time()
        
        with BenchmarkTimer("total_benchmark") as total_timer:
            for i, query_item in enumerate(tqdm(benchmark_queries, desc=f"Benchmarking {variant}")):
                with BenchmarkTimer("single_query") as query_timer:
                    try:
                        # Query the cache
                        result = cache.query(
                            query=query_item.query,
                            conversation_id=query_item.conversation_id,
                            ground_truth_response=query_item.expected_response
                        )
                        
                        # Record individual query result
                        query_result = {
                            'query_id': query_item.id,
                            'variant': variant,
                            'cache_hit': result['cache_hit'],
                            'latency_ms': result['latency_ms'],
                            'category': query_item.category,
                            'similarity_group': query_item.similarity_group,
                            'conversation_id': query_item.conversation_id,
                        }
                        
                        query_results.append(query_result)
                        
                    except Exception as e:
                        print(f"Error processing query {query_item.id}: {e}")
                        # Record failed query
                        query_result = {
                            'query_id': query_item.id,
                            'variant': variant,
                            'cache_hit': False,
                            'latency_ms': query_timer.elapsed_ms,
                            'category': query_item.category,
                            'error': str(e),
                        }
                        query_results.append(query_result)
        
        end_time = time.time()
        
        # Get final cache statistics
        cache_stats = cache.get_stats()
        
        # Compile benchmark results
        benchmark_result = {
            'variant': variant,
            'total_queries': len(benchmark_queries),
            'total_time_s': end_time - start_time,
            'queries_per_second': len(benchmark_queries) / (end_time - start_time),
            'cache_statistics': cache_stats['cache_statistics'],
            'performance_metrics': cache_stats['performance_metrics'],
            'query_results': query_results,
            'timestamp': time.time(),
        }
        
        # Add feature-specific statistics
        if 'context_statistics' in cache_stats:
            benchmark_result['context_statistics'] = cache_stats['context_statistics']
        
        if 'pca_statistics' in cache_stats:
            benchmark_result['pca_statistics'] = cache_stats['pca_statistics']
        
        if 'tau_statistics' in cache_stats:
            benchmark_result['tau_statistics'] = cache_stats['tau_statistics']
        
        print(f"Benchmark completed for {variant}:")
        print(f"  - Total queries: {benchmark_result['total_queries']}")
        print(f"  - Hit rate: {benchmark_result['cache_statistics']['hit_rate']:.2%}")
        print(f"  - Avg latency: {benchmark_result['performance_metrics']['avg_latency_ms']:.2f}ms")
        print(f"  - P95 latency: {benchmark_result['performance_metrics']['p95_latency_ms']:.2f}ms")
        print(f"  - Throughput: {benchmark_result['queries_per_second']:.1f} queries/sec")
        
        return benchmark_result
    
    def run_comparative_benchmark(
        self,
        queries: List[QueryItem],
        variants: Optional[List[str]] = None,
        warmup: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run comparative benchmark across multiple variants.
        
        Args:
            queries: List of queries to benchmark
            variants: List of cache variants to test
            warmup: Whether to warm up caches
            output_path: Path to save results
            
        Returns:
            Complete benchmark results
        """
        if variants is None:
            variants = self.config.benchmark.variants
        
        print(f"Starting comparative benchmark with variants: {variants}")
        print(f"Dataset size: {len(queries)} queries")
        
        benchmark_results = {
            'metadata': {
                'total_queries': len(queries),
                'variants': variants,
                'warmup_enabled': warmup,
                'timestamp': time.time(),
                'config': asdict(self.config),
            },
            'results': {}
        }
        
        # Run benchmark for each variant
        for variant in variants:
            try:
                result = self.run_single_benchmark(variant, queries, warmup)
                benchmark_results['results'][variant] = result
                self.results.append(result)
                
            except Exception as e:
                print(f"Benchmark failed for variant {variant}: {e}")
                benchmark_results['results'][variant] = {
                    'variant': variant,
                    'error': str(e),
                    'timestamp': time.time(),
                }
        
        # Save results if path provided
        if output_path:
            self._save_results(benchmark_results, output_path)
        
        # Print comparative summary
        self._print_comparative_summary(benchmark_results)
        
        return benchmark_results
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save benchmark results to file.
        
        Args:
            results: Benchmark results dictionary
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")
        
        # Also save as CSV for easy analysis
        csv_path = output_path.replace('.json', '_summary.csv')
        self._save_summary_csv(results, csv_path)
    
    def _save_summary_csv(self, results: Dict[str, Any], csv_path: str):
        """Save summary statistics as CSV.
        
        Args:
            results: Benchmark results
            csv_path: Path to save CSV file
        """
        summary_data = []
        
        for variant, result in results['results'].items():
            if 'error' in result:
                continue
                
            row = {
                'variant': variant,
                'hit_rate': result['cache_statistics']['hit_rate'],
                'avg_latency_ms': result['performance_metrics']['avg_latency_ms'],
                'p95_latency_ms': result['performance_metrics']['p95_latency_ms'],
                'p99_latency_ms': result['performance_metrics']['p99_latency_ms'],
                'queries_per_second': result['queries_per_second'],
                'total_queries': result['total_queries'],
                'total_time_s': result['total_time_s'],
                'avg_memory_mb': result['performance_metrics'].get('avg_memory_mb', 0),
                'avg_cpu_percent': result['performance_metrics'].get('avg_cpu_percent', 0),
            }
            
            # Add feature-specific metrics
            if 'context_statistics' in result:
                row['context_enabled'] = result['context_statistics'].get('enabled', False)
                row['avg_context_similarity'] = result['performance_metrics'].get('avg_context_similarity')
            
            if 'pca_statistics' in result:
                row['pca_enabled'] = result['pca_statistics'].get('enabled', False)
                row['compression_ratio'] = result['pca_statistics'].get('compression_ratio')
                row['explained_variance'] = result['pca_statistics'].get('explained_variance')
            
            if 'tau_statistics' in result:
                row['tau_enabled'] = result['tau_statistics'].get('enabled', False)
                row['final_threshold'] = result['tau_statistics'].get('current_threshold')
                row['tau_aggregations'] = result['tau_statistics'].get('aggregator_statistics', {}).get('total_aggregations', 0)
            
            summary_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to {csv_path}")
    
    def _print_comparative_summary(self, results: Dict[str, Any]):
        """Print comparative summary of benchmark results.
        
        Args:
            results: Benchmark results dictionary
        """
        print("\n" + "="*80)
        print("COMPARATIVE BENCHMARK SUMMARY")
        print("="*80)
        
        # Extract successful results
        successful_results = {
            variant: result for variant, result in results['results'].items()
            if 'error' not in result
        }
        
        if not successful_results:
            print("No successful benchmark results to compare.")
            return
        
        # Create comparison table
        print(f"\n{'Variant':<12} {'Hit Rate':<10} {'Avg Latency':<12} {'P95 Latency':<12} {'Throughput':<12} {'Memory':<10}")
        print("-" * 80)
        
        for variant, result in successful_results.items():
            hit_rate = result['cache_statistics']['hit_rate']
            avg_latency = result['performance_metrics']['avg_latency_ms']
            p95_latency = result['performance_metrics']['p95_latency_ms']
            throughput = result['queries_per_second']
            memory = result['performance_metrics'].get('avg_memory_mb', 0)
            
            print(f"{variant:<12} {hit_rate:<10.2%} {avg_latency:<12.1f} {p95_latency:<12.1f} {throughput:<12.1f} {memory:<10.1f}")
        
        # Find best performing variant for each metric
        print("\n" + "-" * 40)
        print("BEST PERFORMERS:")
        print("-" * 40)
        
        # Best hit rate
        best_hit_rate = max(successful_results.items(), key=lambda x: x[1]['cache_statistics']['hit_rate'])
        print(f"Best Hit Rate: {best_hit_rate[0]} ({best_hit_rate[1]['cache_statistics']['hit_rate']:.2%})")
        
        # Best latency (lowest)
        best_latency = min(successful_results.items(), key=lambda x: x[1]['performance_metrics']['avg_latency_ms'])
        print(f"Best Latency: {best_latency[0]} ({best_latency[1]['performance_metrics']['avg_latency_ms']:.1f}ms)")
        
        # Best throughput
        best_throughput = max(successful_results.items(), key=lambda x: x[1]['queries_per_second'])
        print(f"Best Throughput: {best_throughput[0]} ({best_throughput[1]['queries_per_second']:.1f} q/s)")
        
        # Memory efficiency (lowest)
        memory_results = {k: v for k, v in successful_results.items() 
                         if v['performance_metrics'].get('avg_memory_mb', 0) > 0}
        if memory_results:
            best_memory = min(memory_results.items(), key=lambda x: x[1]['performance_metrics']['avg_memory_mb'])
            print(f"Best Memory Usage: {best_memory[0]} ({best_memory[1]['performance_metrics']['avg_memory_mb']:.1f}MB)")

def load_queries_from_file(file_path: str) -> List[QueryItem]:
    """Load queries from JSON file.
    
    Args:
        file_path: Path to queries JSON file
        
    Returns:
        List of QueryItem objects
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    queries = []
    for query_data in data['queries']:
        query = QueryItem(**query_data)
        queries.append(query)
    
    return queries

def run_benchmark(
    queries_path: str,
    variants: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    warmup: bool = True
) -> Dict[str, Any]:
    """Run benchmark with specified parameters.
    
    Args:
        queries_path: Path to queries JSON file
        variants: List of variants to benchmark
        output_path: Path to save results
        config_path: Path to configuration file
        warmup: Whether to warm up cache
        
    Returns:
        Benchmark results
    """
    # Load queries
    queries = load_queries_from_file(queries_path)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(config_path)
    
    # Run benchmark
    results = runner.run_comparative_benchmark(
        queries=queries,
        variants=variants,
        warmup=warmup,
        output_path=output_path
    )
    
    return results

def main():
    """Command-line interface for benchmark runner."""
    parser = argparse.ArgumentParser(description="Run Enhanced GPTCache benchmarks")
    parser.add_argument("--queries", "--dataset", "-q", default="data/prompts_large.json", 
                       help="Path to queries JSON file (default: data/prompts_large.json)")
    parser.add_argument("--output", "-o", help="Output path for results")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--variants", "-v", nargs="+", 
                       choices=["baseline", "context", "pca", "tau", "full"],
                       help="Cache variants to benchmark")
    parser.add_argument("--no-warmup", action="store_true", help="Disable cache warmup")
    parser.add_argument("--sample", type=int, help="Use only first N queries (for CI/testing)")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        timestamp = int(time.time())
        args.output = f"data/results/benchmark_{timestamp}.json"
    
    # Load and potentially sample queries
    queries = load_queries_from_file(args.queries)
    
    if args.sample:
        queries = queries[:args.sample]
        print(f"Using sample of {len(queries)} queries for benchmarking")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(args.config)
    
    # Run benchmark
    results = runner.run_comparative_benchmark(
        queries=queries,
        variants=args.variants,
        warmup=not args.no_warmup,
        output_path=args.output
    )
    
    print(f"\nBenchmark completed! Results saved to {args.output}")

if __name__ == "__main__":
    main()

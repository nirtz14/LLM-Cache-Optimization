"""Analyze and visualize Enhanced GPTCache benchmark results."""
import json
import argparse
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ResultsAnalyzer:
    """Analyzes benchmark results and generates visualizations."""
    
    def __init__(self, results_path: str):
        """Initialize results analyzer.
        
        Args:
            results_path: Path to benchmark results JSON file
        """
        self.results_path = results_path
        self.results = self._load_results()
        self.summary_df = self._create_summary_dataframe()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        with open(self.results_path, 'r') as f:
            return json.load(f)
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame from results."""
        summary_data = []
        
        for variant, result in self.results['results'].items():
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
                'avg_memory_mb': result['performance_metrics'].get('avg_memory_mb', 0),
                'avg_cpu_percent': result['performance_metrics'].get('avg_cpu_percent', 0),
            }
            
            # Add feature-specific metrics
            if 'context_statistics' in result:
                ctx_stats = result['context_statistics']
                row.update({
                    'context_enabled': ctx_stats.get('enabled', False),
                    'total_conversations': ctx_stats.get('total_conversations', 0),
                    'avg_turns_per_conversation': ctx_stats.get('avg_turns_per_conversation', 0),
                    'cached_context_embeddings': ctx_stats.get('cached_context_embeddings', 0),
                })
            
            if 'pca_statistics' in result:
                pca_stats = result['pca_statistics']
                row.update({
                    'pca_enabled': pca_stats.get('enabled', False),
                    'compression_ratio': pca_stats.get('compression_ratio', 1.0),
                    'explained_variance': pca_stats.get('explained_variance', 0.0),
                    'total_compressions': pca_stats.get('total_compressions', 0),
                })
            
            if 'tau_statistics' in result:
                tau_stats = result['tau_statistics']
                agg_stats = tau_stats.get('aggregator_statistics', {})
                row.update({
                    'tau_enabled': tau_stats.get('enabled', False),
                    'final_threshold': tau_stats.get('current_threshold', 0.8),
                    'total_aggregations': agg_stats.get('total_aggregations', 0),
                    'num_users': tau_stats.get('num_users', 0),
                })
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def generate_performance_comparison(self, output_dir: str):
        """Generate performance comparison visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced GPTCache Performance Comparison', fontsize=16)
        
        # Hit Rate Comparison
        axes[0, 0].bar(self.summary_df['variant'], self.summary_df['hit_rate'])
        axes[0, 0].set_title('Cache Hit Rate by Variant')
        axes[0, 0].set_ylabel('Hit Rate (%)')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(self.summary_df['hit_rate']):
            axes[0, 0].text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
        
        # Average Latency Comparison
        axes[0, 1].bar(self.summary_df['variant'], self.summary_df['avg_latency_ms'])
        axes[0, 1].set_title('Average Latency by Variant')
        axes[0, 1].set_ylabel('Latency (ms)')
        for i, v in enumerate(self.summary_df['avg_latency_ms']):
            axes[0, 1].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
        
        # Throughput Comparison
        axes[1, 0].bar(self.summary_df['variant'], self.summary_df['queries_per_second'])
        axes[1, 0].set_title('Throughput by Variant')
        axes[1, 0].set_ylabel('Queries/Second')
        for i, v in enumerate(self.summary_df['queries_per_second']):
            axes[1, 0].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
        
        # Memory Usage Comparison
        if self.summary_df['avg_memory_mb'].sum() > 0:
            axes[1, 1].bar(self.summary_df['variant'], self.summary_df['avg_memory_mb'])
            axes[1, 1].set_title('Memory Usage by Variant')
            axes[1, 1].set_ylabel('Memory (MB)')
            for i, v in enumerate(self.summary_df['avg_memory_mb']):
                if v > 0:
                    axes[1, 1].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'Memory data not available', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].set_title('Memory Usage by Variant')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison saved to {output_dir}/performance_comparison.png")
    
    def generate_latency_distribution(self, output_dir: str):
        """Generate latency distribution visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create detailed query results DataFrame
        query_data = []
        for variant, result in self.results['results'].items():
            if 'error' in result or 'query_results' not in result:
                continue
            
            for query_result in result['query_results']:
                query_data.append({
                    'variant': variant,
                    'latency_ms': query_result['latency_ms'],
                    'cache_hit': query_result['cache_hit'],
                    'category': query_result.get('category', 'unknown'),
                })
        
        if not query_data:
            print("No query-level data available for latency distribution analysis")
            return
        
        query_df = pd.DataFrame(query_data)
        
        # Latency distribution by variant
        plt.figure(figsize=(12, 8))
        
        # Box plot
        plt.subplot(2, 2, 1)
        sns.boxplot(data=query_df, x='variant', y='latency_ms')
        plt.title('Latency Distribution by Variant')
        plt.ylabel('Latency (ms)')
        plt.xticks(rotation=45)
        
        # Histogram
        plt.subplot(2, 2, 2)
        for variant in query_df['variant'].unique():
            variant_data = query_df[query_df['variant'] == variant]['latency_ms']
            plt.hist(variant_data, alpha=0.7, label=variant, bins=30)
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency Histogram by Variant')
        plt.legend()
        
        # Cache hit vs miss latency
        plt.subplot(2, 2, 3)
        hit_data = query_df[query_df['cache_hit']]['latency_ms']
        miss_data = query_df[~query_df['cache_hit']]['latency_ms']
        
        plt.hist([hit_data, miss_data], label=['Cache Hit', 'Cache Miss'], 
                alpha=0.7, bins=30)
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency: Cache Hit vs Miss')
        plt.legend()
        
        # Latency by query category
        plt.subplot(2, 2, 4)
        sns.boxplot(data=query_df, x='category', y='latency_ms')
        plt.title('Latency by Query Category')
        plt.ylabel('Latency (ms)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Latency distribution saved to {output_dir}/latency_distribution.png")
    
    def generate_feature_analysis(self, output_dir: str):
        """Generate feature-specific analysis visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if we have the required columns for feature analysis
        required_cols = ['context_enabled', 'pca_enabled', 'tau_enabled']
        missing_cols = [col for col in required_cols if col not in self.summary_df.columns]
        
        if missing_cols:
            print(f"[WARN] Feature analysis columns missing: {missing_cols} - skipping feature plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature-Specific Analysis', fontsize=16)
        
        # Context feature analysis
        context_variants = self.summary_df[self.summary_df.get('context_enabled', False)]
        if not context_variants.empty:
            axes[0, 0].scatter(context_variants['total_conversations'], 
                              context_variants['hit_rate'], 
                              s=100, alpha=0.7)
            axes[0, 0].set_xlabel('Total Conversations')
            axes[0, 0].set_ylabel('Hit Rate')
            axes[0, 0].set_title('Context: Conversations vs Hit Rate')
            
            # Add variant labels
            for _, row in context_variants.iterrows():
                axes[0, 0].annotate(row['variant'], 
                                   (row['total_conversations'], row['hit_rate']),
                                   xytext=(5, 5), textcoords='offset points')
        else:
            axes[0, 0].text(0.5, 0.5, 'No context-enabled variants', 
                           transform=axes[0, 0].transAxes, ha='center', va='center')
            axes[0, 0].set_title('Context Analysis')
        
        # PCA feature analysis
        pca_variants = self.summary_df[self.summary_df.get('pca_enabled', False)]
        if not pca_variants.empty:
            axes[0, 1].scatter(pca_variants['compression_ratio'], 
                              pca_variants['avg_latency_ms'], 
                              s=100, alpha=0.7, c=pca_variants['explained_variance'])
            axes[0, 1].set_xlabel('Compression Ratio')
            axes[0, 1].set_ylabel('Average Latency (ms)')
            axes[0, 1].set_title('PCA: Compression vs Latency')
            
            # Add colorbar for explained variance
            scatter = axes[0, 1].scatter(pca_variants['compression_ratio'], 
                                        pca_variants['avg_latency_ms'], 
                                        s=100, alpha=0.7, 
                                        c=pca_variants['explained_variance'])
            plt.colorbar(scatter, ax=axes[0, 1], label='Explained Variance')
            
            # Add variant labels
            for _, row in pca_variants.iterrows():
                axes[0, 1].annotate(row['variant'], 
                                   (row['compression_ratio'], row['avg_latency_ms']),
                                   xytext=(5, 5), textcoords='offset points')
        else:
            axes[0, 1].text(0.5, 0.5, 'No PCA-enabled variants', 
                           transform=axes[0, 1].transAxes, ha='center', va='center')
            axes[0, 1].set_title('PCA Analysis')
        
        # Tau feature analysis
        tau_variants = self.summary_df[self.summary_df.get('tau_enabled', False)]
        if not tau_variants.empty:
            axes[1, 0].scatter(tau_variants['final_threshold'], 
                              tau_variants['hit_rate'], 
                              s=tau_variants['total_aggregations'] * 10, 
                              alpha=0.7)
            axes[1, 0].set_xlabel('Final Threshold')
            axes[1, 0].set_ylabel('Hit Rate')
            axes[1, 0].set_title('Tau: Threshold vs Hit Rate\n(Size = Aggregations)')
            
            # Add variant labels
            for _, row in tau_variants.iterrows():
                axes[1, 0].annotate(row['variant'], 
                                   (row['final_threshold'], row['hit_rate']),
                                   xytext=(5, 5), textcoords='offset points')
        else:
            axes[1, 0].text(0.5, 0.5, 'No tau-enabled variants', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('Tau Analysis')
        
        # Combined feature effectiveness
        feature_scores = []
        for _, row in self.summary_df.iterrows():
            # Calculate composite score
            score = (row['hit_rate'] * 0.4 + 
                    (1 - row['avg_latency_ms'] / 1000) * 0.3 + 
                    row['queries_per_second'] / 100 * 0.3)
            feature_scores.append(score)
        
        self.summary_df['composite_score'] = feature_scores
        
        axes[1, 1].bar(self.summary_df['variant'], self.summary_df['composite_score'])
        axes[1, 1].set_title('Composite Performance Score')
        axes[1, 1].set_ylabel('Score (Higher is Better)')
        axes[1, 1].set_xlabel('Variant')
        
        for i, v in enumerate(self.summary_df['composite_score']):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature analysis saved to {output_dir}/feature_analysis.png")
    
    def generate_summary_report(self, output_dir: str):
        """Generate a comprehensive text summary report."""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = f'{output_dir}/benchmark_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced GPTCache Benchmark Report\n\n")
            
            # Metadata
            f.write("## Benchmark Metadata\n\n")
            metadata = self.results['metadata']
            f.write(f"- **Total Queries**: {metadata['total_queries']}\n")
            f.write(f"- **Variants Tested**: {', '.join(metadata['variants'])}\n")
            f.write(f"- **Warmup Enabled**: {metadata['warmup_enabled']}\n")
            f.write(f"- **Timestamp**: {metadata['timestamp']}\n\n")
            
            # Performance Summary
            f.write("## Performance Summary\n\n")
            f.write("| Variant | Hit Rate | Avg Latency (ms) | P95 Latency (ms) | Throughput (q/s) | Memory (MB) |\n")
            f.write("|---------|----------|------------------|------------------|------------------|-------------|\n")
            
            for _, row in self.summary_df.iterrows():
                f.write(f"| {row['variant']} | {row['hit_rate']:.2%} | "
                       f"{row['avg_latency_ms']:.1f} | {row['p95_latency_ms']:.1f} | "
                       f"{row['queries_per_second']:.1f} | {row['avg_memory_mb']:.1f} |\n")
            
            f.write("\n")
            
            # Best Performers
            f.write("## Best Performers\n\n")
            
            best_hit_rate = self.summary_df.loc[self.summary_df['hit_rate'].idxmax()]
            f.write(f"- **Best Hit Rate**: {best_hit_rate['variant']} ({best_hit_rate['hit_rate']:.2%})\n")
            
            best_latency = self.summary_df.loc[self.summary_df['avg_latency_ms'].idxmin()]
            f.write(f"- **Best Latency**: {best_latency['variant']} ({best_latency['avg_latency_ms']:.1f}ms)\n")
            
            best_throughput = self.summary_df.loc[self.summary_df['queries_per_second'].idxmax()]
            f.write(f"- **Best Throughput**: {best_throughput['variant']} ({best_throughput['queries_per_second']:.1f} q/s)\n")
            
            if self.summary_df['avg_memory_mb'].sum() > 0:
                best_memory = self.summary_df.loc[self.summary_df['avg_memory_mb'].idxmin()]
                f.write(f"- **Best Memory**: {best_memory['variant']} ({best_memory['avg_memory_mb']:.1f}MB)\n")
            
            f.write("\n")
            
            # Feature Analysis
            f.write("## Feature Analysis\n\n")
            
            # Context analysis
            context_variants = self.summary_df[self.summary_df.get('context_enabled', False)]
            if not context_variants.empty:
                f.write("### Context-Chain Filtering\n\n")
                f.write("Variants with context filtering enabled:\n\n")
                for _, row in context_variants.iterrows():
                    f.write(f"- **{row['variant']}**: {row['total_conversations']} conversations, "
                           f"{row['avg_turns_per_conversation']:.1f} avg turns/conversation\n")
                f.write("\n")
            
            # PCA analysis
            pca_variants = self.summary_df[self.summary_df.get('pca_enabled', False)]
            if not pca_variants.empty:
                f.write("### PCA Embedding Compression\n\n")
                f.write("Variants with PCA compression enabled:\n\n")
                for _, row in pca_variants.iterrows():
                    f.write(f"- **{row['variant']}**: {row['compression_ratio']:.1f}x compression, "
                           f"{row['explained_variance']:.2%} variance explained\n")
                f.write("\n")
            
            # Tau analysis
            tau_variants = self.summary_df[self.summary_df.get('tau_enabled', False)]
            if not tau_variants.empty:
                f.write("### Federated τ-Tuning\n\n")
                f.write("Variants with τ-tuning enabled:\n\n")
                for _, row in tau_variants.iterrows():
                    f.write(f"- **{row['variant']}**: Final threshold {row['final_threshold']:.3f}, "
                           f"{row['total_aggregations']} aggregations\n")
                f.write("\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            
            best_overall = self.summary_df.loc[self.summary_df['composite_score'].idxmax()]
            f.write(f"The **{best_overall['variant']}** variant achieved the highest composite performance score "
                   f"of {best_overall['composite_score']:.3f}, combining good performance across hit rate, "
                   f"latency, and throughput metrics.\n\n")
            
            # Feature effectiveness
            baseline_hit_rate = self.summary_df[self.summary_df['variant'] == 'baseline']['hit_rate'].iloc[0] if 'baseline' in self.summary_df['variant'].values else 0
            
            improvements = []
            for _, row in self.summary_df.iterrows():
                if row['variant'] != 'baseline' and baseline_hit_rate > 0:
                    improvement = (row['hit_rate'] - baseline_hit_rate) / baseline_hit_rate * 100
                    if improvement > 1:  # Only report meaningful improvements
                        improvements.append((row['variant'], improvement))
            
            if improvements:
                f.write("### Key Improvements over Baseline:\n\n")
                for variant, improvement in sorted(improvements, key=lambda x: x[1], reverse=True):
                    f.write(f"- **{variant}**: {improvement:.1f}% hit rate improvement\n")
            
        print(f"Summary report saved to {report_path}")
    
    def analyze_all(self, output_dir: str):
        """Generate all analyses and visualizations."""
        print(f"Generating comprehensive analysis in {output_dir}...")
        
        self.generate_performance_comparison(output_dir)
        self.generate_latency_distribution(output_dir)
        self.generate_feature_analysis(output_dir)
        self.generate_summary_report(output_dir)
        
        print(f"Analysis complete! All outputs saved to {output_dir}")

def analyze_benchmark_results(
    results_path: str,
    output_dir: str = "data/analysis"
) -> None:
    """Analyze benchmark results and generate visualizations.
    
    Args:
        results_path: Path to benchmark results JSON file
        output_dir: Directory to save analysis outputs
    """
    analyzer = ResultsAnalyzer(results_path)
    analyzer.analyze_all(output_dir)

def main():
    """Command-line interface for results analysis."""
    parser = argparse.ArgumentParser(description="Analyze Enhanced GPTCache benchmark results")
    parser.add_argument("--results", "-r", required=True, help="Path to benchmark results JSON file")
    parser.add_argument("--output", "-o", default="data/analysis", help="Output directory for analysis")
    parser.add_argument("--skip-feature-plot", action="store_true", help="Skip feature analysis plot generation")
    
    args = parser.parse_args()
    
    if args.skip_feature_plot:
        analyzer = ResultsAnalyzer(args.results)
        analyzer.generate_performance_comparison(args.output)
        analyzer.generate_latency_distribution(args.output) 
        analyzer.generate_summary_report(args.output)
        print(f"Analysis complete (skipped feature plot)! Outputs saved to {args.output}")
    else:
        analyze_benchmark_results(args.results, args.output)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analysis tools for Enhanced GPTCache benchmark results - FIXED VERSION.
"""

# FIX 1: Use non-interactive matplotlib backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # This MUST be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import os

class ResultsAnalyzer:
    """Analyze and visualize Enhanced GPTCache benchmark results."""
    
    def __init__(self, results_file: str):
        """Initialize with results file path."""
        self.results_file = results_file
        self.results_data = self._load_results()
        self.results = self.results_data  # Add this line for test compatibility
        self.summary_df = self._create_summary_dataframe()
    
    def _load_results(self) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame from results."""
        summary_data = []
        
        # Track if any result has feature statistics
        has_context_stats = False
        has_pca_stats = False
        has_tau_stats = False
        
        for variant, result in self.results_data['results'].items():
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
            
            # Only add feature columns if the statistics exist in the results
            if 'context_statistics' in result:
                row['context_enabled'] = self._get_feature_enabled(result, 'context')
                has_context_stats = True
            
            if 'pca_statistics' in result:
                row['pca_enabled'] = self._get_feature_enabled(result, 'pca')
                has_pca_stats = True
            
            if 'tau_statistics' in result:
                row['tau_enabled'] = self._get_feature_enabled(result, 'tau')
                has_tau_stats = True
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Only add missing feature columns if at least one result had them
        # This preserves the test's ability to test missing columns
        if has_context_stats:
            if 'context_enabled' not in df.columns:
                df['context_enabled'] = False
            df['context_enabled'] = df['context_enabled'].fillna(False).astype(bool)
        
        if has_pca_stats:
            if 'pca_enabled' not in df.columns:
                df['pca_enabled'] = False
            df['pca_enabled'] = df['pca_enabled'].fillna(False).astype(bool)
        
        if has_tau_stats:
            if 'tau_enabled' not in df.columns:
                df['tau_enabled'] = False
            df['tau_enabled'] = df['tau_enabled'].fillna(False).astype(bool)
        
        return df
    
    def _get_feature_enabled(self, result: Dict[str, Any], feature: str) -> bool:
        """Safely extract if a feature is enabled from result data."""
        # Check various places where feature info might be stored
        if f'{feature}_statistics' in result:
            return result[f'{feature}_statistics'].get('enabled', False)
        
        # Check if variant name contains the feature
        variant = result.get('variant', '')
        if variant == 'full':
            return True
        elif variant == feature:
            return True
        elif feature in variant:
            return True
        
        return False
    
    def generate_performance_comparison(self, output_dir: str):
        """Generate performance comparison charts."""
        print("Generating performance comparison charts...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced GPTCache Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Hit Rate Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(self.summary_df['variant'], self.summary_df['hit_rate'] * 100)
        ax1.set_title('Cache Hit Rate by Variant')
        ax1.set_ylabel('Hit Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Latency Comparison
        ax2 = axes[0, 1]
        x_pos = np.arange(len(self.summary_df))
        ax2.bar(x_pos - 0.2, self.summary_df['avg_latency_ms'], 0.4, label='Average', alpha=0.8)
        ax2.bar(x_pos + 0.2, self.summary_df['p95_latency_ms'], 0.4, label='P95', alpha=0.8)
        ax2.set_title('Latency Comparison')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.summary_df['variant'], rotation=45)
        ax2.legend()
        
        # 3. Throughput Comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(self.summary_df['variant'], self.summary_df['queries_per_second'])
        ax3.set_title('Throughput Comparison')
        ax3.set_ylabel('Queries per Second')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 4. Memory Usage Comparison
        ax4 = axes[1, 1]
        bars4 = ax4.bar(self.summary_df['variant'], self.summary_df['avg_memory_mb'])
        ax4.set_title('Memory Usage Comparison')
        ax4.set_ylabel('Memory (MB)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'performance_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Important: close figure to free memory
        
        print(f"Performance comparison saved to {output_path}")
    
    def generate_feature_analysis(self, output_dir: str):
        """Generate feature-specific analysis."""
        print("Generating feature analysis...")
        
        feature_cols = ['context_enabled', 'pca_enabled', 'tau_enabled']
        
        
        missing_cols = [col for col in feature_cols if col not in self.summary_df.columns]
        
        if missing_cols:
            print(f"[WARN] Feature analysis columns missing: {missing_cols} - skipping feature plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Filter variants by feature
        try:
            context_variants = self.summary_df[self.summary_df['context_enabled'] == True]
            pca_variants = self.summary_df[self.summary_df['pca_enabled'] == True]
            tau_variants = self.summary_df[self.summary_df['tau_enabled'] == True]
            
            # Create feature impact plot
            features = ['Context Filtering', 'PCA Compression', 'Tau Tuning']
            hit_rates = []
            
            baseline_hit_rate = self.summary_df[self.summary_df['variant'] == 'baseline']['hit_rate'].iloc[0]
            
            # Calculate feature impacts
            context_impact = context_variants['hit_rate'].mean() - baseline_hit_rate if len(context_variants) > 0 else 0
            pca_impact = pca_variants['hit_rate'].mean() - baseline_hit_rate if len(pca_variants) > 0 else 0  
            tau_impact = tau_variants['hit_rate'].mean() - baseline_hit_rate if len(tau_variants) > 0 else 0
            
            impacts = [context_impact * 100, pca_impact * 100, tau_impact * 100]
            
            colors = ['red' if x < 0 else 'green' for x in impacts]
            bars = plt.bar(features, impacts, color=colors, alpha=0.7)
            
            plt.title('Feature Impact on Hit Rate', fontsize=14, fontweight='bold')
            plt.ylabel('Hit Rate Change (%)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar, impact in zip(bars, impacts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                        f'{impact:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'feature_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Feature analysis saved to {output_path}")
            
        except Exception as e:
            print(f"[ERROR] Feature analysis failed: {e}")
    
    def generate_summary_report(self, output_dir: str):
        """Generate summary report."""
        print("Generating summary report...")
        
        report_lines = []
        report_lines.append("# Enhanced GPTCache Benchmark Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Performance Summary
        report_lines.append("## Performance Summary")
        report_lines.append("")
        
        best_hit_rate = self.summary_df.loc[self.summary_df['hit_rate'].idxmax()]
        worst_hit_rate = self.summary_df.loc[self.summary_df['hit_rate'].idxmin()]
        
        report_lines.append(f"**Best Hit Rate**: {best_hit_rate['variant']} ({best_hit_rate['hit_rate']:.1%})")
        report_lines.append(f"**Lowest Hit Rate**: {worst_hit_rate['variant']} ({worst_hit_rate['hit_rate']:.1%})")
        report_lines.append("")
        
        # Calculate improvements
        if 'baseline' in self.summary_df['variant'].values:
            baseline = self.summary_df[self.summary_df['variant'] == 'baseline'].iloc[0]
            report_lines.append("## Improvements over Baseline")
            report_lines.append("")
            
            for _, row in self.summary_df.iterrows():
                if row['variant'] != 'baseline':
                    hit_diff = (row['hit_rate'] - baseline['hit_rate']) * 100
                    latency_diff = ((row['avg_latency_ms'] - baseline['avg_latency_ms']) / baseline['avg_latency_ms']) * 100
                    throughput_diff = ((row['queries_per_second'] - baseline['queries_per_second']) / baseline['queries_per_second']) * 100
                    
                    report_lines.append(f"**{row['variant'].capitalize()}**:")
                    report_lines.append(f"  - Hit rate change: {hit_diff:+.1f}%")
                    report_lines.append(f"  - Latency change: {latency_diff:+.1f}%") 
                    report_lines.append(f"  - Throughput change: {throughput_diff:+.1f}%")
                    report_lines.append("")
        
        # Save report
        report_path = os.path.join(output_dir, 'benchmark_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to {report_path}")
    
    def analyze_all(self, output_dir: str):
        """Run all analysis and generate all outputs."""
        print(f"Generating comprehensive analysis in {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all analyses
        self.generate_performance_comparison(output_dir)
        self.generate_latency_distribution(output_dir)
        self.generate_feature_analysis(output_dir)
        self.generate_summary_report(output_dir)
        
        print("All analyses completed!")
    
    def generate_latency_distribution(self, output_dir: str):
        """Generate latency distribution plots."""
        print("Generating latency distribution...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        data_for_box = [self.summary_df['avg_latency_ms'], self.summary_df['p95_latency_ms']]
        labels = ['Average Latency', 'P95 Latency']
        
        box_plot = ax1.boxplot(data_for_box, tick_labels=labels, patch_artist=True)
        ax1.set_title('Latency Distribution Across Variants')
        ax1.set_ylabel('Latency (ms)')
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        # Scatter plot
        x_pos = np.arange(len(self.summary_df))
        ax2.scatter(x_pos, self.summary_df['avg_latency_ms'], label='Average', alpha=0.7, s=100)
        ax2.scatter(x_pos, self.summary_df['p95_latency_ms'], label='P95', alpha=0.7, s=100)
        
        ax2.set_title('Latency by Variant')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.summary_df['variant'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'latency_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Latency distribution saved to {output_path}")

def main():
    """Command line interface for results analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Enhanced GPTCache benchmark results")
    parser.add_argument("--input", "-i", required=True, help="Input results JSON file")
    parser.add_argument("--output", "-o", default="data/analysis", help="Output directory")
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.input)
    analyzer.analyze_all(args.output)

if __name__ == "__main__":
    main()
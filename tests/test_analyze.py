"""Unit tests for results analysis functionality."""
import pytest
import json
import tempfile
import os
from pathlib import Path
from benchmark.analyze_results import ResultsAnalyzer
import pandas as pd


class TestAnalyzeResults:
    """Test results analysis functionality."""
    
    def create_test_results_file(self, include_feature_columns: bool = True):
        """Create a temporary test results JSON file."""
        test_results = {
            "metadata": {
                "total_queries": 10,
                "variants": ["baseline", "full"],
                "warmup_enabled": False,
                "timestamp": "2024-01-01T00:00:00Z"
            },
            "results": {
                "baseline": {
                    "cache_statistics": {"hit_rate": 0.0},
                    "performance_metrics": {
                        "avg_latency_ms": 10.0,
                        "p95_latency_ms": 15.0,
                        "p99_latency_ms": 20.0,
                        "avg_memory_mb": 100.0,
                        "avg_cpu_percent": 50.0
                    },
                    "queries_per_second": 100.0,
                    "total_queries": 10,
                    "query_results": [
                        {
                            "latency_ms": 10.0,
                            "cache_hit": False,
                            "category": "novel"
                        }
                    ]
                },
                "full": {
                    "cache_statistics": {"hit_rate": 0.2},
                    "performance_metrics": {
                        "avg_latency_ms": 8.0,
                        "p95_latency_ms": 12.0,
                        "p99_latency_ms": 16.0,
                        "avg_memory_mb": 120.0,
                        "avg_cpu_percent": 55.0
                    },
                    "queries_per_second": 120.0,
                    "total_queries": 10,
                    "query_results": [
                        {
                            "latency_ms": 8.0,
                            "cache_hit": True,
                            "category": "repetitive"
                        }
                    ]
                }
            }
        }
        
        # Add feature statistics if requested
        if include_feature_columns:
            test_results["results"]["full"].update({
                "context_statistics": {
                    "enabled": True,
                    "total_conversations": 5,
                    "avg_turns_per_conversation": 2.0,
                    "cached_context_embeddings": 10
                },
                "pca_statistics": {
                    "enabled": True,
                    "compression_ratio": 6.0,
                    "explained_variance": 0.95,
                    "total_compressions": 20
                },
                "tau_statistics": {
                    "enabled": True,
                    "current_threshold": 0.75,
                    "num_users": 1,
                    "aggregator_statistics": {
                        "total_aggregations": 3
                    }
                }
            })
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_results, temp_file, indent=2)
        temp_file.close()
        
        return temp_file.name
    
    def test_analyzer_loads_results(self):
        """Test that analyzer can load and parse results."""
        results_file = self.create_test_results_file()
        
        try:
            analyzer = ResultsAnalyzer(results_file)
            
            # Check that results are loaded
            assert analyzer.results is not None
            assert "metadata" in analyzer.results
            assert "results" in analyzer.results
            
            # Check that summary dataframe is created
            assert not analyzer.summary_df.empty
            assert len(analyzer.summary_df) == 2  # baseline + full variants
            
        finally:
            os.unlink(results_file)
    
    def test_feature_analysis_with_missing_columns(self, capfd):
        """Test that feature analysis gracefully handles missing feature columns."""
        # Create results file without feature statistics
        results_file = self.create_test_results_file(include_feature_columns=False)
        
        try:
            analyzer = ResultsAnalyzer(results_file)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # This should not raise an error and should print a warning
                analyzer.generate_feature_analysis(temp_dir)
                
                # Check that warning was printed
                captured = capfd.readouterr()
                assert "[WARN] Feature analysis columns missing:" in captured.out
                assert "skipping feature plot" in captured.out
                
                # Check that no feature_analysis.png was created
                assert not os.path.exists(os.path.join(temp_dir, "feature_analysis.png"))
                
        finally:
            os.unlink(results_file)
    
    def test_feature_analysis_with_feature_columns(self):
        """Test that feature analysis works when feature columns are present."""
        # Create results file with feature statistics
        results_file = self.create_test_results_file(include_feature_columns=True)
        
        try:
            analyzer = ResultsAnalyzer(results_file)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # This should work without errors
                analyzer.generate_feature_analysis(temp_dir)
                
                # Check that feature_analysis.png was created
                assert os.path.exists(os.path.join(temp_dir, "feature_analysis.png"))
                
        finally:
            os.unlink(results_file)
    
    def test_performance_comparison_generation(self):
        """Test that performance comparison plots are generated."""
        results_file = self.create_test_results_file()
        
        try:
            analyzer = ResultsAnalyzer(results_file)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                analyzer.generate_performance_comparison(temp_dir)
                
                # Check that plot was created
                assert os.path.exists(os.path.join(temp_dir, "performance_comparison.png"))
                
        finally:
            os.unlink(results_file)
    
    def test_latency_distribution_generation(self):
        """Test that latency distribution plots are generated."""
        results_file = self.create_test_results_file()
        
        try:
            analyzer = ResultsAnalyzer(results_file)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                analyzer.generate_latency_distribution(temp_dir)
                
                # Check that plot was created
                assert os.path.exists(os.path.join(temp_dir, "latency_distribution.png"))
                
        finally:
            os.unlink(results_file)
    
    def test_summary_report_generation(self):
        """Test that summary report is generated."""
        results_file = self.create_test_results_file()
        
        try:
            analyzer = ResultsAnalyzer(results_file)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                analyzer.generate_summary_report(temp_dir)
                
                # Check that report was created
                report_path = os.path.join(temp_dir, "benchmark_report.md")
                assert os.path.exists(report_path)
                
                # Check report contains expected content
                with open(report_path, 'r') as f:
                    content = f.read()
                    assert "# Enhanced GPTCache Benchmark Report" in content
                    assert "baseline" in content
                    assert "full" in content
                
        finally:
            os.unlink(results_file)
    
    def test_analyze_all_without_feature_columns(self, capfd):
        """Test that analyze_all works even when feature columns are missing."""
        results_file = self.create_test_results_file(include_feature_columns=False)
        
        try:
            analyzer = ResultsAnalyzer(results_file)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                analyzer.analyze_all(temp_dir)
                
                # Check that main plots were created
                assert os.path.exists(os.path.join(temp_dir, "performance_comparison.png"))
                assert os.path.exists(os.path.join(temp_dir, "latency_distribution.png"))
                assert os.path.exists(os.path.join(temp_dir, "benchmark_report.md"))
                
                # Feature analysis should have been skipped with warning
                captured = capfd.readouterr()
                assert "[WARN] Feature analysis columns missing:" in captured.out
                assert not os.path.exists(os.path.join(temp_dir, "feature_analysis.png"))
                
        finally:
            os.unlink(results_file)

#!/bin/bash
set -e

echo "Enhanced GPTCache Docker Container"
echo "=================================="

# Function to show help
show_help() {
    echo "Available commands:"
    echo "  generate-queries  - Generate synthetic query datasets"
    echo "  benchmark        - Run cache benchmarks"
    echo "  analyze          - Analyze benchmark results"
    echo "  test             - Run test suite"
    echo "  shell            - Start interactive shell"
    echo ""
    echo "Examples:"
    echo "  docker run enhanced-gptcache generate-queries --output data/queries.json --count 500"
    echo "  docker run enhanced-gptcache benchmark --queries data/queries.json --output data/results.json"
    echo "  docker run enhanced-gptcache analyze --results data/results.json --output data/analysis/"
    echo "  docker run enhanced-gptcache test"
}

# Function to generate queries
generate_queries() {
    echo "Generating synthetic queries..."
    python -m benchmark.generate_queries "$@"
}

# Function to run benchmarks
run_benchmark() {
    echo "Running cache benchmarks..."
    python -m benchmark.benchmark_runner "$@"
}

# Function to analyze results
analyze_results() {
    echo "Analyzing benchmark results..."
    python -m benchmark.analyze_results "$@"
}

# Function to run tests
run_tests() {
    echo "Running test suite..."
    pytest tests/ --cov=src --cov-report=term-missing "$@"
}

# Function to start shell
start_shell() {
    echo "Starting interactive shell..."
    exec /bin/bash
}

# Main command handling
case "$1" in
    generate-queries)
        shift
        generate_queries "$@"
        ;;
    benchmark)
        shift  
        run_benchmark "$@"
        ;;
    analyze)
        shift
        analyze_results "$@"
        ;;
    test)
        shift
        run_tests "$@"
        ;;
    shell)
        start_shell
        ;;
    --help|-h|help)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

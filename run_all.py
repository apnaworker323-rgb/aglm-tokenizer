"""
Command Line Runner for AGLM Universal Multilingual Tokenizer Suite.
"""

import sys
import os

# Add package root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aglm_tokenizer.experiments.run_global_benchmark import GlobalBenchmarkRunner


def main():
    runner = GlobalBenchmarkRunner(output_dir="./benchmark_results")
    results = runner.run_all()
    print("\nBenchmark successfully executed. All artifacts saved to ./benchmark_results")


if __name__ == "__main__":
    main()

"""
main_experiments.py
Main script to run fair hierarchical clustering experiments.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from experiments.runner import ExperimentRunner
from visualization.dendograms import (
    plot_colored_tree,
    plot_balance_distribution,
    plot_comparison,
    plot_cost_vs_size,
    plot_parameter_effect,
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run fair hierarchical clustering experiments"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run a minimal test experiment"
    )
    args = parser.parse_args()

    # Create experiment runner
    runner = ExperimentRunner(output_dir=args.output_dir)

    if args.test:
        # Run a single test experiment
        print("Running test experiment...")
        result = runner.run_experiment(
            name="test", dataset="synthetic", sample_size=50, h=4, k=2, epsilon=1 / 16
        )

        # Plot trees
        plot_colored_tree(
            hierarchy=result.vanilla_hierarchy,
            color_ids=result.colors,
            title="Vanilla Hierarchical Clustering",
            save_path=os.path.join(args.output_dir, "test_vanilla_tree.png"),
        )

        plot_colored_tree(
            hierarchy=result.fair_hierarchy,
            color_ids=result.colors,
            title="Fair Hierarchical Clustering",
            save_path=os.path.join(args.output_dir, "test_fair_tree.png"),
        )

        # Plot balance comparison
        global_balance = result.vanilla_balance_stats["global_balance"]
        plot_comparison(
            vanilla_balances=result.vanilla_balance_stats["balances"],
            fair_balances=result.fair_balance_stats["balances"],
            data_balance=global_balance,
            save_path=os.path.join(args.output_dir, "test_balance_comparison.png"),
        )

        print("Test experiment complete!")

    else:
        # Run full experiments
        print("Running full experiments...")

        # Experiment 1: Varying sample size
        sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        print(f"Experiment 1: Varying sample size {sizes}")
        size_results = runner.run_varying_size(
            base_name="vary_size",
            dataset="synthetic",
            sizes=sizes,
            h=4,
            k=2,
            epsilon=1 / 16,
            seed=230,
        )

        # Plot cost vs size
        cost_ratios = [r.cost_ratio for r in size_results]
        plot_cost_vs_size(
            sizes=sizes,
            cost_ratios=cost_ratios,
            save_path=os.path.join(args.output_dir, "cost_vs_size.png"),
        )

        # Experiment 2: Varying h parameter
        h_values = [2, 4, 8, 16, 32]
        print(f"Experiment 2: Varying h parameter {h_values}")
        h_results = runner.run_varying_h(
            base_name="vary_h",
            dataset="synthetic",
            sample_size=2056 * 2,
            h_values=h_values,
            k=16,
            epsilon=1 / 16,
            seed=230,
        )

        # Experiment 3: Varying k parameter
        #

        # Save all results to CSV
        df = runner.save_results_to_csv()

        # Plot parameter effects
        plot_parameter_effect(
            df=df,
            x_param="h",
            y_metric="cost_ratio",
            title="Effect of h on Cost Ratio",
            save_path=os.path.join(args.output_dir, "h_vs_cost.png"),
        )

        plot_parameter_effect(
            df=df,
            x_param="h",
            y_metric="improvement_ratio",
            title="Effect of h on Fairness Improvement",
            save_path=os.path.join(args.output_dir, "h_vs_improvement.png"),
        )

        print("All experiments complete!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.2f}s")

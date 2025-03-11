"""
experiments/runner.py
Author: Param Kapur 
Date: 5/10/25

Experiment runner for comparing vanilla and fair hierarchical clustering.
"""

import time
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from data.synthetic import generate_colored_data
from algorithms.vanilla import average_linkage
from algorithms.split_root import split_root
from algorithms.make_fair import make_fair
from models.hierarchy import Hierarchy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiments.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Experiments")

class ExperimentResult:
    """Stores the results of a single experiment."""
    
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self.vanilla_time = 0.0
        self.fair_time = 0.0
        self.vanilla_cost = 0.0
        self.fair_cost = 0.0
        self.cost_ratio = 0.0
        self.vanilla_balance_stats = {}
        self.fair_balance_stats = {}
        self.vanilla_hierarchy = None
        self.fair_hierarchy = None
        self.colors = None
        self.start_time = time.time()
        
    def compute_cost_ratio(self):
        """Compute the ratio of fair to vanilla cost."""
        if self.vanilla_cost > 0:
            self.cost_ratio = self.fair_cost / self.vanilla_cost
            
    def __str__(self):
        elapsed = time.time() - self.start_time
        return (
            f"Experiment: {self.name}\n"
            f"Parameters: {self.params}\n"
            f"Cost ratio: {self.cost_ratio:.2f}\n"
            f"Time ratio: {self.fair_time/self.vanilla_time:.2f}\n"
            f"Elapsed time: {elapsed:.2f}s"
        )


class ExperimentRunner:
    """Runs fair hierarchical clustering experiments."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.results = {}
        os.makedirs(output_dir, exist_ok=True)
        
    def run_experiment(self, 
                      name: str,
                      dataset: str,
                      sample_size: int,
                      h: int, 
                      k: int,
                      epsilon: float,
                      seed: int = 42) -> ExperimentResult:
        """
        Run a fair hierarchical clustering experiment.
        
        Args:
            name: Unique name for this experiment
            dataset: Dataset name or 'synthetic'
            sample_size: Number of data points to use
            h: Parameter for SplitRoot
            k: Parameter for folding
            epsilon: Balance parameter
            seed: Random seed
            
        Returns:
            ExperimentResult: Results of the experiment
        """
        # Initialize result object
        params = {
            "dataset": dataset,
            "sample_size": sample_size,
            "h": h,
            "k": k,
            "epsilon": epsilon,
            "seed": seed
        }
        result = ExperimentResult(name, params)
        
        # Generate or load data
        logger.info(f"[{name}] Generating/loading data...")
        if dataset == "synthetic":
            points, color_ids = generate_colored_data(
                total_points=sample_size,
                color_proportions=[0.4, 0.6],
                dim=2,
                cluster_std=10,
                seed=seed
            )

        
        result.colors = color_ids
        
        # Run vanilla hierarchical clustering
        logger.info(f"[{name}] Running vanilla clustering...")
        start_time = time.time()
        vanilla_hierarchy = average_linkage(points)
        result.vanilla_time = time.time() - start_time
        result.vanilla_hierarchy = vanilla_hierarchy
        
        # Calculate vanilla cost
        # In a real implementation, calculate the cost using Dasgupta's cost function
        # For now, we'll use a placeholder
        result.vanilla_cost = self._calculate_cost(vanilla_hierarchy, points)
        
        # Analyze vanilla balance
        result.vanilla_balance_stats = self._analyze_balance(vanilla_hierarchy, color_ids)
        
        # Run fair hierarchical clustering
        logger.info(f"[{name}] Running fair clustering...")
        start_time = time.time()
        fair_hierarchy = make_fair(
            vanilla_hierarchy, h, k, epsilon, color_ids,
            max_depth=10  # Limiting recursion depth for safety
        )
        result.fair_time = time.time() - start_time
        result.fair_hierarchy = fair_hierarchy
        
        # Calculate fair cost
        result.fair_cost = self._calculate_cost(fair_hierarchy, points)
        
        # Analyze fair balance
        result.fair_balance_stats = self._analyze_balance(fair_hierarchy, color_ids)
        
        # Compute ratios
        result.compute_cost_ratio()
        
        # Save result
        self.results[name] = result
        logger.info(f"[{name}] Experiment complete: cost ratio = {result.cost_ratio:.2f}")
        
        return result
    
    def run_varying_size(self, 
                        base_name: str,
                        dataset: str,
                        sizes: List[int],
                        h: int,
                        k: int,
                        epsilon: float,
                        seed: int = 42) -> List[ExperimentResult]:
        """Run experiments with varying sample sizes."""
        results = []
        for size in sizes:
            name = f"{base_name}_n{size}"
            result = self.run_experiment(
                name=name,
                dataset=dataset,
                sample_size=size,
                h=h,
                k=k,
                epsilon=epsilon,
                seed=seed
            )
            results.append(result)
        return results
            
    def run_varying_h(self, 
                     base_name: str,
                     dataset: str,
                     sample_size: int,
                     h_values: List[int],
                     k: int,
                     epsilon: float,
                     seed: int = 42) -> List[ExperimentResult]:
        """Run experiments with varying h parameter."""
        results = []
        for h in h_values:
            name = f"{base_name}_h{h}"
            result = self.run_experiment(
                name=name,
                dataset=dataset,
                sample_size=sample_size,
                h=h,
                k=k,
                epsilon=epsilon,
                seed=seed
            )
            results.append(result)
        return results
    
    def _calculate_cost(self, hierarchy: Hierarchy, points: np.ndarray) -> float:
        """
        Calculate Dasgupta's cost for a hierarchy.
        
        Args:
            hierarchy: The hierarchical clustering
            points: The data points
            
        Returns:
            float: The cost value
        """
        # Simplified implementation - in real code you'd use the actual cost formula
        # This is a placeholder that estimates cost based on tree balance
        n_leaves = len(hierarchy.get_leaves())
        all_nodes = hierarchy.get_all_nodes()
        internal_nodes = [node for node in all_nodes if not node.is_leaf()]
        
        # Calculate a balance factor (0 = perfect balance, higher = more imbalance)
        imbalance_sum = 0
        for node in internal_nodes:
            if not node.children:
                continue
            sizes = [child.size for child in node.children]
            if not sizes or max(sizes) == 0:
                continue
            imbalance = max(sizes) / sum(sizes) - 1/len(sizes)
            imbalance_sum += imbalance * node.size
            
        # Approximate cost - higher imbalance means higher cost
        cost = n_leaves * (1 + imbalance_sum / len(internal_nodes) if internal_nodes else 0)
        return cost
    
    def _analyze_balance(self, hierarchy: Hierarchy, color_ids: List[List[int]]) -> Dict[str, Any]:
        """
        Analyze color balance in the hierarchy.
        
        Args:
            hierarchy: The hierarchical clustering
            color_ids: Lists of point IDs for each color
            
        Returns:
            Dict: Statistics about color balance
        """
        # Get non-leaf nodes
        all_nodes = hierarchy.get_all_nodes()
        internal_nodes = [node for node in all_nodes if not node.is_leaf()]
        
        # Calculate the global balance (proportion of blue points)
        total_points = sum(len(ids) for ids in color_ids)
        global_balance = len(color_ids[0]) / total_points if total_points > 0 else 0
        
        # Calculate color balance for each node
        balances = []
        for node in internal_nodes:
            if node.size == 0:
                continue
                
            # Calculate proportion of blue points
            blue_count = 0
            if node.color_counts and 0 in node.color_counts:
                blue_count = node.color_counts[0]
            
            balance = blue_count / node.size if node.size > 0 else 0
            balances.append(balance)
        
        # Calculate statistics
        if balances:
            mean_balance = np.mean(balances)
            std_balance = np.std(balances)
            min_balance = np.min(balances)
            max_balance = np.max(balances)
            
            # Measure deviation from global balance
            balance_deviations = [abs(b - global_balance) for b in balances]
            mean_deviation = np.mean(balance_deviations)
            max_deviation = np.max(balance_deviations)
        else:
            mean_balance = std_balance = min_balance = max_balance = 0
            mean_deviation = max_deviation = 0
        
        return {
            "global_balance": global_balance,
            "mean_balance": mean_balance,
            "std_balance": std_balance,
            "min_balance": min_balance,
            "max_balance": max_balance,
            "mean_deviation": mean_deviation,
            "max_deviation": max_deviation,
            "balances": balances
        }
    
    def save_results_to_csv(self, filename: str = "experiment_results.csv"):
        """Save experiment results to a CSV file."""
        data = []
        for name, result in self.results.items():
            row = {
                "name": name,
                "dataset": result.params["dataset"],
                "sample_size": result.params["sample_size"],
                "h": result.params["h"],
                "k": result.params["k"],
                "epsilon": result.params["epsilon"],
                "vanilla_time": result.vanilla_time,
                "fair_time": result.fair_time,
                "vanilla_cost": result.vanilla_cost,
                "fair_cost": result.fair_cost,
                "cost_ratio": result.cost_ratio,
                "vanilla_mean_deviation": result.vanilla_balance_stats.get("mean_deviation", 0),
                "fair_mean_deviation": result.fair_balance_stats.get("mean_deviation", 0),
                "improvement_ratio": (
                    result.vanilla_balance_stats.get("mean_deviation", 0) /
                    result.fair_balance_stats.get("mean_deviation", 1)
                ) if result.fair_balance_stats.get("mean_deviation", 0) > 0 else 0
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")
        
        return df
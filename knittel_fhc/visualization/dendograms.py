"""
Visualization tools for hierarchical clustering trees.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union
from scipy.cluster.hierarchy import dendrogram
from models.hierarchy import Hierarchy
from models.node import Node

def plot_colored_tree(
    hierarchy: Hierarchy,
    color_ids: Optional[List[List[int]]] = None,
    title: str = "Hierarchical Clustering",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a custom visualization of the hierarchical clustering tree.
    This is a simple alternative to scipy's dendrogram when conversion is difficult.
    
    Args:
        hierarchy: The hierarchical clustering to visualize
        color_ids: Lists of point indices for each color category
        title: Plot title
        figsize: Figure size
        save_path: If provided, save the figure to this path
    """
    # Make sure sizes are up-to-date
    hierarchy.update_all_sizes()
    
    # Set up figure
    plt.figure(figsize=figsize)
    
    # Plot tree recursively
    def plot_node(node, x, y, width, height=1.0):
        if node.is_leaf():
            # Plot leaf node
            color = 'gray'
            if color_ids is not None:
                # Find which color group this leaf belongs to
                for color_idx, ids in enumerate(color_ids):
                    if node.id in ids:
                        color = ['blue', 'red', 'green', 'purple', 'orange'][min(color_idx, 4)]
                        break
                        
            plt.plot(x, 0, 'o', color=color, markersize=8)
            plt.text(x, -0.3, str(node.id), ha='center', fontsize=8)
            return
            
        # For internal nodes, split the width among children
        child_count = len(node.children)
        if child_count == 0:
            return
            
        child_width = width / child_count
        
        # Calculate positions for children
        positions = []
        start = x - width/2 + child_width/2
        for i in range(child_count):
            positions.append(start + i * child_width)
            
        # Draw horizontal line connecting children
        plt.plot([positions[0], positions[-1]], [height, height], 'k-')
        
        # Recursively plot children
        for i, child in enumerate(node.children):
            child_x = positions[i]
            # Draw vertical line to child
            child_height = height - 1.0
            if not child.is_leaf():
                plt.plot([child_x, child_x], [height, child_height], 'k-')
                
            # Plot child subtree
            plot_node(child, child_x, child_height, child_width, child_height)
    
    # Start plotting from root
    max_height = hierarchy.compute_height()
    root_y = max_height
    plot_node(hierarchy.root, 0, root_y, 10, root_y)
    
    # Add a legend if needed
    if color_ids:
        color_map = ['blue', 'red', 'green', 'purple', 'orange']
        legend_elements = []
        for i in range(min(len(color_ids), len(color_map))):
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i],
                      markersize=10, label=f'Color {i}')
            )
        plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title and labels
    plt.title(title)
    plt.xlabel('Data Points')
    plt.ylabel('Height')
    
    # Remove axis ticks
    plt.xticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_dendrogram_with_scipy(
    points: np.ndarray,
    color_ids: Optional[List[List[int]]] = None,
    title: str = "Hierarchical Clustering Dendrogram",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a dendrogram using scipy's functions directly.
    This avoids conversion issues by using the scipy linkage result directly.
    
    Args:
        points: The original data points
        color_ids: Lists of point indices for each color category
        title: Plot title
        figsize: Figure size
        save_path: If provided, save the figure to this path
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    
    # Compute linkage matrix directly
    Z = linkage(points, method='average')
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create color mapping for leaf labels if color_ids provided
    leaf_colors = None
    if color_ids is not None:
        n_samples = points.shape[0]
        
        # Create color map
        color_map = {0: 'blue', 1: 'red', 2: 'green', 3: 'purple', 4: 'orange'}
        
        # Assign colors to points
        point_colors = ['gray'] * n_samples
        for color_idx, ids in enumerate(color_ids):
            color = color_map.get(color_idx, f'C{color_idx}')
            for id in ids:
                if 0 <= id < n_samples:
                    point_colors[id] = color
    
    # Plot dendrogram
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        link_color_func=lambda k: 'k',  # black links
    )
    
    # If we have color mappings, add colored dots for the leaves
    if color_ids:
        ax = plt.gca()
        xlbls = ax.get_xticklabels()
        
        # Extract original indices from labels
        leaf_indices = [int(lbl.get_text()) for lbl in xlbls]
        
        # Add colored dots
        for i, idx in enumerate(leaf_indices):
            x = float(xlbls[i].get_position()[0])
            color = 'gray'
            for color_idx, ids in enumerate(color_ids):
                if idx in ids:
                    color = color_map.get(color_idx, f'C{color_idx}')
                    break
            plt.plot(x, 0, 'o', color=color, markersize=8)
        
        # Add a legend for colors
        legend_elements = []
        for i in range(len(color_ids)):
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map.get(i, f'C{i}'),
                      markersize=10, label=f'Color {i}')
            )
        plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title and labels
    plt.title(title)
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
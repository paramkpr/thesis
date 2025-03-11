"""
tests/test_data_synthetic.py

Generates some data... okay no I'm too lazy to write this. 

"""

import matplotlib.pyplot as plt
from data.synthetic import generate_colored_data
from utils.distance import calculate_distance_matrix, convert_to_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from algorithms.vanilla import average_linkage

points, color_ids = generate_colored_data(
    total_points=128,
    color_proportions=[0.3, 0.7],  # 30% blue, 70% red
    cluster_std=2,
    seed=42
)


distance_matrix = calculate_distance_matrix(points)
similarity_matrix = convert_to_similarity(distance_matrix)
print(f"Distance Matrix Shape: {distance_matrix.shape} \n", distance_matrix)
print(f"Similarity Matrix Shape: {similarity_matrix.shape} \n", similarity_matrix)

blues = points[color_ids[0][0]:color_ids[0][-1]]
reds = points[color_ids[1][0]:color_ids[1][-1]]


plt.scatter(blues[:, 0], blues[:, 1], c='red', cmap='viridis', s=100, edgecolors='k')
plt.scatter(reds[:, 0], reds[:, 1], c='blue', cmap='viridis', s=100, edgecolors='k')

Z = linkage(points, 'average')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)


binary_Z = linkage(distance_matrix, 'average')
fig_bin = plt.figure(figsize=(25, 10))
binary_dn = dendrogram(binary_Z)

average_linkage = average_linkage(similarity_matrix)


plt.show()


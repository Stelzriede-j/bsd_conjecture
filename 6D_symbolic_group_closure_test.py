
# BSD Phase 6D: Symbolic Group Law Closure Test
# Author: Jacob Stelzriede (with OpenAI)
# April 2025

"""
This script analyzes symbolic triplets detected from centroid positions
and tests for consistency of symbolic group law closure.

Specifically, it checks:
- Whether triplet-defined addition rules are globally consistent
- If chaining symbolic additions (gi + gj = gk, gj + gk = gm) ever leads to conflict

Results indicate symbolic group law integrity under twist-compression dynamics.
"""

import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist, squareform

# === TUNING PARAMETERS ===
Nx, Ny = 200, 200         # Grid resolution
freq = 8                # Twist injection frequency
closure_epsilon = 0.03    # Closure threshold for distance match
lock_zone_threshold = 0.2 # Minimum field amplitude for zone detection
# ==========================

x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Simulate twist field
field = np.sin(2 * np.pi * freq * R)
mask = np.abs(field) > lock_zone_threshold
labeled_array, num_features = label(mask)
raw_centroids = center_of_mass(mask, labeled_array, range(1, num_features + 1))
centroids = [(x / Nx, y / Ny) for x, y in raw_centroids]

# Build symbolic triplets
triplets = set()
D = squareform(pdist(centroids))
for i in range(len(centroids)):
    for j in range(i + 1, len(centroids)):
        for k in range(j + 1, len(centroids)):
            if abs(D[i, j] + D[j, k] - D[i, k]) < closure_epsilon:
                triplets.add(tuple(sorted((i, j, k))))

# Check for symbolic closure conflicts
group_map = {}
conflict_count = 0
conflicts = []

for i, j, k in triplets:
    key1 = tuple(sorted((i, j)))
    key2 = tuple(sorted((j, k)))
    if key1 in group_map and group_map[key1] != k:
        conflict_count += 1
        conflicts.append((key1, group_map[key1], k))
    if key2 in group_map and group_map[key2] != i:
        conflict_count += 1
        conflicts.append((key2, group_map[key2], i))
    group_map[key1] = k
    group_map[key2] = i

# Output results
print(f"Total symbolic triplets: {len(triplets)}")
print(f"Total symbolic group law conflicts detected: {conflict_count}")
print("Sample conflicts:")
for idx, (pair, expected, actual) in enumerate(conflicts[:10]):
    print(f"{idx+1}. {pair} → expected: {expected}, but also got: {actual}")


# BSD Phase 6E: Symbolic Group Law Conflict Sweep with Logic Toggle
# Author: Jacob Stelzriede (with OpenAI)
# April 2025

"""
This script sweeps over twist frequencies and checks symbolic group law behavior.
It includes a toggle for unidirectional vs bidirectional conflict checking.

Outputs:
- Frequency
- Triplet count
- Conflict count
- Coherence score = triplets / (1 + conflicts)
"""

import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist, squareform

# === TUNING PARAMETERS ===
Nx, Ny = 200, 200
closure_epsilon = 0.03
lock_zone_threshold = 0.2
freq_range = range(1, 40)
check_bidirectional = True  # <-- Toggle this for deeper group law validation
# ==========================

x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

triplet_data = []

for freq in freq_range:
    field = np.sin(2 * np.pi * freq * R)
    mask = np.abs(field) > lock_zone_threshold
    labeled_array, num_features = label(mask)
    raw_centroids = center_of_mass(mask, labeled_array, range(1, num_features + 1))
    centroids = [(x / Nx, y / Ny) for x, y in raw_centroids]

    triplets = set()
    D = squareform(pdist(centroids))
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            for k in range(j + 1, len(centroids)):
                if abs(D[i, j] + D[j, k] - D[i, k]) < closure_epsilon:
                    triplets.add(tuple(sorted((i, j, k))))

    group_map = {}
    conflicts = 0
    for i, j, k in triplets:
        key1 = tuple(sorted((i, j)))
        if key1 in group_map and group_map[key1] != k:
            conflicts += 1
        group_map[key1] = k

        if check_bidirectional:
            key2 = tuple(sorted((j, k)))
            if key2 in group_map and group_map[key2] != i:
                conflicts += 1
            group_map[key2] = i

    triplet_count = len(triplets)
    coherence_score = triplet_count / (1 + conflicts)
    triplet_data.append((freq, triplet_count, conflicts, coherence_score))

# Output results
print("Freq | Triplets | Conflicts | Coherence Score")
print("-----|----------|-----------|----------------")
for freq, t, c, s in triplet_data:
    print(f"{freq:>4} | {t:>8} | {c:>9} | {s:>16.3f}")

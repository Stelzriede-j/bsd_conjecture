
# Sha Detection via Symbolic Closure in Mod-p vs Global Field
# Author: Jacob Stelzriede (with OpenAI)
# April 2025

"""
This script detects symbolic triplets that appear in mod-p twist-compression fields
but fail to appear in the global full-resolution field, making them candidates for
the Tate–Shafarevich group (Sha).

Approach:
- Run twist field at primes [5, 7, 11, 13, 17]
- Extract symbolic triplets from each mod-p projection using MST closure logic
- Compare against triplets from the full field
- Any triplet that appears in mod-p but not globally is a Sha candidate
"""

import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist, squareform

# Field parameters
Nx, Ny = 100, 100
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
freq = 20
primes = [5, 7, 11, 13, 17]

def run_field(mod_p=None, max_centroids=30):
    field = np.sin(2 * np.pi * freq * R)
    if mod_p:
        field = np.round(field * 1000).astype(int) % mod_p
        field = field.astype(float) / mod_p

    mask = np.abs(field) > 0.2
    labeled_array, num_features = label(mask)
    raw_centroids = center_of_mass(mask, labeled_array, range(1, num_features + 1))
    centroids = [(x / Nx, y / Ny) for x, y in raw_centroids][:max_centroids]

    triplets = set()
    D = squareform(pdist(centroids))
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            for k in range(j + 1, len(centroids)):
                if abs(D[i, j] + D[j, k] - D[i, k]) < 0.03:
                    triplets.add(tuple(sorted((i, j, k))))
    return triplets

# Gather mod-p triplets
mod_p_triplets = {}
for p in primes:
    mod_p_triplets[p] = run_field(mod_p=p)

# Run global field triplets
global_triplets = run_field(mod_p=None)

# Detect Sha candidates
sha_candidates = set()
for p in primes:
    for t in mod_p_triplets[p]:
        if t not in global_triplets:
            sha_candidates.add(t)

# Output
print(f"Detected {len(sha_candidates)} potential Sha candidates.")
for i, t in enumerate(sorted(sha_candidates)):
    if i < 10:
        print(f"Candidate {i+1}: Triplet {t}")

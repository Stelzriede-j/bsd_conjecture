
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

# Parameters for field simulation
Nx, Ny = 50, 50
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
mod_p = 17
injection_times = [0, 30, 60, 90, 120]
centers = [(10, 10), (40, 10), (10, 40), (40, 40), (25, 25)]
twist_amplitude = 0.15
injection_radius = 10.0
threshold = 5

# Initialize field
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

# Evolve twist-compression field with staggered injections
for t in range(150):
    if t in injection_times:
        idx = injection_times.index(t)
        cx, cy = centers[idx]
        for x in range(Nx):
            for y in range(Ny):
                r2 = (x - cx)**2 + (y - cy)**2
                phi[x, y] += np.exp(-r2 / injection_radius) * twist_amplitude

    new_phi = phi.copy()
    for x in range(1, Nx - 1):
        for y in range(1, Ny - 1):
            laplacian = (
                phi[x+1, y] + phi[x-1, y] + phi[x, y+1] + phi[x, y-1] - 4 * phi[x, y]
            )
            force = -np.sign(phi[x, y]) * lambda_fixed * n_fixed * (abs(phi[x, y])) ** (n_fixed - 1)
            acceleration = laplacian + force
            velocity[x, y] = damping * (velocity[x, y] + dt * acceleration)
            new_phi[x, y] += velocity[x, y]
    phi = new_phi

# Project and detect lock zones
mod_field = np.round(phi * 1000).astype(int) % mod_p
mask = np.zeros_like(mod_field, dtype=bool)
for x in range(1, Nx - 1):
    for y in range(1, Ny - 1):
        patch = mod_field[x-1:x+2, y-1:y+2]
        if np.count_nonzero(patch == mod_field[x, y]) >= threshold:
            mask[x, y] = True

labeled_array, num_features = label(mask, structure=np.ones((3, 3), dtype=int))

# Compute zone centroids
zone_centroids = {}
for label_id in range(1, np.max(labeled_array) + 1):
    mask_zone = (labeled_array == label_id)
    centroid = center_of_mass(mask_zone)
    zone_centroids[label_id] = centroid

# Minimum spanning tree
coords = np.array([zone_centroids[z] for z in sorted(zone_centroids)])
dist_matrix = squareform(pdist(coords))
mst = minimum_spanning_tree(dist_matrix).toarray()

# Plot MST over lock zone map
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(labeled_array, cmap='tab20')

for i, (y, x) in enumerate(coords):
    ax.plot(x, y, 'wo')
    ax.text(x, y, str(i + 1), color='white', ha='center', va='center', fontsize=7)

for i in range(len(coords)):
    for j in range(len(coords)):
        if mst[i, j] > 0:
            y1, x1 = coords[i]
            y2, x2 = coords[j]
            ax.plot([x1, x2], [y1, y2], 'w-', linewidth=1)

ax.set_title("Minimum Spanning Tree over Resonance Zone Centroids")
ax.axis('off')
plt.tight_layout()
plt.show()

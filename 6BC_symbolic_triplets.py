
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

# Parameters
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

# Field initialization
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

# Field evolution with injections
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

# Lock zone detection
mod_field = np.round(phi * 1000).astype(int) % mod_p
mask = np.zeros_like(mod_field, dtype=bool)
for x in range(1, Nx - 1):
    for y in range(1, Ny - 1):
        patch = mod_field[x-1:x+2, y-1:y+2]
        if np.count_nonzero(patch == mod_field[x, y]) >= threshold:
            mask[x, y] = True

labeled_array, num_features = label(mask, structure=np.ones((3, 3), dtype=int))

# Compute centroids and MST
zone_centroids = {}
for label_id in range(1, np.max(labeled_array) + 1):
    mask_zone = (labeled_array == label_id)
    centroid = center_of_mass(mask_zone)
    zone_centroids[label_id] = centroid

coords = np.array([zone_centroids[k] for k in sorted(zone_centroids)])
dist_matrix = squareform(pdist(coords))
mst = minimum_spanning_tree(dist_matrix).toarray()

# Extract symbolic triplets from top 5 zones by area
zone_areas = {label_id: np.sum(labeled_array == label_id) for label_id in zone_centroids}
top_ids = sorted(zone_areas, key=zone_areas.get, reverse=True)[:5]
top_coords = {i: zone_centroids[i] for i in top_ids}

symbolic_operations = []
for i in top_ids:
    for j in top_ids:
        for k in top_ids:
            if len(set([i, j, k])) < 3:
                continue
            pi = np.array(top_coords[i])
            pj = np.array(top_coords[j])
            pk = np.array(top_coords[k])
            dij = np.linalg.norm(pi - pj)
            djk = np.linalg.norm(pj - pk)
            dik = np.linalg.norm(pi - pk)
            if np.abs(dij + djk - dik) < 1.0:
                symbolic_operations.append(f"g_{i} + g_{j} = g_{k}")

symbolic_operations = sorted(set(symbolic_operations))

print("Symbolic Triplets:")
for op in symbolic_operations:
    print(op)

#~~
triplets = []
for op in symbolic_operations:
    parts = op.replace('g_', '').replace('=', '+').split('+')
    if len(parts) == 3:
        i, j, k = map(int, parts)
        triplets.append((i - 1, j - 1, k - 1))  # adjust to 0-based indexing

# Save centroids
np.save("centroids.npy", np.array([zone_centroids[k] for k in sorted(zone_centroids)]))

# Save triplets
with open("symbolic_triplets.json", "w") as f:
    json.dump(triplets, f)
#~~

# Plot MST + symbolic triplets
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

for op in symbolic_operations:
    parts = op.replace('g_', '').replace('=', '+').split('+')
    if len(parts) == 3:
        i, j, k = map(int, parts)
        pi = np.array(zone_centroids[i])[::-1]
        pj = np.array(zone_centroids[j])[::-1]
        pk = np.array(zone_centroids[k])[::-1]
        ax.plot([pi[0], pj[0], pk[0]], [pi[1], pj[1], pk[1]], 'r-', linewidth=2, alpha=0.7)

ax.set_title("Phase 6C: Symbolic Triplets + MST from Live Twist Field")
ax.axis('off')
plt.tight_layout()
plt.show()

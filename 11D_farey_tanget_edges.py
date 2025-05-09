#FC4 Farey Tangent Edges



from fractions import Fraction
from scipy.ndimage import center_of_mass, label
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np

# Parameters
Nx, Ny = 50, 50
mod_p = 17
threshold = 5
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
injection_times = [0, 30, 60, 90, 120]
centers = [(10, 10), (40, 10), (10, 40), (40, 40), (25, 25)]
twist_amplitude = 0.15
injection_radius = 10.0

# Field evolution
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))
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

# Zone analysis
zone_centroids = {
    label_id: center_of_mass(labeled_array == label_id)
    for label_id in range(1, np.max(labeled_array) + 1)
}
zone_areas = {
    label_id: np.sum(labeled_array == label_id)
    for label_id in zone_centroids
}
top_ids = sorted(zone_areas, key=zone_areas.get, reverse=True)[:5]
top_coords = {i: zone_centroids[i] for i in top_ids}

# Rational fitting
fitted_rationals = {}
for i in top_ids:
    x_norm = zone_centroids[i][1] / Nx
    f = Fraction(x_norm).limit_denominator(20)
    fitted_rationals[i] = {'p': f.numerator, 'q': f.denominator}

# MST construction
coords = np.array([zone_centroids[i] for i in top_ids])
dist_matrix = squareform(pdist(coords))
mst_matrix = minimum_spanning_tree(dist_matrix).toarray()

# Farey tangency check
def is_farey_neighbor(p1, q1, p2, q2):
    return abs(p1 * q2 - p2 * q1) == 1

mst_edges = []
farey_edges = []
for i in range(len(coords)):
    for j in range(len(coords)):
        if mst_matrix[i, j] > 0:
            zone_i = top_ids[i]
            zone_j = top_ids[j]
            mst_edges.append((zone_i, zone_j))
            pi, qi = fitted_rationals[zone_i]['p'], fitted_rationals[zone_i]['q']
            pj, qj = fitted_rationals[zone_j]['p'], fitted_rationals[zone_j]['q']
            if is_farey_neighbor(pi, qi, pj, qj):
                farey_edges.append((zone_i, zone_j))

print("MST Edges:", mst_edges)
print("Farey-Aligned MST Edges:", farey_edges)
print(f"{len(farey_edges)} of {len(mst_edges)} edges are Farey neighbors")


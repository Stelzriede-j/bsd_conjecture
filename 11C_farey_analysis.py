
# FC3 - Farey Analysis


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

# Zone properties
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

# Symbolic triplet extraction
triplets = []
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
                triplets.append((i, j, k))
triplets = sorted(set(triplets))

# Rational fit for Farey check
def fit_rational(x, max_q=20):
    f = Fraction(x).limit_denominator(max_q)
    return f.numerator, f.denominator

# Rational fitting for x-coordinates
fitted_rationals = {}
for i in top_ids:
    x_norm = zone_centroids[i][1] / Nx
    p, q = fit_rational(x_norm)
    fitted_rationals[i] = {'p': p, 'q': q}

# Farey check
def is_farey_neighbor(p1, q1, p2, q2):
    return abs(p1 * q2 - p2 * q1) == 1

farey_triplets = []
for i, j, k in triplets:
    pi, qi = fitted_rationals[i]['p'], fitted_rationals[i]['q']
    pj, qj = fitted_rationals[j]['p'], fitted_rationals[j]['q']
    pk, qk = fitted_rationals[k]['p'], fitted_rationals[k]['q']
    if (is_farey_neighbor(pi, qi, pj, qj) or
        is_farey_neighbor(pj, qj, pk, qk) or
        is_farey_neighbor(pi, qi, pk, qk)):
        farey_triplets.append((i, j, k))

import pandas as pd
df = pd.DataFrame(farey_triplets, columns=['i', 'j', 'k'])
print("Farey-aligned symbolic triplets:")
print(df)


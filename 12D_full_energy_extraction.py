
# Phase 12D: Extract resonance zone energies for FC2 (q^2 scaling)

import numpy as np
from sympy import symbols, Eq, solve, Rational
from scipy.ndimage import label, center_of_mass

# Setup
x, y = symbols('x y')
curve = Eq(y**2, x**3 - 4*x + 4)
x_vals = [Rational(n, 1) for n in range(-6, 7)]
Nx, Ny = 50, 50
mod_p = 17
threshold = 5
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 3.5
scale = 6
offset_x, offset_y = 25, 25
twist_amplitude = 0.15
injection_radius = 4.0
injection_times = [0, 30, 60, 90, 120]

# Rational points → grid centers
rational_points = []
for x_val in x_vals:
    rhs = curve.rhs.subs(x, x_val)
    y_solutions = solve(Eq(y**2, rhs), y)
    for y_val in y_solutions:
        if y_val.is_rational:
            rational_points.append((float(x_val), float(y_val)))

seed_centers = [(int(offset_x + scale * xp), int(offset_y - scale * yp)) for xp, yp in rational_points]
centers = list({(x, y) for x, y in seed_centers if 0 <= x < Nx and 0 <= y < Ny})[:5]

# Field evolution
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))
for t in range(150):
    if t in injection_times and t // 30 < len(centers):
        cx, cy = centers[t // 30]
        for i in range(Nx):
            for j in range(Ny):
                r2 = (i - cx)**2 + (j - cy)**2
                phi[i, j] += np.exp(-r2 / injection_radius) * twist_amplitude

    new_phi = phi.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            laplacian = (
                phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - 4 * phi[i, j]
            )
            force = -np.sign(phi[i, j]) * lambda_fixed * n_fixed * abs(phi[i, j])**(n_fixed - 1)
            acceleration = laplacian + force
            velocity[i, j] = damping * (velocity[i, j] + dt * acceleration)
            new_phi[i, j] += velocity[i, j]
    phi = new_phi

# Lock zone detection
mod_field = np.round(phi * 1000).astype(int) % mod_p
mask = np.zeros_like(mod_field, dtype=bool)
for i in range(1, Nx - 1):
    for j in range(1, Ny - 1):
        patch = mod_field[i-1:i+2, j-1:j+2]
        if np.count_nonzero(patch == mod_field[i, j]) >= threshold:
            mask[i, j] = True

labeled_array, num_features = label(mask, structure=np.ones((3, 3), dtype=int))

# Zone energy extraction
zone_energies = []
zone_centroids = {}
for label_id in range(1, np.max(labeled_array) + 1):
    mask = (labeled_array == label_id)
    energy = 0.5*np.sum(phi[mask]**2)
    zone_energies.append(energy)
    zone_centroids[label_id] = center_of_mass(mask)

# Output results
for idx, (zid, centroid) in enumerate(zone_centroids.items()):
    print(f"Zone {zid}: Centroid = {centroid}, Energy = {zone_energies[idx]:.4f}")

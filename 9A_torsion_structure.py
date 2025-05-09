
# Phase 9A: Torsion curve twist field evolution and symbolic analysis

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from scipy.optimize import minimize
from sympy.abc import x, y
from sympy import Eq, lambdify, latex

# Parameters
Nx, Ny = 50, 50
mod_p = 17
threshold = 5
twist_amplitude = 0.125
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
injection_radius = 10.0
injection_times = [0, 30, 60]
torsion_centers = [(13, 25), (25, 25), (31, 25)]

# Initialize twist field
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

for t in range(150):
    if t in injection_times and t // 30 < len(torsion_centers):
        cx, cy = torsion_centers[t // 30]
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
zone_centroids = {label: center_of_mass(labeled_array == label)
                  for label in range(1, num_features + 1)}
coords_xy = np.array([(c[1], c[0]) for c in zone_centroids.values()])

# Curve fit
def curve_error(params, coords):
    a, b = params
    return sum((yi**2 - (xi**3 + a*xi + b))**2 for xi, yi in coords)

result = minimize(curve_error, (0.0, 0.0), args=(coords_xy,))
a_fit, b_fit = result.x

# Ensure symbolic x, y
from sympy.abc import x, y
curve_eq = Eq(y**2, x**3 + a_fit * x + b_fit)
curve_str = f"y^2 = x^3 + ({a_fit:.4f})x + ({b_fit:.4f})"
curve_latex = latex(curve_eq)
curve_fn = lambdify(x, x**3 + a_fit * x + b_fit, 'numpy')

# Plot field resonance and curve echo
x_vals = np.linspace(0, 50, 400)
y_vals = np.sqrt(np.clip(curve_fn(x_vals), 0, None))

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='Fitted Curve', color='black')
plt.plot(x_vals, -y_vals, color='black')
plt.scatter(coords_xy[:, 0], coords_xy[:, 1], color='crimson', label='Resonance Zones')
plt.title("Phase 9A: Torsion Curve → Field → Symbolic Echo")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Symbolic triplet extraction
zone_areas = {label: np.sum(labeled_array == label) for label in zone_centroids}
top_ids = sorted(zone_areas, key=zone_areas.get, reverse=True)[:5]
top_coords = {i: zone_centroids[i] for i in top_ids}
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
            if abs(dij + djk - dik) < 1.0:
                triplets.append(f"g_{i} + g_{j} = g_{k}")

# Final Output
print("Recovered curve:")
print(curve_str)
print("LaTeX curve:")
print(curve_latex)
print(f"Total lock zones: {num_features}")
print(f"Symbolic triplets: {sorted(set(triplets))}")

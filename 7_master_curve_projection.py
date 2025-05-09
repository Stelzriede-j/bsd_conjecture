
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, lambdify, latex
from sympy.core.numbers import Rational
from scipy.optimize import minimize
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

# --- CONFIG ---
curve_id = "y^2 = x^3 - 4x + 4"
x, y = symbols('x y')
seed_curve = Eq(y**2, x**3 - 4*x + 4)
x_vals = [Rational(n, 1) for n in [-2, -1, 0, 1, 2, 3, 4]]
Nx, Ny = 50, 50
twist_amplitude = 0.125
mod_p = 17
threshold = 5
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
injection_times = [0, 30, 60, 90, 120]
scale = 6
offset_x, offset_y = 25, 25
injection_radius = 10.0

# --- Step 1: Seed rational points from curve ---
rational_points = []
for x_val in x_vals:
    rhs = x_val**3 - 4*x_val + 4
    y_solutions = solve(Eq(y**2, rhs), y)
    for y_val in y_solutions:
        if y_val.is_rational:
            rational_points.append((float(x_val), float(y_val)))

centers = [(int(offset_x + scale * x), int(offset_y - scale * y)) for x, y in rational_points]
centers = list({(x, y) for x, y in centers if 0 <= x < Nx and 0 <= y < Ny})[:5]

# --- Step 2: Field evolution ---
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

for t in range(150):
    if t in injection_times and t // 30 < len(centers):
        cx, cy = centers[t // 30]
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
            force = -np.sign(phi[x, y]) * lambda_fixed * n_fixed * abs(phi[x, y])**(n_fixed - 1)
            acceleration = laplacian + force
            velocity[x, y] = damping * (velocity[x, y] + dt * acceleration)
            new_phi[x, y] += velocity[x, y]
    phi = new_phi

# --- Step 3: Lock zone detection ---
mod_field = np.round(phi * 1000).astype(int) % mod_p
mask = np.zeros_like(mod_field, dtype=bool)
for x in range(1, Nx - 1):
    for y in range(1, Ny - 1):
        patch = mod_field[x-1:x+2, y-1:y+2]
        if np.count_nonzero(patch == mod_field[x, y]) >= threshold:
            mask[x, y] = True

labeled_array, num_features = label(mask, structure=np.ones((3, 3), dtype=int))
zone_centroids = {label: center_of_mass(labeled_array == label)
                  for label in range(1, num_features + 1)}
coords_xy = np.array([(c[1], c[0]) for c in zone_centroids.values()])
dist_matrix = squareform(pdist(coords_xy))
mst = minimum_spanning_tree(dist_matrix).toarray()

# --- Step 4: Symbolic triplet logic ---
zone_areas = {label: np.sum(labeled_array == label) for label in zone_centroids}
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
            if abs(dij + djk - dik) < 1.0:
                symbolic_operations.append(f"g_{i} + g_{j} = g_{k}")
symbolic_operations = sorted(set(symbolic_operations))

# --- Step 5: Elliptic curve fit to all lock zones ---
def curve_error(params, coords):
    a, b = params
    return sum((yi**2 - (xi**3 + a*xi + b))**2 for xi, yi in coords)

result = minimize(curve_error, (0.0, 0.0), args=(coords_xy,))
a_fit, b_fit = result.x
from sympy.abc import x, y
curve_eq_fitted = Eq(y**2, x**3 + a_fit * x + b_fit)
curve_str = f"y^2 = x^3 + ({a_fit:.4f})x + ({b_fit:.4f})"
curve_latex = latex(curve_eq_fitted)
from sympy.abc import x, y
curve_fn = lambdify(x, x**3 + a_fit * x + b_fit, 'numpy')

# --- Step 6: Plotting ---
x_vals_plot = np.linspace(0, 50, 400)
y_vals = np.sqrt(np.clip(curve_fn(x_vals_plot), 0, None))

plt.figure(figsize=(10, 6))
plt.plot(x_vals_plot, y_vals, label='Fitted Curve', color='black')
plt.plot(x_vals_plot, -y_vals, color='black')
plt.scatter(coords_xy[:, 0], coords_xy[:, 1], color='crimson', label='Resonance Generators')
plt.title(f"Phase 7 Master: Curve Echo from {curve_id}")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 7: Summary Output ---
# Print results
print("Recovered curve:")
print (f"y^2 = x^3 + ({a_fit:.4f})x + ({b_fit:.4f})")

curve_eq = Eq(y**2, x**3 + a_fit * x + b_fit)
curve_str = str(curve_eq)
print("Symbolic curve string:\n", curve_str)

print("LaTeX version:")
print(curve_latex)

print("Symbolic triplets:", symbolic_operations)
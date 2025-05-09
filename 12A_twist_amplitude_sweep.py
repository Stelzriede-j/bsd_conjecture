# Re-run Phase 12A: Sweep twist_amplitude with n = 3.5 (after kernel reset)

import numpy as np
import pandas as pd
from sympy import Rational, symbols, Eq, solve
from scipy.optimize import minimize
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist, squareform

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
injection_radius = 4.0
injection_times = [0, 30, 60, 90, 120]
twist_amplitudes = [0.05, 0.10, 0.125, 0.15, 0.20]

# Generate injection points
rational_points = []
for x_val in x_vals:
    rhs = curve.rhs.subs(x, x_val)
    y_solutions = solve(Eq(y**2, rhs), y)
    for y_val in y_solutions:
        if y_val.is_rational:
            rational_points.append((float(x_val), float(y_val)))

seed_centers = [(int(offset_x + scale * xp), int(offset_y - scale * yp)) for xp, yp in rational_points]
centers = list({(x, y) for x, y in seed_centers if 0 <= x < Nx and 0 <= y < Ny})[:5]

results = []

for amp in twist_amplitudes:
    phi = np.zeros((Nx, Ny))
    velocity = np.zeros((Nx, Ny))

    for t in range(150):
        if t in injection_times and t // 30 < len(centers):
            cx, cy = centers[t // 30]
            for i in range(Nx):
                for j in range(Ny):
                    r2 = (i - cx)**2 + (j - cy)**2
                    phi[i, j] += np.exp(-r2 / injection_radius) * amp

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

    def curve_error(params, coords):
        a, b = params
        return sum((yi**2 - (xi**3 + a*xi + b))**2 for xi, yi in coords)

    result = minimize(curve_error, (0.0, 0.0), args=(coords_xy,))
    a_fit, b_fit = result.x

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

    results.append({
        "Amplitude": amp,
        "Lock Zones": num_features,
        "Triplets": len(set(triplets)),
        "a_fit": round(a_fit, 4),
        "b_fit": round(b_fit, 4)
    })


df_sweep = pd.DataFrame(results)
print(df_sweep)
df_sweep.to_csv("phase12a_twist_amplitude_sweep.csv", index=False)
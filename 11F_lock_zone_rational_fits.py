# FC6: Lock Zone q-value Histogram from Rational Fits

from fractions import Fraction
from scipy.ndimage import center_of_mass, label
import matplotlib.pyplot as plt
import numpy as np

# Parameters
Nx, Ny = 50, 50
mod_p = 17
threshold = 5
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
injection_times = [0, 50, 100]
centers = [(10, 10), (40, 10), (25, 40)]
twist_amplitude = 0.15
injection_radius = 10.0

# Initialize fields
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

# Evolution loop
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
            force = -np.sign(phi[x, y]) * lambda_fixed * n_fixed * abs(phi[x, y]) ** (n_fixed - 1)
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

# Extract centroids and q-values
centroids = center_of_mass(mask, labeled_array, range(1, num_features + 1))
lock_zone_x = [x / Nx for x, _ in centroids]
q_values = [Fraction(x).limit_denominator(20).denominator for x in lock_zone_x]

# Plot q histogram
plt.figure(figsize=(6, 4))
plt.hist(q_values, bins=range(1, max(q_values)+2), edgecolor='black', align='left')
plt.xlabel("Denominator q (from p/q fit)")
plt.ylabel("Number of Lock Zones")
plt.title("FC6: Lock Zone Count vs. Rational Denominator q")
plt.grid(True)
plt.tight_layout()
plt.show()

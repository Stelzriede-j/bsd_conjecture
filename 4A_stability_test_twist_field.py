
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Parameters
Nx, Ny = 50, 50
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
mod_p = 17
injection_times = [0, 50, 100]
centers = [(10, 10), (40, 10), (25, 40)]
twist_amplitude = 0.15
injection_radius = 10.0
threshold = 5

# New random seed (implicit)
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

# Evolve with dynamic twist injection
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

# Analyze mod p and resonance mask
mod_field = np.round(phi * 1000).astype(int) % mod_p
mask = np.zeros_like(mod_field, dtype=bool)
for x in range(1, Nx - 1):
    for y in range(1, Ny - 1):
        patch = mod_field[x-1:x+2, y-1:y+2]
        if np.count_nonzero(patch == mod_field[x, y]) >= threshold:
            mask[x, y] = True

# Label resonance zones
labeled_array, num_features = label(mask, structure=np.ones((3, 3), dtype=int))

# Energy density
energy_density = 0.5 * velocity**2 + lambda_fixed * (np.abs(phi))**n_fixed

# Plot output
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(energy_density, cmap='inferno')
axs[0].set_title("Twist Field Energy Density (Shaken)")
axs[0].axis('off')
axs[1].imshow(labeled_array, cmap='tab20')
axs[1].set_title(f"Resonance Lock Zones (New Seed): {num_features}")
axs[1].axis('off')
plt.tight_layout()
plt.show()

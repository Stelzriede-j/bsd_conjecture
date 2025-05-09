import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.stats import entropy as spectral_entropy
from scipy.ndimage import label

# Parameters
Nx, Ny = 50, 50
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
mod_p = 17
injection_times = [0, 50, 100]  # Twist injections at different times
centers = [(10, 10), (40, 10), (25, 40)]  # Spatial centers for twist seeds
twist_amplitude = 0.15
injection_radius = 10.0
threshold = 5  # 3x3 match count to register a local resonance

# Initialize twist field and velocity
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

# Evolution loop with timed twist injections
for t in range(150):
    if t in injection_times:
        idx = injection_times.index(t)
        cx, cy = centers[idx]
        for x in range(Nx):
            for y in range(Ny):
                r2 = (x - cx)**2 + (y - cy)**2
                phi[x, y] += np.exp(-r2 / injection_radius) * twist_amplitude

    # Evolve field
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

# Project to mod p space
mod_field = np.round(phi * 1000).astype(int) % mod_p

# Lock region detection via local pattern alignment
mask = np.zeros_like(mod_field, dtype=bool)
for x in range(1, Nx - 1):
    for y in range(1, Ny - 1):
        patch = mod_field[x-1:x+2, y-1:y+2]
        if np.count_nonzero(patch == mod_field[x, y]) >= threshold:
            mask[x, y] = True

# Label lock zones
labeled_array, num_features = label(mask, structure=np.ones((3, 3), dtype=int))

# Plot twist field and lock zones
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(mod_field, cmap='viridis')
axs[0].set_title("Field mod 17 (Dynamic Injection)")
axs[0].axis('off')
axs[1].imshow(labeled_array, cmap='tab20')
axs[1].set_title(f"Detected Lock Zones: {num_features}")
axs[1].axis('off')
plt.tight_layout()
plt.show()

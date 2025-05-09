
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Parameters
Nx, Ny = 50, 50
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
mod_p = 3
injection_times = [0, 50, 100]
centers = [(10, 10), (40, 10), (25, 40)]
twist_amplitude = 0.15
injection_radius = 10.0
threshold = 5

# Initialize field
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

# Evolve field with dynamic twist injection
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

# Compute mod field and lock mask
mod_field = np.round(phi * 1000).astype(int) % mod_p
mask = np.zeros_like(mod_field, dtype=bool)
for x in range(1, Nx - 1):
    for y in range(1, Ny - 1):
        patch = mod_field[x-1:x+2, y-1:y+2]
        if np.count_nonzero(patch == mod_field[x, y]) >= threshold:
            mask[x, y] = True

# Label lock zones
labeled_array, num_features = label(mask, structure=np.ones((3, 3), dtype=int))

# Energy density from twist-compression model
energy_density = 0.5 * velocity**2 + lambda_fixed * (np.abs(phi))**n_fixed

# Plot overlay
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(energy_density, cmap='inferno')
axs[0].set_title("Field Energy Density")
axs[0].axis('off')
axs[1].imshow(labeled_array, cmap='tab20')
axs[1].set_title(f"Resonance Lock Zones (Count = {num_features})")
axs[1].axis('off')
plt.tight_layout()
plt.show()

# Plot energy per resonance zone
zone_energies = {}
for label_id in range(1, num_features + 1):
    mask_zone = (labeled_array == label_id)
    total_energy = np.sum(energy_density[mask_zone])
    zone_energies[label_id] = total_energy

sorted_zones = sorted(zone_energies.items(), key=lambda x: x[1], reverse=True)
zone_ids = [z[0] for z in sorted_zones]
energies = [z[1] for z in sorted_zones]

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(energies) + 1), energies, color='crimson')
plt.xlabel("Lock Zone ID (sorted by energy)")
plt.ylabel("Total Term Energy")
plt.title("Term Energy per Resonant Lock Zone")
plt.grid(True)
plt.tight_layout()
plt.show()

np.save("phi.npy", phi)


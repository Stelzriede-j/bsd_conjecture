
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.interpolate import interp1d

# Parameters
Nx, Ny = 50, 50
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
mod_p = 17
injection_times = [0, 30, 60, 90, 120]
centers = [(10, 10), (40, 10), (10, 40), (40, 40), (25, 25)]
twist_amplitude = 0.15
injection_radius = 10.0
threshold = 5

# Initialize field
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

# Evolve field with 5 staggered injections
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

# Mod projection + lock zone detection
mod_field = np.round(phi * 1000).astype(int) % mod_p
mask = np.zeros_like(mod_field, dtype=bool)
for x in range(1, Nx - 1):
    for y in range(1, Ny - 1):
        patch = mod_field[x-1:x+2, y-1:y+2]
        if np.count_nonzero(patch == mod_field[x, y]) >= threshold:
            mask[x, y] = True

labeled_array, num_features = label(mask, structure=np.ones((3, 3), dtype=int))
energy_density = 0.5 * velocity**2 + lambda_fixed * (np.abs(phi))**n_fixed

# Synthetic L-function setup (exp decay)
prime_set = [13, 17, 19]
approx_entropies = {
    13: 7.46,
    17: 7.48,
    19: 7.49
}
resonance_weights = [1 / approx_entropies[p] for p in prime_set]

s_values = np.linspace(0.5, 2.0, 200)
L_synthetic = []
for s in s_values:
    value = sum(R * np.exp(-p**s) for R, p in zip(resonance_weights, prime_set))
    L_synthetic.append(value)

# Derivatives at s = 1 using NumPy version
def numerical_derivative(f, x, dx=1e-5, order=1):
    if order == 1:
        return (f(x + dx) - f(x - dx)) / (2 * dx)
    elif order == 2:
        return (f(x + dx) - 2*f(x) + f(x - dx)) / (dx ** 2)
    else:
        raise ValueError("Only first and second derivatives supported.")

L_interp = interp1d(s_values, L_synthetic, kind='cubic', fill_value="extrapolate")
s_target = 1.0
L_value = L_interp(s_target)
L_prime = numerical_derivative(L_interp, s_target, order=1)
L_double_prime = numerical_derivative(L_interp, s_target, order=2)

# Plot L-function
plt.figure(figsize=(10, 5))
plt.plot(s_values, L_synthetic, color='forestgreen')
plt.axhline(0, linestyle=':', color='black')
plt.title("Synthetic L-function (Rank Expansion Field)")
plt.xlabel("s")
plt.ylabel("L_field(s)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Energy per lock zone
zone_energies = {}
for label_id in range(1, num_features + 1):
    mask_zone = (labeled_array == label_id)
    total_energy = np.sum(energy_density[mask_zone])
    zone_energies[label_id] = total_energy

# Plot energy spectrum
sorted_zones = sorted(zone_energies.items(), key=lambda x: x[1], reverse=True)
zone_ids = [z[0] for z in sorted_zones]
energies = [z[1] for z in sorted_zones]

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(energies) + 1), energies, color='slateblue')
plt.xlabel("Lock Zone ID (sorted by energy)")
plt.ylabel("Twist Energy")
plt.title("Energy Distribution per Lock Zone (Rank Expansion)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Output diagnostics
print(f"L(1) = {L_value}")
print(f"L'(1) = {L_prime}")
print(f"L''(1) = {L_double_prime}")

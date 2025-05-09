# Setup: twist field evolution under different (lambda, n) pairs, measure entropy at mod p=13
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.fft import fft2, fftshift
import scipy.stats as stats
import seaborn as sns

# Parameters to scan
lambdas = [0.5, 1.0, 1.5, 2.0]
ns = [1.5, 2.0, 2.5, 3.0]
mod_p = 17
Nx, Ny = 50, 50
fft_threshold = 20
dt = 0.05
damping = 0.95

# Grid to store entropy values
entropy_grid = np.zeros((len(lambdas), len(ns)))

# Loop over parameter grid
for i, lam in enumerate(lambdas):
    for j, n_val in enumerate(ns):
        # Initialize field
        phi = np.random.uniform(-0.1, 0.1, (Nx, Ny))
        velocity = np.zeros((Nx, Ny))

        # Evolve field with current (lambda, n)
        for _ in range(150):
            new_phi = phi.copy()
            for x in range(1, Nx - 1):
                for y in range(1, Ny - 1):
                    laplacian = (
                        phi[x+1, y] + phi[x-1, y] + phi[x, y+1] + phi[x, y-1] - 4 * phi[x, y]
                    )
                    force = -np.sign(phi[x, y]) * lam * n_val * (abs(phi[x, y])) ** (n_val - 1)
                    acceleration = laplacian + force
                    velocity[x, y] = damping * (velocity[x, y] + dt * acceleration)
                    new_phi[x, y] += velocity[x, y]
            phi = new_phi

        # Compute entropy of FFT of mod p field
        mod_field = np.round(phi * 1000).astype(int) % mod_p
        fft_result = fftshift(np.abs(fft2(mod_field)))
        spectrum = fft_result.flatten()
        spectrum /= np.sum(spectrum) + 1e-12
        entropy = stats.entropy(spectrum)
        entropy_grid[i, j] = entropy

# Plot heatmap of entropy landscape
plt.figure(figsize=(8, 6))
sns.heatmap(entropy_grid, xticklabels=ns, yticklabels=lambdas, annot=True, fmt=".3f", cmap="viridis")
plt.title(f"Spectral Entropy at mod {mod_p} vs (λ, n)")
plt.xlabel("n")
plt.ylabel("λ")
plt.tight_layout()
plt.show()
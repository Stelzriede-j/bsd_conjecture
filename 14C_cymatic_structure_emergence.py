# Re-run full 3D cymatic temple surface after reset

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift
from mpl_toolkits.mplot3d import Axes3D

# Parameters for 2D field
Nx, Ny = 200, 200
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Sweep frequencies and collect structure scores
frequencies = np.linspace(1, 60, 100)
structure_scores = []

def detect_structure(pattern):
    spectrum = np.abs(fftshift(fft2(pattern)))
    center = spectrum[Nx//2 - 10:Nx//2 + 10, Ny//2 - 10:Ny//2 + 10]
    return np.sum(center) / np.sum(spectrum)

for f in frequencies:
    pattern = np.sin(2 * np.pi * f * R)
    smoothed = gaussian_filter(pattern, sigma=1.2)
    score = detect_structure(smoothed)
    structure_scores.append(score)

# Build 3D radial temple from structure scores
grid_size = 50
X3D, Y3D = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
Z3D = np.zeros_like(X3D)

# Radially map structure scores into Z3D
for i in range(grid_size):
    for j in range(grid_size):
        r = np.sqrt(X3D[i, j]**2 + Y3D[i, j]**2)
        f_index = int(np.clip(r * (len(structure_scores) - 1), 0, len(structure_scores) - 1))
        Z3D[i, j] = structure_scores[f_index]

# Plot 3D surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X3D, Y3D, Z3D, cmap='viridis', edgecolor='none')
ax.set_title("Cymatic Temple: Radial Structure Emergence")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Structure Score")
plt.tight_layout()
plt.show()

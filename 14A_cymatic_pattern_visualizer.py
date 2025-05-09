
# Cymatic Pattern Visualizer for Fixed Frequency Fields
# Author: Jacob Stelzriede (with OpenAI)
# April 2025

"""
This script simulates a scalar wave pattern at a fixed frequency and visualizes
the resulting field in 2D, revealing cymatic-style nodal patterns and waveforms.

Use this to:
- Explore real cymatic geometries like 432 Hz, 440 Hz, 528 Hz
- Visualize standing wave patterns as rings, nodes, and structure zones
- Compare wave field symmetry and nodal layout across different frequencies

Parameters:
- frequency: fixed frequency to visualize
- Nx, Ny: field resolution (recommended: 200 x 200 for clean detail)
- smoothing: optional Gaussian blur (default 1.2)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

FREQUENCY = 5555

def plot_cymatic_pattern(
    frequency=FREQUENCY,
    Nx=400,  # increased resolution
    Ny=400,
    smoothing=1.2,
    contour_cmap='viridis',  # new color map for left graph
    nodal_cmap='Blues'       # color scale for nodal grains
):
    # Grid setup
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Generate field pattern
    wave_pattern = np.sin(2 * np.pi * frequency * R)
    pattern_smooth = gaussian_filter(wave_pattern, sigma=smoothing)

    # Compute nodal density (how close to zero)
    nodal_intensity = 1 - np.clip(np.abs(pattern_smooth) / 0.05, 0, 1)

    # Plot 2D field as contour and enhanced nodal map
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Contour (wave field)
    axs[0].contourf(X, Y, pattern_smooth, levels=200, cmap=contour_cmap)
    axs[0].set_title(f"Cymatic Wave Pattern @ {frequency} Hz", fontsize=12)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_aspect('equal')

    # Nodal intensity visualization
    nodal_plot = axs[1].imshow(nodal_intensity, cmap=nodal_cmap, extent=[-1, 1, -1, 1], origin='lower')
    axs[1].set_title(f"Nodal Density Map @ {frequency} Hz", fontsize=12)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    fig.colorbar(nodal_plot, ax=axs[1], shrink=0.8, label="Nodal Intensity")

    plt.tight_layout()
    plt.show()

# Example usage with enhanced visuals
plot_cymatic_pattern(frequency=FREQUENCY)



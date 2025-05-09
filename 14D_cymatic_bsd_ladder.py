
# BSD Phase 14D: Cymatic Mass Gap Ladder & BSD Analog L-function
# Author: Jacob Stelzriede (with OpenAI)
# April 2025

"""
This script loads or re-runs a cymatic frequency sweep (as in Phase 14B),
then visualizes:
- Structure score vs frequency
- Cumulative symbolic mass vs frequency (analog of BSD L-function)
- Delta-f gap spacing between resonance lock-ins

Requires: cymatic_resonance_sweep() function (or import results from 14B)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift

# Cymatic resonance sweep (reconstructed from 14B)
def cymatic_resonance_sweep(freq_start=1, freq_stop=10000, num_steps=2000, threshold=0.055, smoothing=1.2):
    Nx, Ny = 100, 100
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    frequencies = np.linspace(freq_start, freq_stop, num_steps)
    lock_frequencies = []
    symbolic_scores = []
    delta_f = []

    previous_lock = None

    def detect_structure(pattern):
        spectrum = np.abs(fftshift(fft2(pattern)))
        center = spectrum[Nx//2 - 10:Nx//2 + 10, Ny//2 - 10:Ny//2 + 10]
        return np.sum(center) / np.sum(spectrum)

    for f in frequencies:
        pattern = np.sin(2 * np.pi * f * R)
        smoothed = gaussian_filter(pattern, sigma=smoothing)
        score = detect_structure(smoothed)

        if score > threshold:
            lock_frequencies.append(f)
            symbolic_scores.append(score)
            if previous_lock is not None:
                delta_f.append(f - previous_lock)
            previous_lock = f

    return lock_frequencies, symbolic_scores, delta_f

# Run sweep
locks, scores, gaps = cymatic_resonance_sweep()

# Plot symbolic structure vs frequency
plt.figure(figsize=(10, 4))
plt.plot(locks, scores, label="Symbolic Score per Lock-In", color='purple')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Structure Score")
plt.title("Symbolic Mass vs Frequency")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot cumulative BSD-style ladder
cumulative_mass = np.cumsum(scores)
plt.figure(figsize=(10, 4))
plt.plot(locks, cumulative_mass, label="Cumulative Symbolic Mass", color='green')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Cumulative Score")
plt.title("Cymatic BSD Analog – Symbolic Emergence Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot delta-f (mass gap) histogram
plt.figure(figsize=(6, 4))
plt.hist(gaps, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Delta-f between Lock-ins")
plt.ylabel("Frequency")
plt.title("Mass Gap Histogram (Δf)")
plt.grid(True)
plt.tight_layout()
plt.show()

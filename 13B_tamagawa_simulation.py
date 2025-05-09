
# Tamagawa Simulation via Mod-p Resonance Loss
# Author: Jacob Stelzriede (with OpenAI)
# April 2025

"""
This script runs a twist-compression field simulation across a list of primes
(mod-p projections) to identify symbolic degradation. The resulting field
resonance scores are used to estimate symbolic Tamagawa numbers for each prime.

Method:
- Inject twist field
- Apply mod-p projection for each prime
- Count symbolic lock zones or resonance energy (structure score)
- Normalize scores and compute loss = expected / actual

The result is a symbolic estimate of Tamagawa number c_p for each prime.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift

# Field parameters
Nx, Ny = 100, 100
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
injection_frequency = 20
primes = [3, 5, 7, 11, 13, 17, 19, 23, 29]

def simulate_structure_score_mod_p(p, smoothing=1.2):
    field = np.sin(2 * np.pi * injection_frequency * R)
    field = np.round(field * 1000).astype(int) % p
    field = field.astype(float) / p

    smoothed = gaussian_filter(field, sigma=smoothing)
    spectrum = np.abs(fftshift(fft2(smoothed)))
    center = spectrum[Nx//2 - 10:Nx//2 + 10, Ny//2 - 10:Ny//2 + 10]
    score = np.sum(center) / np.sum(spectrum)
    return score

# Run simulation across all primes
scores = {p: simulate_structure_score_mod_p(p) for p in primes}
max_score = max(scores.values())
tamagawa_estimates = {p: round(max_score / s, 3) if s > 0 else float('inf') for p, s in scores.items()}

# Output
for p in primes:
    print(f"Prime {p}: Score = {scores[p]:.4f}, Estimated c_p = {tamagawa_estimates[p]}")
